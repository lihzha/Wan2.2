"""End-to-end training for the rank-1 trajectory adaptor.

The Wan TI2V-5B DiT and VAE are frozen. Gradients flow only into the
adaptor (`models/trajectory_adaptor.py`). Loss is Wan's own flow-matching
denoising loss with stochastic σ and ε at every step — this naturally
averages over the per-seed basin noise that broke the factor-regression
plan (see docs/adaptor_design.md §1.5–§1.6).

For each training step:
    z_0     = z_video                                        (precomputed VAE encoding)
    ε       ~ N(0, I)
    σ_t     = sigma_schedule[t]                              (t sampled per example)
    z_t     = (1 − σ_t) z_0 + σ_t ε
    z_t     = first_frame_pin(z_t, z_I0)                    (Wan's i2v pin)
    C_traj  = adaptor(actions, z_I0)                         [B, N, L, D]
    C_t     = C_traj[b, t[b]]                                [B, L, D]
    (CFG dropout: with prob p, replace C_t with T5("")  )
    v_pred  = DiT(z_t, σ_t, context=C_t)
    v_target= ε − z_0
    loss    = MSE(v_pred, v_target)   masked to mask2 == 1   (i.e. non-frame-0)

Usage:
  python scripts/train_adaptor.py \\
      --triplets_root data/triplets \\
      --adaptor_init runs/batch_inv_positive/factorization/adaptor_init.pt \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --output_dir runs/adaptor_train_v0 \\
      --train_names triplet_0 triplet_1 ... triplet_44 \\
      --val_names   triplet_45 ... triplet_49 \\
      --batch_size 1 --total_steps 50000

Notes
-----
* Sec. 3.4: batch_size 1–4; DiT memory dominates. Single-GPU H100 with
  bf16 + grad-checkpointing comfortably runs batch_size=1; larger needs
  more VRAM headroom.
* CFG dropout: with prob `cfg_dropout`, replace C_t with T5("") so the
  adaptor doesn't crowd out the unconditional pathway at inference (CFG
  semantics are preserved only if the model has seen both forks).
* The optimization basin from positive_inversion contributed roughly half
  the trajectory direction (Sec. 1.5). End-to-end with stochastic ε/σ
  averages over that noise — we treat μ, β as warm-starts, not labels.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import itertools
import json
import os
import random
import sys
import time
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import DataLoader, Dataset

import embedding_search as ES
from models.trajectory_adaptor import TrajectoryAdaptor
from models.z_init_predictor import ZInitPredictor
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    # Data.
    p.add_argument("--triplets_root", default="data/triplets")
    p.add_argument(
        "--adaptor_init",
        default="runs/batch_inv_positive/factorization/adaptor_init.pt",
    )
    p.add_argument(
        "--train_names", nargs="*", default=None,
        help="Triplet folder names to train on. If omitted, uses all "
        "triplets except --val_names.",
    )
    p.add_argument(
        "--val_names", nargs="*",
        default=["45", "46", "47", "48", "49"],
        help="Held-out triplet names (folder basenames under triplets_root, "
        "or under --val_triplets_root when that is set). Default = last 5 "
        "(matches §3.6 milestone M2). Pass an empty list (no values) to skip "
        "validation.",
    )
    p.add_argument(
        "--val_triplets_root", default=None,
        help="If set, validation samples are loaded from this directory "
        "instead of --triplets_root, and ALL of its subdirs are used as the "
        "val set (so --val_names is ignored and train uses every subdir of "
        "--triplets_root). Use this with the DROID cache, which has separate "
        "train/ and val/ splits.",
    )
    p.add_argument(
        "--overfit_one", default=None,
        help="If set to a triplet folder name (e.g. '0'), train on JUST that "
        "single triplet (M1 sanity check).",
    )

    # Model.
    p.add_argument("--ckpt_dir", required=True,
                   help="Wan2.2 TI2V-5B checkpoint dir (DiT + T5 + VAE).")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--action_dim", type=int, default=8)
    p.add_argument("--action_len", type=int, default=32)
    p.add_argument("--action_repr", choices=["delta", "raw"], default="delta")
    p.add_argument("--b_head_init_scale", type=float, default=1e-4)
    # Architecture variants (see models/trajectory_adaptor.py / docs §3.12).
    p.add_argument("--arch_fusion", choices=["concat", "cross_attn"],
                   default="concat",
                   help="concat = pooled image + MLP action (original); "
                   "cross_attn = token-level image (2-D pos) + action, "
                   "query cross-attention.")
    p.add_argument("--arch_head", choices=["rank1", "rankk", "perstep"],
                   default="rank1",
                   help="rank1 = single shared dir (original); rankk = sum of "
                   "K dirs; perstep = general per-step residual MLP.")
    p.add_argument("--rank_k", type=int, default=4,
                   help="K for --arch_head rankk.")
    p.add_argument("--n_xattn_layers", type=int, default=2)
    p.add_argument("--step_emb_dim", type=int, default=128)

    # Sigma schedule (must match inversion to make μ, β meaningful).
    p.add_argument("--sampling_steps", type=int, default=25,
                   help="N. The adaptor's μ, β are indexed by step 0..N-1.")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sigma_schedule_path", default=None,
                   help="Optional .pt containing a custom sigmas tensor "
                   "[N+1], or a dict with key 'sigmas'. Overrides "
                   "--sampling_steps/--shift for the actual rollout grid.")
    p.add_argument(
        "--t_sampling", choices=["uniform", "logit_normal"], default="uniform",
        help="How to draw step index t. Logit-normal maps continuous u ~ "
        "Normal(0, 1) -> σ = sigmoid(u), then picks nearest discrete sigma. "
        "This emphasizes mid-σ steps (consistent with flow-matching norms).",
    )
    p.add_argument("--logit_normal_mu", type=float, default=0.0)
    p.add_argument("--logit_normal_sigma", type=float, default=1.0)

    # Optim.
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--total_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_mu_beta", type=float, default=None,
                   help="Separate LR for μ, β (default: lr/10).")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--lr_min_ratio", type=float, default=0.1,
                   help="Cosine decay floor (lr * ratio).")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_guide_scale", type=float, default=5.0,
                   help="CFG scale w the loss is consistent with: the "
                   "prediction regressed to the flow target is "
                   "v_uncond + w·(v_cond − v_uncond). MUST match the scale the "
                   "μ/β warm-start was inverted at (adaptor_init.pt → w=5) and "
                   "the eval guide_scale. w=1 trains the bare conditional "
                   "velocity (only valid with a w=1 warm-start). See §3.13.")
    p.add_argument("--cfg_dropout", type=float, default=0.0,
                   help="Probability of replacing C_t with T5(''). NOTE: a "
                   "no-op for the adaptor because the DiT is frozen — a "
                   "dropped step produces no adaptor gradient. Left at 0.")
    p.add_argument("--freeze_mu_beta_steps", type=int, default=0,
                   help="Hold μ, β frozen for the first N optimizer steps.")
    p.add_argument("--loss_type", choices=["denoise", "rollout", "context"],
                   default="denoise",
                   help="denoise = one-step flow-matching loss; rollout = "
                   "backprop through the same Euler replay used by eval and "
                   "minimize final latent MSE; context = directly regress "
                   "the adaptor's full C_traj to --target_context_path. "
                   "Rollout/context are intended for one-sample overfit/debug runs.")
    p.add_argument("--rollout_uncond_grad", action="store_true",
                   help="For --loss_type rollout, also backprop through the "
                   "unconditional CFG branch. Default treats it as a frozen "
                   "control variate, matching the diagnostic optimizer.")
    p.add_argument("--init_context_path", default=None,
                   help="Optional .pt with positive_embeddings/tensor [N,L,D] "
                   "used to overwrite adaptor.mu before training. Useful for "
                   "one-sample rollout overfit from a clip-specific oracle.")
    p.add_argument("--target_context_path", default=None,
                   help="Optional .pt with positive_embeddings/tensor [N,L,D] "
                   "used as a direct auxiliary target for the adaptor's full "
                   "C_traj. Intended for one-sample oracle-overfit debugging.")
    p.add_argument("--context_loss_weight", type=float, default=0.0,
                   help="Weight for --target_context_path MSE. Defaults off.")
    p.add_argument("--zero_residual_on_init_context", action="store_true",
                   help="When --init_context_path is set, also zero the "
                   "adaptor residual head so C_t starts exactly at mu_t.")
    p.add_argument("--train_mu_only", action="store_true",
                   help="Freeze all non-mu/beta adaptor parameters. Useful "
                   "for rollout one-sample debugging, where mu is already the "
                   "full per-step context tensor and residual heads can "
                   "destabilize the first few updates.")

    # Misc.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fixed_noise", action="store_true",
        help="Use ONE fixed ε₀ (seeded by --seed, the same noise eval's "
        "z_init uses) for every training step instead of fresh randn. Makes "
        "the (action,I_0)→C_t target a well-defined deterministic function "
        "(no noise basin to marginalize — see §3.14), matching the "
        "deterministic prediction task and the fixed-seed eval. Recommended "
        "for this task; turns the adaptor into an amortized inversion.")
    p.add_argument("--fixed_z_init_path", default=None,
                   help="Optional .pt file containing the exact initial "
                   "latent [C,F,H,W] to use for --loss_type rollout. This "
                   "removes RNG as a train/eval variable for one-sample "
                   "overfit diagnostics.")
    p.add_argument("--predict_z_init", action="store_true",
                   help="Train an initial-latent predictor and use its output "
                   "as the rollout sampler z_init. The predictor is initialized "
                   "as a zero delta on top of the fixed seed-noise base.")
    p.add_argument("--z_init_hidden", type=int, default=512)
    p.add_argument("--z_init_lr", type=float, default=None,
                   help="Learning rate for --predict_z_init parameters "
                   "(default: --lr).")
    p.add_argument("--target_z_init_path", default=None,
                   help="Optional .pt containing oracle z_init for an auxiliary "
                   "masked MSE target. For one-sample overfit this is usually "
                   "runs/droid_inv/train/<episode>/positive_embeddings.pt.")
    p.add_argument("--z_init_loss_weight", type=float, default=0.0,
                   help="Auxiliary target-z_init MSE weight. Rollout loss "
                   "still provides the main end-to-end signal.")
    p.add_argument("--z_init_pretrain_steps", type=int, default=0,
                   help="With --predict_z_init and --target_z_init_path, run "
                   "this many initial steps on z_init MSE only before enabling "
                   "the rollout objective.")
    p.add_argument("--freeze_z_init_after_pretrain", action="store_true",
                   help="Freeze the z_init predictor once --z_init_pretrain_steps "
                   "has completed. Useful when supervised z_init reaches the "
                   "right basin and rollout gradients would otherwise move it "
                   "out of distribution.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--val_eps_repeats", type=int, default=8,
                   help="Number of (ε, t) draws averaged per validation "
                   "example. With --fixed_noise, ε is fixed so set this to 1.")
    p.add_argument("--ckpt_interval", type=int, default=5000)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
        help="Wrap each DiT block with torch.utils.checkpoint. ~2.5× memory "
        "savings at ~30%% wall-time cost. Required for batch_size > 1 on H100.",
    )
    p.add_argument("--no_gradient_checkpointing", action="store_false",
                   dest="gradient_checkpointing")
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TripletLatentDataset(Dataset):
    """Loads precomputed (actions, z_I0, z_video) per triplet.

    Run scripts/precompute_features.py first to populate z_I0.pt /
    z_video.pt under data/triplets/<i>/. Caching is keyed on file mtime
    so this dataset never re-encodes.
    """

    def __init__(self, triplets_root: str, names: Sequence[str]):
        self.records = []
        for n in names:
            tdir = os.path.join(triplets_root, n)
            if not os.path.isdir(tdir):
                # Allow "triplet_0" style names too — strip prefix.
                alt = n[len("triplet_"):] if n.startswith("triplet_") else n
                tdir = os.path.join(triplets_root, alt)
            actions_p = os.path.join(tdir, "actions.npy")
            z_I0_p = os.path.join(tdir, "z_I0.pt")
            z_video_p = os.path.join(tdir, "z_video.pt")
            for path in (actions_p, z_I0_p, z_video_p):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing {path}. Run scripts/precompute_features.py first."
                    )
            self.records.append({
                "name": os.path.basename(tdir),
                "actions": actions_p,
                "z_I0": z_I0_p,
                "z_video": z_video_p,
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        actions = torch.from_numpy(np.load(r["actions"])).float()      # [T, A]
        z_I0 = torch.load(r["z_I0"], map_location="cpu", weights_only=False).float()
        z_video = torch.load(r["z_video"], map_location="cpu", weights_only=False).float()
        return {
            "name": r["name"],
            "actions": actions,
            "z_I0": z_I0,
            "z_video": z_video,
        }


def collate(batch):
    return {
        "name": [b["name"] for b in batch],
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
        "z_I0": torch.stack([b["z_I0"] for b in batch], dim=0),
        "z_video": torch.stack([b["z_video"] for b in batch], dim=0),
    }


# ---------------------------------------------------------------------------
# Pipeline + sigma schedule
# ---------------------------------------------------------------------------


def build_wan_pipeline(ckpt_dir: str) -> WanTI2V:
    """Same pipeline used by scripts/embedding_search.py, but everything
    stays in inference mode (frozen)."""
    print(f"[train] loading Wan TI2V-5B from {ckpt_dir}…")
    pipe = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    # Ensure everything is frozen / eval.
    pipe.model.eval().requires_grad_(False)
    pipe.vae.model.eval().requires_grad_(False)
    pipe.text_encoder.model.eval().requires_grad_(False)
    print(f"[train] loaded. device={pipe.device} param_dtype={pipe.param_dtype}")
    return pipe


def load_sigma_schedule(path: str, device) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    sigmas = d["sigmas"] if isinstance(d, dict) else d
    sigmas = sigmas.float().to(device)
    if sigmas.dim() != 1 or sigmas.numel() < 2:
        raise ValueError(f"expected sigmas [N+1], got {tuple(sigmas.shape)}")
    return sigmas


def build_sigma_schedule(
    pipe: WanTI2V,
    sampling_steps: int,
    shift: float,
    sigma_schedule_path: str | None = None,
):
    """Mirror scripts/embedding_search.py:_build_inversion_geometry — same
    sigma sequence used during positive_inversion, so the adaptor's μ, β
    align with the σ-indexed embeddings the warm-start was derived from."""
    if sigma_schedule_path is not None:
        sigmas = load_sigma_schedule(sigma_schedule_path, pipe.device)
        if sigmas.numel() != sampling_steps + 1:
            raise ValueError(
                f"--sigma_schedule_path length {sigmas.numel()} does not match "
                f"--sampling_steps + 1 = {sampling_steps + 1}")
        return sigmas
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=pipe.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(sampling_steps, device=pipe.device, shift=shift)
    sigmas = scheduler.sigmas.to(pipe.device).float()   # length N+1
    return sigmas


def build_latent_geometry(pipe: WanTI2V, z_video_shape):
    """Compute (seq_len, mask2) from the latent shape. mask2 has 0 at the
    first temporal slot (frame 0) and 1 elsewhere — Wan's i2v latent pin."""
    C, F_z, H_z, W_z = z_video_shape
    seq_len = (F_z * H_z * W_z) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int((seq_len + pipe.sp_size - 1) // pipe.sp_size) * pipe.sp_size
    mask2 = torch.ones(C, F_z, H_z, W_z, dtype=torch.float32, device=pipe.device)
    mask2[:, 0] = 0.0
    return seq_len, mask2


# ---------------------------------------------------------------------------
# DiT grad-checkpointing context manager (re-used from embedding_search.py)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def model_grad_checkpointing(pipe: WanTI2V):
    original = []
    for block in pipe.model.blocks:
        original.append(block.forward)

        def make_ckpt(orig_forward):
            def ckpt_forward(x, **kwargs):
                return checkpoint_utils.checkpoint(
                    orig_forward, x, use_reentrant=False, **kwargs)
            return ckpt_forward

        block.forward = make_ckpt(block.forward)
    try:
        yield
    finally:
        for block, orig in zip(pipe.model.blocks, original):
            block.forward = orig


# ---------------------------------------------------------------------------
# Timestep helpers
# ---------------------------------------------------------------------------


def build_timestep_tensor(
    sigmas_t_batch: torch.Tensor,    # [B], in [0, sigma_max]
    mask2_zero: torch.Tensor,        # [F_z, H_z, W_z]
    patch_size,                      # tuple (1, 2, 2)
    seq_len: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Build [B, seq_len] per-position timestep tensor that the Wan DiT
    expects. Mirrors WanTI2V.i2v_diff's `temp_ts` construction — positions
    where mask2 == 0 (frame 0) get timestep 0, others get σ · T_train."""
    B = sigmas_t_batch.shape[0]
    ts = sigmas_t_batch * num_train_timesteps                         # [B]
    pps = patch_size[1] * patch_size[2]
    # mask2_zero subsampled by patch in HxW, then flattened.
    # Wan's stride is patch_size[1], patch_size[2] (== 2,2) in H,W.
    m = mask2_zero[:, ::patch_size[1], ::patch_size[2]].flatten()      # [n_tok]
    n_tok = m.numel()
    per_tok = m[None, :] * ts[:, None]                                 # [B, n_tok]
    pad = ts[:, None].expand(-1, seq_len - n_tok).contiguous()          # [B, pad]
    return torch.cat([per_tok, pad], dim=1)                            # [B, seq_len]


def apply_first_frame_pin(latent, z_I0_full, mask2_full):
    """Wan i2v's `latent = (1 − mask2) · z_I0 + mask2 · latent`."""
    return (1.0 - mask2_full) * z_I0_full + mask2_full * latent


def load_context_tensor(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, torch.Tensor):
        ctx = d.float()
    elif "positive_embeddings" in d:
        pe = d["positive_embeddings"]
        ctx = torch.stack([e.float() for e in pe], dim=0) if isinstance(pe, list) else pe.float()
    else:
        raise KeyError(f"{path} has neither tensor content nor positive_embeddings")
    if ctx.dim() != 3:
        raise ValueError(f"expected context tensor [N,L,D], got {tuple(ctx.shape)}")
    return ctx


def load_z_init_tensor(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, torch.Tensor):
        z = d.float()
    elif "z_init" in d:
        z = d["z_init"].float()
    else:
        raise KeyError(f"{path} has neither tensor content nor z_init")
    if z.dim() != 4:
        raise ValueError(f"expected z_init [C,F,H,W], got {tuple(z.shape)}")
    return z


def masked_latent_mse(pred: torch.Tensor, target: torch.Tensor, mask2: torch.Tensor):
    """MSE over non-pinned latent positions."""
    if target.dim() == 4:
        target = target.unsqueeze(0).expand_as(pred)
    mask = mask2.unsqueeze(0).expand_as(pred)
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def zero_adaptor_residual(adaptor: TrajectoryAdaptor):
    """Make the adaptor output exactly mu at initialization."""
    if adaptor.head == "perstep":
        last = adaptor.residual_mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        return
    if hasattr(adaptor, "gamma_head"):
        torch.nn.init.zeros_(adaptor.gamma_head.weight)
        torch.nn.init.zeros_(adaptor.gamma_head.bias)
    if hasattr(adaptor, "b_head"):
        torch.nn.init.zeros_(adaptor.b_head.weight)
        torch.nn.init.zeros_(adaptor.b_head.bias)


# ---------------------------------------------------------------------------
# Step-index sampler
# ---------------------------------------------------------------------------


def sample_step_indices(
    batch_size: int,
    N: int,
    sigmas: torch.Tensor,           # [N+1]
    kind: str,
    ln_mu: float,
    ln_sigma: float,
    device,
) -> torch.Tensor:
    """Return [B] int64 indices into 0..N-1.

    "uniform"      : uniform on {0..N-1}
    "logit_normal" : draw u ~ Normal(ln_mu, ln_sigma), σ_target = sigmoid(u),
                     pick nearest discrete sigmas[0..N-1].
    """
    if kind == "uniform":
        return torch.randint(0, N, (batch_size,), device=device)
    if kind == "logit_normal":
        u = torch.randn(batch_size, device=device) * ln_sigma + ln_mu
        sigma_target = torch.sigmoid(u)                                   # [B] in (0, 1)
        # Nearest discrete σ_i for i in 0..N-1.
        candidates = sigmas[:N]                                            # [N]
        diff = (candidates[None, :] - sigma_target[:, None]).abs()         # [B, N]
        return diff.argmin(dim=1)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Forward step (training or validation)
# ---------------------------------------------------------------------------


def adaptor_denoising_loss(
    pipe: WanTI2V,
    adaptor: TrajectoryAdaptor,
    batch: dict,
    sigmas: torch.Tensor,           # [N+1]
    null_context: torch.Tensor,     # [L_null, 4096], T5("")
    seq_len: int,
    mask2: torch.Tensor,            # [C, F_z, H_z, W_z], full
    cfg_dropout: float,
    t_kind: str,
    ln_mu: float,
    ln_sigma: float,
    train: bool,
    train_guide_scale: float = 5.0,
    eps_override: Optional[torch.Tensor] = None,
    t_override: Optional[torch.Tensor] = None,
):
    """Returns (loss, dict_of_logs).

    When `train=True`, gradient checkpointing wraps the DiT and gradients
    flow into the adaptor. When False, runs under no_grad — used for
    validation.

    `train_guide_scale` (w) makes the loss CFG-consistent with inference:
    the prediction regressed to the flow target is the *CFG-combined*
    velocity v_uncond + w·(v_cond − v_uncond), exactly what the deployed
    sampler integrates. w must match the scale the μ/β warm-start (and the
    eval) use — the inversions that produced adaptor_init.pt ran at w=5, so
    the conditional contexts are "deflated" by ~1/w and only make sense
    amplified back up. Training at w=1 from a w=5 warm-start yields static
    video (the conditional velocity is ≈ unconditional). v_uncond does not
    depend on the adaptor, so it is computed once under no_grad (≈1 extra
    forward, no extra backward). See docs/adaptor_design.md §3.13.
    """
    device = pipe.device
    dtype = pipe.param_dtype

    actions = batch["actions"].to(device, non_blocking=True)           # [B, T, A]
    z_I0 = batch["z_I0"].to(device, non_blocking=True).float()         # [B, C, 1, H_z, W_z]
    z_video = batch["z_video"].to(device, non_blocking=True).float()   # [B, C, F_z, H_z, W_z]
    B = z_video.shape[0]
    C, F_z, H_z, W_z = z_video.shape[1:]

    # ---- broadcast z_I0 to full temporal length for the pin ----
    z_I0_full = z_I0.expand(-1, -1, F_z, -1, -1).contiguous()           # [B, C, F_z, H_z, W_z]

    # ---- pick step index t per example ----
    N = adaptor.num_steps
    if t_override is None:
        t = sample_step_indices(B, N, sigmas, t_kind, ln_mu, ln_sigma, device)
    else:
        t = t_override.to(device)
    sigma_t = sigmas[t]                                                # [B]

    # ---- noise + forward path ----
    if eps_override is None:
        eps = torch.randn_like(z_video)
    else:
        eps = eps_override.to(device)
    sigma_b = sigma_t.view(B, 1, 1, 1, 1)
    z_t = (1.0 - sigma_b) * z_video + sigma_b * eps
    mask2_b = mask2.unsqueeze(0)                                       # [1, C, F_z, H_z, W_z]
    z_t = (1.0 - mask2_b) * z_I0_full + mask2_b * z_t                  # first-frame pin

    # ---- adaptor → per-step context, then gather t-th step ----
    C_traj = adaptor(actions, z_I0)                                    # [B, N, L, D]
    idx = t.view(B, 1, 1, 1).expand(-1, 1, C_traj.shape[2], C_traj.shape[3])
    C_t = C_traj.gather(1, idx).squeeze(1)                             # [B, L, D]

    # ---- CFG dropout (mask whole rows). Training only. ----
    if train and cfg_dropout > 0.0:
        drop_mask = (torch.rand(B, device=device) < cfg_dropout)        # [B]
        if drop_mask.any():
            null_b = null_context.unsqueeze(0).expand(B, -1, -1).to(C_t.dtype)
            # Tile null_context to L if shapes differ.
            if null_b.shape[1] != C_t.shape[1]:
                # Truncate or pad to L. For Wan's text_embedding, both work
                # since it pads to text_len anyway, but we keep shapes equal
                # so the gather stays clean.
                Lt = C_t.shape[1]
                if null_b.shape[1] > Lt:
                    null_b = null_b[:, :Lt]
                else:
                    pad = null_b.new_zeros(B, Lt - null_b.shape[1], null_b.shape[2])
                    null_b = torch.cat([null_b, pad], dim=1)
            drop_mask_b = drop_mask.view(B, 1, 1).to(C_t.dtype)
            C_t = (1.0 - drop_mask_b) * C_t + drop_mask_b * null_b

    # ---- build timestep tensor [B, seq_len] ----
    mask2_zero = mask2[0]                                              # [F_z, H_z, W_z]
    timestep_tensor = build_timestep_tensor(
        sigma_t, mask2_zero, pipe.patch_size, seq_len, pipe.num_train_timesteps,
    )

    # ---- DiT forward(s) (frozen, autocast bf16, optional grad-ckpt) ----
    x_list = [z_t[b] for b in range(B)]
    ctx_list = [C_t[b] for b in range(B)]

    def _dit(context_list, with_grad):
        cm = contextlib.nullcontext() if with_grad else torch.no_grad()
        with cm, torch.amp.autocast("cuda", dtype=dtype):
            out = pipe.model(
                x_list, t=timestep_tensor, context=context_list, seq_len=seq_len)
        return torch.stack([v.float() for v in out], dim=0)   # [B, C, F_z, H_z, W_z]

    # Conditional velocity carries the adaptor gradient (only when training).
    v_cond = _dit(ctx_list, with_grad=train)

    use_cfg = abs(train_guide_scale - 1.0) > 1e-6
    if use_cfg:
        # Unconditional velocity: frozen DiT + T5(""), independent of the
        # adaptor, so no_grad (no extra backward graph). One extra forward.
        null_list = [null_context for _ in range(B)]
        v_uncond = _dit(null_list, with_grad=False)
        v_pred = v_uncond + train_guide_scale * (v_cond - v_uncond)
    else:
        v_pred = v_cond

    # ---- v_target = ε − z_0 (flow-matching convention) ----
    v_target = eps - z_video

    # ---- masked MSE: only non-frame-0 positions count. ----
    # mask2_b is [1, C, F_z, H_z, W_z] with frame 0 zeroed, so mask2_b.sum()
    # ALREADY counts the channel dim. The valid element count is therefore
    # mask2_b.sum() * B (batch), NOT * C. (Earlier this multiplied by C, which
    # under-reported the loss ~48× and masked a large velocity error.)
    mse = (v_pred - v_target).pow(2) * mask2_b
    n_valid = mask2_b.sum().clamp_min(1.0) * v_pred.shape[0]            # * B
    loss = mse.sum() / n_valid

    logs = {
        "loss": float(loss.item()),
        "sigma_mean": float(sigma_t.mean().item()),
        "t_mean": float(t.float().mean().item()),
        "v_pred_l2": float(v_pred.detach().norm().item()),
        "v_target_l2": float(v_target.detach().norm().item()),
    }
    return loss, logs


def adaptor_rollout_loss(
    pipe: WanTI2V,
    adaptor: TrajectoryAdaptor,
    batch: dict,
    sigmas: torch.Tensor,
    null_context: torch.Tensor,
    seq_len: int,
    mask2: torch.Tensor,
    train_guide_scale: float,
    eps_override: Optional[torch.Tensor] = None,
    uncond_grad: bool = False,
):
    """Closed-loop replay loss used for one-sample overfit/debug.

    This optimizes the actual deployed objective: start from eval's initial
    noise, integrate all sampler steps with adaptor-produced contexts, and
    minimize final latent MSE to z_video. It is intentionally batch_size=1 for
    memory and because the target use here is a hard feasibility check.
    """
    device = pipe.device
    actions = batch["actions"].to(device, non_blocking=True)
    z_I0 = batch["z_I0"].to(device, non_blocking=True).float()
    z_video_b = batch["z_video"].to(device, non_blocking=True).float()
    if z_video_b.shape[0] != 1:
        raise ValueError("--loss_type rollout currently requires batch_size=1")

    z_video = z_video_b[0]
    C, F_z, H_z, W_z = z_video.shape
    z_I0_full = z_I0[0].expand(-1, F_z, -1, -1).contiguous()
    z_init = (
        eps_override[0].to(device).float()
        if eps_override is not None else torch.randn_like(z_video)
    )

    C_traj = adaptor(actions, z_I0)[0]                                  # [N, L, D]
    mask2_zero = mask2[0]
    z = ES._apply_first_frame_pin(z_init, z_I0_full, mask2)
    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i].item()
        sigma_ip1 = sigmas[i + 1].item()
        v_cond = ES._dit_velocity(
            pipe, z, sigma_i, [C_traj[i]], seq_len, mask2_zero, pipe.param_dtype)
        if abs(train_guide_scale - 1.0) > 1e-6:
            if uncond_grad:
                v_uncond = ES._dit_velocity(
                    pipe, z, sigma_i, [null_context], seq_len,
                    mask2_zero, pipe.param_dtype)
            else:
                with torch.no_grad():
                    v_uncond = ES._dit_velocity(
                        pipe, z.detach(), sigma_i, [null_context], seq_len,
                        mask2_zero, pipe.param_dtype)
            v = v_uncond + train_guide_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = ES._apply_first_frame_pin(
            z + (sigma_ip1 - sigma_i) * v, z_I0_full, mask2)

    loss = F.mse_loss(z, z_video)
    logs = {
        "loss": float(loss.item()),
        "latent_mse": float(loss.item()),
        "ctx_norm": float(C_traj.detach().norm(dim=-1).mean().item()),
        "z_pred_std": float(z.detach().std().item()),
        "z_target_std": float(z_video.detach().std().item()),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_validation(
    pipe, adaptor, val_loader, sigmas, null_context, seq_len, mask2,
    val_eps_repeats, t_kind, ln_mu, ln_sigma, train_guide_scale,
    fixed_eps=None,
):
    # With a fixed ε₀ there is no noise to average over — one pass suffices.
    repeats = 1 if fixed_eps is not None else val_eps_repeats
    losses = []
    for batch in val_loader:
        bsz = batch["z_video"].shape[0]
        eps_ovr = (fixed_eps.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
                   if fixed_eps is not None else None)
        for _ in range(repeats):
            loss, _ = adaptor_denoising_loss(
                pipe=pipe, adaptor=adaptor, batch=batch, sigmas=sigmas,
                null_context=null_context, seq_len=seq_len, mask2=mask2,
                cfg_dropout=0.0, t_kind=t_kind, ln_mu=ln_mu, ln_sigma=ln_sigma,
                train=False, train_guide_scale=train_guide_scale,
                eps_override=eps_ovr,
            )
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def run_rollout_validation(
    pipe, adaptor, val_loader, sigmas, null_context, seq_len, mask2,
    train_guide_scale, fixed_eps=None, z_init_predictor=None,
):
    losses = []
    for batch in val_loader:
        bsz = batch["z_video"].shape[0]
        if bsz != 1:
            raise ValueError("--loss_type rollout validation requires batch_size=1")
        eps_ovr = (fixed_eps.unsqueeze(0) if fixed_eps is not None else None)
        if z_init_predictor is not None:
            actions_z = batch["actions"].to(pipe.device, non_blocking=True)
            z_I0_z = batch["z_I0"].to(pipe.device, non_blocking=True).float()
            eps_ovr = z_init_predictor(actions_z, z_I0_z, eps_ovr)
        loss, _ = adaptor_rollout_loss(
            pipe=pipe, adaptor=adaptor, batch=batch, sigmas=sigmas,
            null_context=null_context, seq_len=seq_len, mask2=mask2,
            train_guide_scale=train_guide_scale, eps_override=eps_ovr,
            uncond_grad=False,
        )
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Cosine LR
# ---------------------------------------------------------------------------


def cosine_lr_factor(step: int, total: int, warmup: int, min_ratio: float):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1.0 + np.cos(np.pi * prog))
    return min_ratio + (1.0 - min_ratio) * cos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _sorted_subdir_names(root):
    names = [os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
             if os.path.isdir(d)]

    def _key(n):
        return (0, int(n)) if n.isdigit() else (1, n)
    names.sort(key=_key)
    return names


def split_names(triplets_root, train_names, val_names, overfit_one,
                val_triplets_root=None):
    """Return (train_names, val_names) lists of folder names.

    If `val_triplets_root` is given, train uses every subdir of
    `triplets_root` and val uses every subdir of `val_triplets_root`
    (separate-directory split — used by the DROID cache, which has its
    own train/ and val/ folders). Otherwise the legacy single-root split
    applies (val carved out of the same root by name).
    """
    all_dirs = _sorted_subdir_names(triplets_root)

    if overfit_one is not None:
        return [overfit_one], []

    if val_triplets_root is not None:
        train = list(train_names) if train_names else all_dirs
        val = _sorted_subdir_names(val_triplets_root)
        return train, val

    val_names = list(val_names) if val_names else []
    if train_names:
        return list(train_names), val_names
    return [n for n in all_dirs if n not in val_names], val_names


def main():
    args = parse_args()
    if not os.path.isdir(args.triplets_root):
        raise FileNotFoundError(f"--triplets_root not found: {args.triplets_root}")
    if args.val_triplets_root is not None and not os.path.isdir(args.val_triplets_root):
        raise FileNotFoundError(
            f"--val_triplets_root not found: {args.val_triplets_root}")
    if not os.path.isfile(args.adaptor_init):
        raise FileNotFoundError(
            f"--adaptor_init not found: {args.adaptor_init}. "
            "Run run_inversion_droid.sh for DROID or pass an existing adaptor_init.pt.")
    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(f"--ckpt_dir not found: {args.ckpt_dir}")
    if args.init_context_path is not None and not os.path.isfile(args.init_context_path):
        raise FileNotFoundError(f"--init_context_path not found: {args.init_context_path}")
    if args.target_context_path is not None and not os.path.isfile(args.target_context_path):
        raise FileNotFoundError(f"--target_context_path not found: {args.target_context_path}")
    if args.sigma_schedule_path is not None and not os.path.isfile(args.sigma_schedule_path):
        raise FileNotFoundError(f"--sigma_schedule_path not found: {args.sigma_schedule_path}")
    if args.loss_type == "context" and args.target_context_path is None:
        raise ValueError("--loss_type context requires --target_context_path")
    if (args.loss_type != "context"
            and args.target_context_path is not None
            and args.context_loss_weight <= 0):
        raise ValueError("--target_context_path requires --context_loss_weight > 0")
    if args.fixed_z_init_path is not None and not os.path.isfile(args.fixed_z_init_path):
        raise FileNotFoundError(f"--fixed_z_init_path not found: {args.fixed_z_init_path}")
    if args.fixed_z_init_path is not None and args.loss_type not in ("rollout", "context"):
        raise ValueError("--fixed_z_init_path is only defined for rollout/context debug runs")
    if args.predict_z_init and args.loss_type != "rollout":
        raise ValueError("--predict_z_init is currently only defined for --loss_type rollout")
    if args.target_z_init_path is not None and not os.path.isfile(args.target_z_init_path):
        raise FileNotFoundError(f"--target_z_init_path not found: {args.target_z_init_path}")
    if args.target_z_init_path is not None and not args.predict_z_init:
        raise ValueError("--target_z_init_path requires --predict_z_init")
    if args.target_z_init_path is not None and args.z_init_loss_weight <= 0:
        raise ValueError("--target_z_init_path requires --z_init_loss_weight > 0")
    if args.predict_z_init and args.fixed_z_init_path is not None:
        raise ValueError("--predict_z_init should learn from seed-noise base; do not also pass --fixed_z_init_path")
    if args.loss_type == "rollout" and args.batch_size != 1:
        raise ValueError("--loss_type rollout currently requires --batch_size 1")
    if (args.loss_type == "rollout"
            and not args.fixed_noise
            and args.fixed_z_init_path is None
            and not args.predict_z_init):
        print("[train] WARNING: rollout loss without --fixed_noise optimizes a "
              "moving initial-noise target; fixed-noise is recommended.")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- split ----
    train_names, val_names = split_names(
        args.triplets_root, args.train_names, args.val_names, args.overfit_one,
        val_triplets_root=args.val_triplets_root)
    val_root = args.val_triplets_root or args.triplets_root
    print(f"[train] train ({len(train_names)}) from {args.triplets_root}: {train_names[:5]}…")
    print(f"[train] val   ({len(val_names)}) from {val_root}: {val_names[:5]}")

    # ---- data ----
    train_ds = TripletLatentDataset(args.triplets_root, train_names)
    val_ds = TripletLatentDataset(val_root, val_names) if val_names else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, collate_fn=collate)
        if val_ds is not None else None
    )

    # ---- Wan pipeline (frozen) ----
    pipe = build_wan_pipeline(args.ckpt_dir)
    sigmas = build_sigma_schedule(
        pipe, args.sampling_steps, args.shift, args.sigma_schedule_path)
    print(f"[train] sigmas: {sigmas[0].item():.4f} → {sigmas[-1].item():.4f} (N+1={len(sigmas)})")

    # Cache T5("") once for CFG dropout, then drop T5 to CPU.
    pipe.text_encoder.model.to(pipe.device)
    with torch.no_grad():
        null_context = pipe.encode_prompt("").detach().float()         # [L_null, 4096]
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    print(f"[train] T5('') shape={tuple(null_context.shape)}")

    # ---- adaptor ----
    adaptor = TrajectoryAdaptor(
        adaptor_init_path=args.adaptor_init,
        hidden=args.hidden,
        action_dim=args.action_dim,
        action_len=args.action_len,
        action_repr=args.action_repr,
        b_head_init_scale=args.b_head_init_scale,
        fusion=args.arch_fusion,
        head=args.arch_head,
        rank_k=args.rank_k,
        n_xattn_layers=args.n_xattn_layers,
        step_emb_dim=args.step_emb_dim,
    ).to(pipe.device)
    if args.init_context_path is not None:
        init_ctx = load_context_tensor(args.init_context_path)
        if tuple(init_ctx.shape) != tuple(adaptor.mu.shape):
            raise ValueError(
                f"--init_context_path shape {tuple(init_ctx.shape)} does not "
                f"match adaptor.mu {tuple(adaptor.mu.shape)}")
        with torch.no_grad():
            adaptor.mu.copy_(init_ctx.to(pipe.device))
            if args.zero_residual_on_init_context:
                zero_adaptor_residual(adaptor)
        print(f"[train] initialized mu from {args.init_context_path}"
              + (" and zeroed residual head"
                 if args.zero_residual_on_init_context else ""))
    print(f"[train] arch: fusion={args.arch_fusion} head={args.arch_head}"
          + (f" (K={args.rank_k})" if args.arch_head == "rankk" else ""))
    print(f"[train] adaptor: N={adaptor.num_steps} L={adaptor.L} D={adaptor.D} "
          f"params={sum(p.numel() for p in adaptor.parameters())/1e6:.2f}M")

    if args.freeze_mu_beta_steps > 0:
        adaptor.freeze_mu_beta()
        print(f"[train] μ, β frozen for the first {args.freeze_mu_beta_steps} steps")
    if args.train_mu_only:
        mu_beta_ids = {id(p) for p in adaptor._mu_beta_params()}
        for p in adaptor.parameters():
            if id(p) not in mu_beta_ids:
                p.requires_grad_(False)
        print("[train] train_mu_only: frozen all non-mu/beta adaptor parameters")

    target_context = None
    if args.target_context_path is not None:
        target_context = load_context_tensor(args.target_context_path).to(pipe.device)
        if tuple(target_context.shape) != tuple(adaptor.mu.shape):
            raise ValueError(
                f"--target_context_path shape {tuple(target_context.shape)} does not "
                f"match adaptor.mu {tuple(adaptor.mu.shape)}")
        print(f"[train] context supervision: target={args.target_context_path} "
              f"weight={args.context_loss_weight:g}")

    # ---- latent geometry (uses first triplet to read shape) ----
    sample = train_ds[0]
    seq_len, mask2 = build_latent_geometry(pipe, sample["z_video"].shape)
    print(f"[train] z_video shape={tuple(sample['z_video'].shape)} seq_len={seq_len}")

    # ---- optional fixed ε₀ / z_init (same initial latent eval uses) ----
    fixed_eps = None
    if args.fixed_z_init_path is not None:
        fixed_eps = load_z_init_tensor(args.fixed_z_init_path).to(pipe.device)
        if tuple(fixed_eps.shape) != tuple(sample["z_video"].shape):
            raise ValueError(
                f"--fixed_z_init_path shape {tuple(fixed_eps.shape)} does not "
                f"match z_video {tuple(sample['z_video'].shape)}")
        torch.save(fixed_eps.detach().cpu(), os.path.join(args.output_dir, "fixed_z_init.pt"))
        print(f"[train] FIXED-Z-INIT mode: loaded exact initial latent from "
              f"{args.fixed_z_init_path} and saved output_dir/fixed_z_init.pt")
    elif args.fixed_noise or args.predict_z_init:
        C_z, F_z, H_z, W_z = sample["z_video"].shape
        g = torch.Generator(device=pipe.device).manual_seed(int(args.seed))
        fixed_eps = torch.randn(
            C_z, F_z, H_z, W_z, generator=g, device=pipe.device,
            dtype=torch.float32)                                   # [C, F_z, H_z, W_z]
        torch.save(fixed_eps.detach().cpu(), os.path.join(args.output_dir, "fixed_z_init.pt"))
        mode = "PRED-Z-INIT base" if args.predict_z_init else "FIXED-NOISE"
        print(f"[train] {mode} mode: one seed latent (seed={args.seed}) reused "
              f"every step; saved output_dir/fixed_z_init.pt")

    z_init_predictor = None
    target_z_init = None
    if args.predict_z_init:
        z_init_predictor = ZInitPredictor(
            latent_shape=sample["z_video"].shape,
            hidden=args.z_init_hidden,
            action_dim=args.action_dim,
            action_len=args.action_len,
            action_repr=args.action_repr,
        ).to(pipe.device)
        z_params_m = sum(p.numel() for p in z_init_predictor.parameters()) / 1e6
        print(f"[train] z_init_predictor: shape={tuple(sample['z_video'].shape)} "
              f"hidden={args.z_init_hidden} params={z_params_m:.2f}M")
        if args.target_z_init_path is not None:
            target_z_init = load_z_init_tensor(args.target_z_init_path).to(pipe.device)
            if tuple(target_z_init.shape) != tuple(sample["z_video"].shape):
                raise ValueError(
                    f"--target_z_init_path shape {tuple(target_z_init.shape)} does not "
                    f"match z_video {tuple(sample['z_video'].shape)}")
            print(f"[train] z_init supervision: target={args.target_z_init_path} "
                  f"weight={args.z_init_loss_weight:g}")

    # ---- optimizer ----
    param_groups = adaptor.parameter_groups(args.lr, args.lr_mu_beta)
    if z_init_predictor is not None:
        param_groups.append({
            "params": list(z_init_predictor.parameters()),
            "lr": args.z_init_lr if args.z_init_lr is not None else args.lr,
            "name": "z_init_predictor",
        })
    optim = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    base_lrs = [g["lr"] for g in optim.param_groups]

    # ---- training loop ----
    log_path = os.path.join(args.output_dir, "train_log.csv")
    with open(log_path, "w") as f:
        if args.loss_type == "rollout":
            f.write("step,loss,latent_mse,ctx_norm,z_pred_std,z_target_std,lr,grad_norm,wall_s,z_init_mse,z_delta_rms\n")
        elif args.loss_type == "context":
            f.write("step,loss,context_mse,ctx_norm,lr,grad_norm,wall_s\n")
        else:
            f.write("step,loss,sigma_mean,t_mean,v_pred_l2,v_target_l2,lr,grad_norm,wall_s\n")
    val_log_path = os.path.join(args.output_dir, "val_log.csv")
    with open(val_log_path, "w") as f:
        f.write("step,val_loss,wall_s\n")

    step = 0
    best_train_loss = float("inf")
    t_start = time.time()
    train_iter = iter(train_loader)
    z_init_frozen_after_pretrain = False
    ckpt_ctx = (
        model_grad_checkpointing(pipe) if args.gradient_checkpointing
        else contextlib.nullcontext()
    )

    with ckpt_ctx:
        while step < args.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Unfreeze μ, β at the configured step.
            if (args.freeze_mu_beta_steps > 0 and step == args.freeze_mu_beta_steps):
                adaptor.unfreeze_mu_beta()
                print(f"[train] step {step}: μ, β unfrozen")
            if (
                args.freeze_z_init_after_pretrain
                and z_init_predictor is not None
                and not z_init_frozen_after_pretrain
                and step >= args.z_init_pretrain_steps
            ):
                for p in z_init_predictor.parameters():
                    p.requires_grad_(False)
                z_init_frozen_after_pretrain = True
                print(f"[train] step {step}: z_init_predictor frozen after pretrain")

            # LR schedule.
            scale = cosine_lr_factor(
                step, args.total_steps, args.warmup_steps, args.lr_min_ratio)
            for g, base in zip(optim.param_groups, base_lrs):
                g["lr"] = base * scale

            adaptor.train()
            if z_init_predictor is not None:
                z_init_predictor.train()
            bsz = batch["z_video"].shape[0]
            eps_ovr = (fixed_eps.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
                       if fixed_eps is not None else None)
            pred_z_init = None
            if z_init_predictor is not None:
                actions_z = batch["actions"].to(pipe.device, non_blocking=True)
                z_I0_z = batch["z_I0"].to(pipe.device, non_blocking=True).float()
                pred_z_init = z_init_predictor(actions_z, z_I0_z, eps_ovr)
                eps_ovr = pred_z_init
            z_pretrain = (
                pred_z_init is not None
                and target_z_init is not None
                and step < args.z_init_pretrain_steps
            )
            if z_pretrain:
                z_delta = pred_z_init - fixed_eps.unsqueeze(0).expand_as(pred_z_init)
                z_loss = masked_latent_mse(pred_z_init, target_z_init, mask2)
                loss = z_loss
                logs = {
                    "loss": float(loss.item()),
                    "latent_mse": float("inf"),
                    "ctx_norm": float(adaptor.mu.detach().norm(dim=-1).mean().item()),
                    "z_pred_std": float(pred_z_init.detach().std().item()),
                    "z_target_std": float(target_z_init.detach().std().item()),
                    "z_init_mse": float(z_loss.item()),
                    "z_delta_rms": float(
                        (z_delta.detach() * mask2.unsqueeze(0)).pow(2).mean().sqrt().item()),
                    "z_pretrain": True,
                }
            elif args.loss_type == "rollout":
                loss, logs = adaptor_rollout_loss(
                    pipe=pipe, adaptor=adaptor, batch=batch, sigmas=sigmas,
                    null_context=null_context, seq_len=seq_len, mask2=mask2,
                    train_guide_scale=args.train_guide_scale,
                    eps_override=eps_ovr,
                    uncond_grad=args.rollout_uncond_grad,
                )
            elif args.loss_type == "context":
                actions_ctx = batch["actions"].to(pipe.device, non_blocking=True)
                z_I0_ctx = batch["z_I0"].to(pipe.device, non_blocking=True).float()
                ctx_pred = adaptor(actions_ctx, z_I0_ctx).float()
                ctx_target = target_context.unsqueeze(0).expand(
                    ctx_pred.shape[0], -1, -1, -1)
                loss = F.mse_loss(ctx_pred, ctx_target)
                logs = {
                    "loss": float(loss.item()),
                    "context_mse": float(loss.item()),
                    "ctx_norm": float(ctx_pred.detach().norm(dim=-1).mean().item()),
                }
            else:
                loss, logs = adaptor_denoising_loss(
                    pipe=pipe, adaptor=adaptor, batch=batch, sigmas=sigmas,
                    null_context=null_context, seq_len=seq_len, mask2=mask2,
                    cfg_dropout=args.cfg_dropout, t_kind=args.t_sampling,
                    ln_mu=args.logit_normal_mu, ln_sigma=args.logit_normal_sigma,
                    train=True, train_guide_scale=args.train_guide_scale,
                    eps_override=eps_ovr,
                )

            if pred_z_init is not None and not z_pretrain:
                z_delta = pred_z_init - fixed_eps.unsqueeze(0).expand_as(pred_z_init)
                logs["z_delta_rms"] = float(
                    (z_delta.detach() * mask2.unsqueeze(0)).pow(2).mean().sqrt().item())
                if target_z_init is not None:
                    z_loss = masked_latent_mse(pred_z_init, target_z_init, mask2)
                    loss = loss + float(args.z_init_loss_weight) * z_loss
                    logs["z_init_mse"] = float(z_loss.item())
                    logs["loss_total"] = float(loss.item())
                    logs["loss"] = float(loss.item())
                else:
                    logs["z_init_mse"] = float("nan")
            elif args.loss_type == "rollout" and not z_pretrain:
                logs["z_delta_rms"] = 0.0
                logs["z_init_mse"] = float("nan")

            if target_context is not None and args.loss_type != "context":
                actions_ctx = batch["actions"].to(pipe.device, non_blocking=True)
                z_I0_ctx = batch["z_I0"].to(pipe.device, non_blocking=True).float()
                ctx_pred = adaptor(actions_ctx, z_I0_ctx).float()
                ctx_target = target_context.unsqueeze(0).expand(
                    ctx_pred.shape[0], -1, -1, -1)
                ctx_loss = F.mse_loss(ctx_pred, ctx_target)
                loss = loss + float(args.context_loss_weight) * ctx_loss
                logs["context_mse"] = float(ctx_loss.item())
                logs["loss_total"] = float(loss.item())
                logs["loss"] = float(loss.item())

            optim.zero_grad(set_to_none=True)
            loss.backward()

            # Grad norm + clip.
            trainable_params = [
                p for group in optim.param_groups for p in group["params"]
                if p.grad is not None
            ]
            gnorm = torch.nn.utils.clip_grad_norm_(
                trainable_params, args.grad_clip,
            )
            if args.loss_type == "rollout":
                train_objective = float(logs["latent_mse"])
            elif args.loss_type == "context":
                train_objective = float(logs["context_mse"])
            else:
                train_objective = float(logs["loss"])
            if train_objective < best_train_loss:
                best_train_loss = train_objective
                best_ckpt = {
                    "step": step,
                    "adaptor": adaptor.state_dict(),
                    "optimizer": optim.state_dict(),
                    "args": vars(args),
                    "best_train_loss": best_train_loss,
                    "pre_optimizer_step": True,
                }
                if z_init_predictor is not None:
                    best_ckpt["z_init_predictor"] = z_init_predictor.state_dict()
                    best_ckpt["z_init_shape"] = tuple(sample["z_video"].shape)
                best_path = os.path.join(args.output_dir, "ckpt_best.pt")
                torch.save(best_ckpt, best_path)
                print(f"[train] saved new best {best_path} loss={best_train_loss:.6f}")
            optim.step()

            elapsed = time.time() - t_start
            if step % args.log_interval == 0 or step == args.total_steps - 1:
                cur_lr = optim.param_groups[-1]["lr"]
                if args.loss_type == "rollout":
                    if logs.get("z_pretrain", False):
                        print(
                            f"[train] step={step:6d} zinit_pretrain_mse="
                            f"{logs['z_init_mse']:.6f} "
                            f"zstd={logs['z_pred_std']:.3f}/{logs['z_target_std']:.3f} "
                            f"|dz|={logs['z_delta_rms']:.3f} "
                            f"|g|={float(gnorm):.2e} lr={cur_lr:.2e} t={elapsed:6.1f}s"
                        )
                    else:
                        print(
                            f"[train] step={step:6d} rollout_mse={logs['latent_mse']:.6f} "
                            f"|C|={logs['ctx_norm']:.3f} "
                            f"zstd={logs['z_pred_std']:.3f}/{logs['z_target_std']:.3f} "
                            + (f" zinit_mse={logs['z_init_mse']:.6f} "
                               f"|dz|={logs['z_delta_rms']:.3f}"
                               if z_init_predictor is not None else "")
                            + (f" ctx_mse={logs['context_mse']:.6f}"
                               if "context_mse" in logs else "")
                            + f" |g|={float(gnorm):.2e} lr={cur_lr:.2e} t={elapsed:6.1f}s"
                        )
                elif args.loss_type == "context":
                    print(
                        f"[train] step={step:6d} context_mse={logs['context_mse']:.6f} "
                        f"|C|={logs['ctx_norm']:.3f} "
                        f"|g|={float(gnorm):.2e} lr={cur_lr:.2e} t={elapsed:6.1f}s"
                    )
                else:
                    print(
                        f"[train] step={step:6d} loss={logs['loss']:.4f} "
                        f"σ̄={logs['sigma_mean']:.3f} t̄={logs['t_mean']:.1f} "
                        f"|v|={logs['v_pred_l2']:.2f}/{logs['v_target_l2']:.2f} "
                        + (f" ctx_mse={logs['context_mse']:.6f}"
                           if "context_mse" in logs else "")
                        + f" |g|={float(gnorm):.2e} lr={cur_lr:.2e} t={elapsed:6.1f}s"
                    )
            with open(log_path, "a") as f:
                if args.loss_type == "rollout":
                    f.write(
                        f"{step},{logs['loss']},{logs['latent_mse']},"
                        f"{logs['ctx_norm']},{logs['z_pred_std']},{logs['z_target_std']},"
                        f"{optim.param_groups[-1]['lr']},{float(gnorm)},{elapsed:.2f},"
                        f"{logs['z_init_mse']},{logs['z_delta_rms']}\n"
                    )
                elif args.loss_type == "context":
                    f.write(
                        f"{step},{logs['loss']},{logs['context_mse']},"
                        f"{logs['ctx_norm']},{optim.param_groups[-1]['lr']},"
                        f"{float(gnorm)},{elapsed:.2f}\n"
                    )
                else:
                    f.write(
                        f"{step},{logs['loss']},{logs['sigma_mean']},"
                        f"{logs['t_mean']},{logs['v_pred_l2']},{logs['v_target_l2']},"
                        f"{optim.param_groups[-1]['lr']},{float(gnorm)},{elapsed:.2f}\n"
                    )

            # Validation.
            if val_loader is not None and (step + 1) % args.val_interval == 0:
                if args.loss_type == "context":
                    print("[train] context loss validation skipped (no per-val context targets)")
                else:
                    adaptor.eval()
                    if args.loss_type == "rollout":
                        val_loss = run_rollout_validation(
                            pipe=pipe, adaptor=adaptor, val_loader=val_loader,
                            sigmas=sigmas, null_context=null_context, seq_len=seq_len,
                            mask2=mask2, train_guide_scale=args.train_guide_scale,
                            fixed_eps=fixed_eps, z_init_predictor=z_init_predictor,
                        )
                    else:
                        val_loss = run_validation(
                            pipe=pipe, adaptor=adaptor, val_loader=val_loader,
                            sigmas=sigmas, null_context=null_context, seq_len=seq_len,
                            mask2=mask2, val_eps_repeats=args.val_eps_repeats,
                            t_kind=args.t_sampling, ln_mu=args.logit_normal_mu,
                            ln_sigma=args.logit_normal_sigma,
                            train_guide_scale=args.train_guide_scale,
                            fixed_eps=fixed_eps,
                        )
                    wall = time.time() - t_start
                    print(f"[train] step={step:6d} VAL loss={val_loss:.4f} (wall {wall:.1f}s)")
                    with open(val_log_path, "a") as f:
                        f.write(f"{step},{val_loss},{wall:.2f}\n")

            # Checkpoint.
            if (step + 1) % args.ckpt_interval == 0 or step == args.total_steps - 1:
                ckpt = {
                    "step": step,
                    "adaptor": adaptor.state_dict(),
                    "optimizer": optim.state_dict(),
                    "args": vars(args),
                }
                if z_init_predictor is not None:
                    ckpt["z_init_predictor"] = z_init_predictor.state_dict()
                    ckpt["z_init_shape"] = tuple(sample["z_video"].shape)
                cpath = os.path.join(args.output_dir, f"ckpt_step{step+1:06d}.pt")
                torch.save(ckpt, cpath)
                latest = os.path.join(args.output_dir, "ckpt_latest.pt")
                torch.save(ckpt, latest)
                print(f"[train] saved {cpath}")

            step += 1

    print(f"[train] done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
