"""Directly optimize per-step positive contexts for one DROID clip.

This is a diagnostic for the adaptor failure mode.  `train_adaptor.py` uses a
teacher-forced one-step denoising loss.  Here we optimize the actual replay
objective used by `eval_adaptor.py`: start from eval's seed noise, run the
25-step Euler sampler, and minimize final latent MSE to the cached z_video.

If this succeeds, seed-0 replay is feasible and the adaptor training objective
needs to be changed.  If this fails while oracle replay from saved inversion
z_init succeeds, arbitrary seed replay is out-of-basin for this interface.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

import embedding_search as ES
import embedding_search_losses as L
from models.trajectory_adaptor import TrajectoryAdaptor
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplet_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--adaptor_init", required=True)
    p.add_argument("--adaptor_ckpt", default=None)
    p.add_argument("--oracle_path", default=None)
    p.add_argument(
        "--context_init_path", default=None,
        help="Optional .pt file with positive_embeddings, used to initialize "
        "contexts directly. Overrides --init.",
    )
    p.add_argument("--init", choices=["adaptor", "oracle", "mu"], default="oracle")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--max_area", type=int, default=61440)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--uncond_grad", action="store_true",
                   help="Also backprop through the unconditional DiT branch. "
                   "More exact, much heavier. Default treats uncond velocity "
                   "as a frozen control variate, matching train_adaptor.py.")
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_len", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--arch_fusion", choices=["concat", "cross_attn"], default="cross_attn")
    p.add_argument("--arch_head", choices=["rank1", "rankk", "perstep"], default="perstep")
    p.add_argument("--rank_k", type=int, default=4)
    p.add_argument("--n_xattn_layers", type=int, default=2)
    p.add_argument("--step_emb_dim", type=int, default=128)
    return p.parse_args()


def build_pipe(ckpt_dir: str) -> WanTI2V:
    print(f"[seed-opt] loading Wan TI2V-5B from {ckpt_dir}...")
    pipe = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    pipe.model.eval().requires_grad_(False)
    pipe.vae.model.eval().requires_grad_(False)
    pipe.text_encoder.model.eval().requires_grad_(False)
    pipe.model.to(pipe.device)
    print(f"[seed-opt] loaded. device={pipe.device} dtype={pipe.param_dtype}")
    return pipe


def build_sigmas(pipe, sampling_steps: int, shift: float):
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=pipe.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(sampling_steps, device=pipe.device, shift=shift)
    return scheduler.sigmas.to(pipe.device).float()


def latent_geometry(pipe, z_video):
    c, fz, hz, wz = z_video.shape
    mask2 = torch.ones(c, fz, hz, wz, device=pipe.device, dtype=torch.float32)
    mask2[:, 0] = 0.0
    seq_len = (fz * hz * wz) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int((seq_len + pipe.sp_size - 1) // pipe.sp_size) * pipe.sp_size
    return mask2, seq_len


def load_adaptor_context(args, pipe, actions, z_i0):
    ckpt_args = {}
    if args.adaptor_ckpt:
        ckpt = torch.load(args.adaptor_ckpt, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
    g = lambda k, d: ckpt_args.get(k, d)
    adaptor = TrajectoryAdaptor(
        adaptor_init_path=args.adaptor_init,
        hidden=g("hidden", args.hidden),
        action_dim=g("action_dim", args.action_dim),
        action_len=g("action_len", args.action_len),
        action_repr=g("action_repr", "delta"),
        b_head_init_scale=g("b_head_init_scale", 1e-4),
        fusion=g("arch_fusion", args.arch_fusion),
        head=g("arch_head", args.arch_head),
        rank_k=g("rank_k", args.rank_k),
        n_xattn_layers=g("n_xattn_layers", args.n_xattn_layers),
        step_emb_dim=g("step_emb_dim", args.step_emb_dim),
    ).to(pipe.device).eval()
    if args.adaptor_ckpt:
        missing, unexpected = adaptor.load_state_dict(ckpt["adaptor"], strict=False)
        if missing or unexpected:
            print(f"[seed-opt] adaptor state mismatch: missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        return adaptor(actions, z_i0).squeeze(0).float()


def load_init_context(args, pipe, actions, z_i0):
    if args.context_init_path:
        d = torch.load(args.context_init_path, map_location=pipe.device, weights_only=False)
        if "positive_embeddings" not in d:
            raise KeyError(f"{args.context_init_path} has no positive_embeddings")
        return torch.stack([e.float() for e in d["positive_embeddings"]], dim=0)
    if args.init == "oracle":
        if not args.oracle_path or not os.path.isfile(args.oracle_path):
            raise FileNotFoundError("--init oracle requires --oracle_path")
        d = torch.load(args.oracle_path, map_location=pipe.device, weights_only=False)
        return torch.stack([e.float() for e in d["positive_embeddings"]], dim=0)
    if args.init == "adaptor":
        if not args.adaptor_ckpt:
            raise ValueError("--init adaptor requires --adaptor_ckpt")
        return load_adaptor_context(args, pipe, actions, z_i0)
    init = torch.load(args.adaptor_init, map_location=pipe.device, weights_only=False)
    return init["mu"].float()


def replay_euler_grad(
    pipe,
    contexts,
    null_context,
    z_init,
    z_i0_full,
    mask2_full,
    seq_len,
    sigmas,
    guide_scale,
    uncond_grad=False,
):
    mask2_zero = mask2_full[0]
    z = ES._apply_first_frame_pin(z_init.float(), z_i0_full, mask2_full)
    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i].item()
        sigma_ip1 = sigmas[i + 1].item()
        v_cond = ES._dit_velocity(
            pipe, z, sigma_i, [contexts[i]], seq_len, mask2_zero, pipe.param_dtype)
        if abs(guide_scale - 1.0) > 1e-6:
            if uncond_grad:
                v_uncond = ES._dit_velocity(
                    pipe, z, sigma_i, [null_context], seq_len, mask2_zero, pipe.param_dtype)
            else:
                with torch.no_grad():
                    v_uncond = ES._dit_velocity(
                        pipe, z.detach(), sigma_i, [null_context], seq_len,
                        mask2_zero, pipe.param_dtype)
            v = v_uncond + guide_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = ES._apply_first_frame_pin(
            z + (sigma_ip1 - sigma_i) * v, z_i0_full, mask2_full)
    return z


def per_frame_ssim(video_pred, video_gt):
    return torch.tensor([
        L.ssim(video_pred[:, k], video_gt[:, k]).item()
        for k in range(video_pred.shape[1])
    ])


def save_outputs(pipe, contexts, null_context, z_init, z_i0_full, mask2, seq_len,
                 sigmas, guide_scale, z_video, ref_pixels, out_dir, tag):
    with torch.no_grad():
        z_pred = ES.regenerate_with_positive_embeds(
            pipe=pipe,
            z_init=z_init,
            z_I0_full=z_i0_full,
            mask2_full=mask2,
            seq_len=seq_len,
            sigmas=sigmas,
            null_context_list=[null_context],
            positive_embeds=[contexts[i].detach().float() for i in range(contexts.shape[0])],
            guide_scale=guide_scale,
            param_dtype=pipe.param_dtype,
        )
        video = pipe.vae.decode([z_pred])[0]
    ES.save_video(video, os.path.join(out_dir, f"{tag}.mp4"))
    ssim = per_frame_ssim(video, ref_pixels)
    return {
        f"{tag}_ssim_avg": float(ssim.mean()),
        f"{tag}_ssim_last": float(ssim[-1]),
        f"{tag}_latent_mse": float(F.mse_loss(z_pred, z_video).item()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    pipe = build_pipe(args.ckpt_dir)
    sigmas = build_sigmas(pipe, args.sampling_steps, args.shift)
    print(f"[seed-opt] sigmas {sigmas[0].item():.4f} -> {sigmas[-1].item():.4f}")

    pipe.text_encoder.model.to(pipe.device)
    with torch.no_grad():
        null_context = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()

    actions = torch.from_numpy(
        np.load(os.path.join(args.triplet_dir, "actions.npy"))).float().unsqueeze(0).to(pipe.device)
    z_i0 = torch.load(
        os.path.join(args.triplet_dir, "z_I0.pt"),
        map_location=pipe.device, weights_only=False).float().unsqueeze(0)
    z_video = torch.load(
        os.path.join(args.triplet_dir, "z_video.pt"),
        map_location=pipe.device, weights_only=False).float()
    c, fz, hz, wz = z_video.shape
    z_i0_full = z_i0.squeeze(0).expand(-1, fz, -1, -1).contiguous().float()
    mask2, seq_len = latent_geometry(pipe, z_video)

    seed_g = torch.Generator(device=pipe.device).manual_seed(int(args.seed))
    z_init = torch.randn(
        c, fz, hz, wz, dtype=torch.float32, generator=seed_g, device=pipe.device)
    z_init = ES._apply_first_frame_pin(z_init, z_i0_full, mask2)

    contexts0 = load_init_context(args, pipe, actions, z_i0)
    contexts = torch.nn.Parameter(contexts0.detach().clone().float())
    print(f"[seed-opt] init={args.init} contexts={tuple(contexts.shape)} "
          f"norm={contexts.detach().norm(dim=-1).mean().item():.3f}")

    with torch.no_grad():
        gt_pixels = pipe.vae.decode([z_video])[0]
    ES.save_video(gt_pixels, os.path.join(args.output_dir, "ground_truth.mp4"))

    # Use the VAE roundtrip as the reference for this diagnostic; it avoids
    # source-video availability changing the optimization conclusion.
    ref_pixels = gt_pixels
    metrics = {}
    metrics.update(save_outputs(
        pipe, contexts.detach(), null_context, z_init, z_i0_full, mask2,
        seq_len, sigmas, args.guide_scale, z_video, ref_pixels,
        args.output_dir, "init_seed0"))

    opt = torch.optim.Adam([contexts], lr=args.lr)
    log_path = os.path.join(args.output_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "loss", "latent_mse", "ctx_norm", "grad_norm", "wall_s"])

    t0 = time.time()
    with ES._model_grad_checkpointing(pipe):
        for it in range(args.iters):
            z_pred = replay_euler_grad(
                pipe=pipe,
                contexts=contexts,
                null_context=null_context,
                z_init=z_init,
                z_i0_full=z_i0_full,
                mask2_full=mask2,
                seq_len=seq_len,
                sigmas=sigmas,
                guide_scale=args.guide_scale,
                uncond_grad=args.uncond_grad,
            )
            latent_mse = F.mse_loss(z_pred, z_video)
            loss = latent_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([contexts], args.grad_clip)
            opt.step()

            if it % args.log_interval == 0 or it == args.iters - 1:
                wall = time.time() - t0
                print(f"[seed-opt] iter={it:04d} loss={loss.item():.6f} "
                      f"|C|={contexts.detach().norm(dim=-1).mean().item():.3f} "
                      f"|g|={float(grad_norm):.3e} t={wall:.1f}s")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        it, float(loss.item()), float(latent_mse.item()),
                        float(contexts.detach().norm(dim=-1).mean().item()),
                        float(grad_norm), wall,
                    ])

            if args.save_interval > 0 and (it + 1) % args.save_interval == 0:
                torch.save(
                    {
                        "positive_embeddings": [
                            contexts.detach()[i].cpu() for i in range(contexts.shape[0])
                        ],
                        "z_init": z_init.detach().cpu(),
                        "sigmas": sigmas.detach().cpu(),
                        "guide_scale": args.guide_scale,
                        "iter": it,
                    },
                    os.path.join(args.output_dir, f"contexts_iter{it+1:04d}.pt"),
                )

    torch.save(
        {
            "positive_embeddings": [
                contexts.detach()[i].cpu() for i in range(contexts.shape[0])
            ],
            "z_init": z_init.detach().cpu(),
            "sigmas": sigmas.detach().cpu(),
            "guide_scale": args.guide_scale,
            "iter": args.iters - 1,
        },
        os.path.join(args.output_dir, "positive_embeddings_seed0.pt"),
    )
    metrics.update(save_outputs(
        pipe, contexts.detach(), null_context, z_init, z_i0_full, mask2,
        seq_len, sigmas, args.guide_scale, z_video, ref_pixels,
        args.output_dir, "optimized_seed0"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[seed-opt] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
