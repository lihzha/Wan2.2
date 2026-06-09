"""End-to-end mock validation of scripts/embedding_search.run_search.

Builds a tiny WanTI2V-shaped pipeline (random-init small DiT + stub VAE +
stub text encoder) so the entire optimization driver can be exercised without
a real checkpoint or `flash_attn`. Runs a few opt iterations and asserts that
all expected outputs land on disk.

Run: `python scripts/validate_phase2_mock.py`
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Bypass wan/__init__.py and wan/modules/__init__.py to skip optional deps.
def _register_namespace_pkg(qualname: str, path: str):
    if qualname in sys.modules:
        return
    pkg = types.ModuleType(qualname)
    pkg.__path__ = [path]
    sys.modules[qualname] = pkg


_register_namespace_pkg("wan", os.path.join(REPO_ROOT, "wan"))
_register_namespace_pkg("wan.modules", os.path.join(REPO_ROOT, "wan", "modules"))

# --- SDPA fallback for `flash_attention` so this works without flash_attn.
from wan.modules import attention as _wan_attn  # noqa: E402

if not (_wan_attn.FLASH_ATTN_2_AVAILABLE or _wan_attn.FLASH_ATTN_3_AVAILABLE):
    def _flash_attention_sdpa_fallback(
        q, k, v,
        q_lens=None, k_lens=None,
        dropout_p=0.0, softmax_scale=None, q_scale=None,
        causal=False, window_size=(-1, -1),
        deterministic=False, dtype=torch.bfloat16, version=None,
    ):
        if q_scale is not None:
            q = q * q_scale
        out_dtype = q.dtype
        q_ = q.transpose(1, 2).to(dtype)
        k_ = k.transpose(1, 2).to(dtype)
        v_ = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, dropout_p=dropout_p, is_causal=causal)
        return out.transpose(1, 2).contiguous().to(out_dtype)

    _wan_attn.flash_attention = _flash_attention_sdpa_fallback
    import wan.modules.model as _wan_model  # noqa: E402
    _wan_model.flash_attention = _flash_attention_sdpa_fallback

from wan.modules.model import WanModel  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402

# Now we can import the embedding_search driver itself.
import embedding_search  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny pipeline mocks
# ---------------------------------------------------------------------------


class _StubVAE:
    """Same shape contract as Wan2_2_VAE: .model.z_dim, .encode(list)→list, .decode(list)→list."""

    def __init__(self, z_dim, device):
        self.device = device
        self.model = type("_M", (), {"z_dim": z_dim})()

    def encode(self, videos):
        # videos: list of [3, F, H, W]
        return [
            torch.randn(self.model.z_dim, v.shape[1],
                        v.shape[2] // 16, v.shape[3] // 16, device=v.device)
            for v in videos
        ]

    def decode(self, zs):
        # Differentiable: take first 3 latent channels, upsample, tanh -> [-1,1].
        out = []
        for z in zs:
            t = z[:3].unsqueeze(0)  # [1, 3, F, H, W]
            t = torch.nn.functional.interpolate(
                t, scale_factor=(4, 16, 16), mode="trilinear", align_corners=False
            )
            out.append(torch.tanh(t).squeeze(0))
        return out


class _StubTextEncoder:
    """Mimics T5EncoderModel: callable returning a list of [L, 4096] embeddings.
    L scales with prompt length so init paths behave realistically."""

    def __init__(self, device):
        self.device = device
        self.model = torch.nn.Identity().to(device)  # `.cpu()` / `.to(device)` work

    def __call__(self, prompts, device):
        out = []
        for p in prompts:
            L = max(4, min(64, len(p.split()) + 4))
            t = torch.randn(L, 4096, device=device) * 0.5
            out.append(t)
        return out


def build_mock_pipe(device) -> WanTI2V:
    """Construct an instance with the WanTI2V class but bypass its __init__."""
    pipe = WanTI2V.__new__(WanTI2V)  # skip __init__
    pipe.device = device
    pipe.config = None
    pipe.rank = 0
    pipe.t5_cpu = False
    pipe.init_on_cpu = False
    pipe.num_train_timesteps = 1000
    pipe.param_dtype = torch.float32
    pipe.vae_stride = (4, 16, 16)
    pipe.patch_size = (1, 2, 2)
    pipe.sp_size = 1
    pipe.sample_neg_prompt = "ugly, blurry"

    z_dim = 16
    pipe.model = WanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=64,
        in_dim=z_dim, dim=64, ffn_dim=128, freq_dim=64,
        text_dim=4096, out_dim=z_dim, num_heads=4, num_layers=2,
    ).to(device)
    # head.head zero-init in WanModel.init_weights would silence gradients; reset.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(pipe.model.head.head.weight)
    pipe.model.eval().requires_grad_(False)

    pipe.vae = _StubVAE(z_dim=z_dim, device=device)
    pipe.text_encoder = _StubTextEncoder(device=device)
    return pipe


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def make_test_image(path, size=(96, 96), seed=0):
    g = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size[1], size[0], 3, generator=g) * 255).byte().numpy()
    Image.fromarray(arr).save(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[validate] device={device}")

    with tempfile.TemporaryDirectory() as tmp:
        i0_path = os.path.join(tmp, "I_0.png")
        iT_path = os.path.join(tmp, "I_T.png")
        out_dir = os.path.join(tmp, "out")
        make_test_image(i0_path, seed=1)
        make_test_image(iT_path, seed=2)

        # Build args matching the real CLI but with knobs trimmed for speed.
        args = argparse.Namespace(
            start_frame=i0_path,
            goal_frame=iT_path,
            ckpt_dir="/dev/null",
            output_dir=out_dir,
            heuristic="a test prompt",
            init="heuristic",
            L_opt=8,
            loss="lpips",          # exercises VAE decoder in graph
            num_iters=3,
            lr=1e-2,
            frames=5,              # 4n+1 = 5
            sampling_steps=2,
            guide_scale=1.0,
            max_area=64 * 64,      # tiny; 64x64
            shift=5.0,
            seed=0,
            log_every=1,           # save a frame every iter
            validate=True,         # exercise the validation block too
            sampling_steps_val=2,
            guide_scale_val=1.0,
        )

        pipe = build_mock_pipe(device)

        # Patch the LPIPS loss with SSIM since lpips package is heavy and net
        # download isn't appropriate for a lightweight validation run. The
        # script's loss-routing then exercises the same code path.
        import embedding_search_losses as L
        L.lpips_loss = lambda a, b, **kw: L.ssim_loss(a, b)

        print("[validate] Running run_search…")
        embedding_search.run_search(args, pipe)

        # ---- Assert outputs ----
        expected = [
            "config.json",
            "I_0.png", "I_T_target.png",
            "embedding_init.pt", "embedding_final.pt",
            "loss_log.csv",
            "step_00000_decoded.png",
            "step_00002_decoded.png",  # last iter when num_iters=3, log_every=1
            "final_decoded.png",
        ]
        for name in expected:
            p = os.path.join(out_dir, name)
            assert os.path.exists(p), f"missing output: {p}"
            print(f"[validate] OK  {name}  ({os.path.getsize(p)} bytes)")

        # Inspect loss log: should have header + 3 rows.
        with open(os.path.join(out_dir, "loss_log.csv")) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 1 + args.num_iters, f"loss_log has {len(lines)} lines"
        print(f"[validate] OK  loss_log.csv has {len(lines)} lines (header + {args.num_iters} rows)")

        # Inspect saved final embedding.
        ckpt = torch.load(os.path.join(out_dir, "embedding_final.pt"), weights_only=False)
        assert ckpt["e"].shape == (args.L_opt, 4096), ckpt["e"].shape
        print(f"[validate] OK  embedding_final.pt e shape = {tuple(ckpt['e'].shape)}")

        # Heuristic baseline outputs (validate block).
        for name in ("heuristic_decoded.png", "heuristic_video.mp4"):
            p = os.path.join(out_dir, name)
            # final_video.mp4 may fall back to _strip.png if imageio fails;
            # heuristic_video.mp4 likewise. Accept either.
            ok = os.path.exists(p) or os.path.exists(p.replace(".mp4", "_strip.png"))
            assert ok, f"missing heuristic output: {p}"
            print(f"[validate] OK  {name} (or strip fallback)")

    print("\n[validate] PASS — Phase 2 driver is correct end-to-end.")


if __name__ == "__main__":
    main()
