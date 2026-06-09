"""Phase 1 smoke test for the differentiable TI2V wrapper.

Verifies that `WanTI2V.i2v_diff` allows gradients to flow from a loss on the
final latent back to a learnable text-embedding tensor. Uses a tiny,
randomly-initialized WanModel + a stub VAE so the test needs no checkpoints
and runs on a single GPU (or CPU) in seconds.

Exit criterion: the learnable `context` tensor receives a non-zero gradient
after `loss.backward()`.
"""
import os
import sys
import types

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Bypass `wan/__init__.py` and `wan/modules/__init__.py` so we don't need
# unrelated optional deps (flash_attn, librosa, decord…). We only need the
# two submodules listed below; register parent packages as minimal namespace
# packages pointing at the repo paths.
def _register_namespace_pkg(qualname: str, path: str):
    if qualname in sys.modules:
        return
    pkg = types.ModuleType(qualname)
    pkg.__path__ = [path]
    sys.modules[qualname] = pkg


_register_namespace_pkg("wan", os.path.join(REPO_ROOT, "wan"))
_register_namespace_pkg("wan.modules", os.path.join(REPO_ROOT, "wan", "modules"))

# flash_attn may not be installed on every dev machine. The project's
# attention.py asserts FLASH_ATTN_2_AVAILABLE inside `flash_attention`, so we
# swap in an SDPA-based fallback for the smoke test. Production runs on H200
# should keep flash_attn for speed.
from wan.modules import attention as _wan_attn  # noqa: E402

if not (_wan_attn.FLASH_ATTN_2_AVAILABLE or _wan_attn.FLASH_ATTN_3_AVAILABLE):
    def _flash_attention_sdpa_fallback(
        q, k, v,
        q_lens=None, k_lens=None,
        dropout_p=0.0, softmax_scale=None, q_scale=None,
        causal=False, window_size=(-1, -1),
        deterministic=False, dtype=torch.bfloat16, version=None,
    ):
        # q, k, v: [B, L, H, D]. We ignore q_lens/k_lens (smoke test uses full length).
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
    # The model module imports the symbol via `from .attention import ...`,
    # so patch the binding there too.
    import wan.modules.model as _wan_model  # noqa: E402
    _wan_model.flash_attention = _flash_attention_sdpa_fallback

from wan.modules.model import WanModel  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402


class _StubVAE:
    """Drop-in stand-in for Wan2_2_VAE exposing .model.z_dim, .encode, .decode.
    Returns shape-correct random tensors; used only for smoke testing."""

    def __init__(self, z_dim, device):
        self.device = device
        # mimic the attribute path i2v_diff reads: self.vae.model.z_dim
        self.model = type("_M", (), {"z_dim": z_dim})()

    def encode(self, videos):
        # videos: list of [3, F, H, W]
        return [
            torch.randn(
                self.model.z_dim,
                v.shape[1],  # usually 1 (single start frame)
                v.shape[2] // 16,
                v.shape[3] // 16,
                device=v.device,
            )
            for v in videos
        ]

    def decode(self, zs):
        # Differentiable identity-ish decoder: upsample (z_dim → 3) so gradients
        # flow back through this path. The first 3 latent channels are taken as
        # an RGB proxy.
        out = []
        for z in zs:
            # z: [z_dim, F, H, W] -> [3, F*4, H*16, W*16]
            t = z[:3].unsqueeze(0)  # [1, 3, F, H, W]
            t = torch.nn.functional.interpolate(
                t, scale_factor=(4, 16, 16), mode="trilinear", align_corners=False
            )
            t = torch.tanh(t).squeeze(0)  # [-1, 1] like real VAE output
            out.append(t)
        return out


class _TinyTI2V:
    """Minimal harness exposing the exact attributes `i2v_diff` reads.
    Reuses the real i2v_diff method via attribute binding.
    """

    i2v_diff = WanTI2V.i2v_diff

    def __init__(self, device):
        self.device = device
        self.vae_stride = (4, 16, 16)
        self.patch_size = (1, 2, 2)
        self.sp_size = 1
        self.param_dtype = torch.float32
        self.num_train_timesteps = 1000
        self.rank = 0

        # Tiny DiT with the same architecture as TI2V but far smaller dims.
        z_dim = 16
        self.model = WanModel(
            model_type="ti2v",
            patch_size=(1, 2, 2),
            text_len=32,
            in_dim=z_dim,
            dim=64,
            ffn_dim=128,
            freq_dim=64,
            text_dim=4096,
            out_dim=z_dim,
            num_heads=4,
            num_layers=2,
        ).to(device)
        # WanModel.init_weights() zeros `head.head.weight` (trained checkpoints
        # overwrite this). For an untrained smoke model that leaves outputs at
        # zero and masks all gradient signal, so re-init with Xavier.
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.model.head.head.weight)

        # Match the real pipeline: freeze params (grad still flows through).
        self.model.eval().requires_grad_(False)

        self.vae = _StubVAE(z_dim=z_dim, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    torch.manual_seed(0)
    pipe = _TinyTI2V(device)

    # Fake start frame (PIL Image). i2v_diff crops/resizes via max_area.
    img = Image.fromarray(
        (torch.rand(64, 64, 3) * 255).byte().numpy()
    )

    # Learnable text embedding: [L_opt, 4096]. Start from Gaussian.
    L_opt = 8
    context = torch.randn(L_opt, 4096, device=device, requires_grad=True)
    context_null = torch.zeros(1, 4096, device=device)  # detached

    print(f"[smoke] context: shape={tuple(context.shape)} requires_grad={context.requires_grad}")

    test_configs = [
        ("ckpt=F decode=F", False, False),
        ("ckpt=T decode=F", True, False),
        ("ckpt=T decode=T", True, True),  # exercises VAE decoder in graph
    ]
    for label, use_ckpt, decode in test_configs:
        print(f"\n[smoke] === {label} ===")
        if context.grad is not None:
            context.grad = None

        out = pipe.i2v_diff(
            img=img,
            context=context,
            context_null=context_null,
            max_area=64 * 64,
            frame_num=5,          # must be 4n+1, tiny horizon
            sampling_steps=2,      # minimal steps — we just need graph connectivity
            guide_scale=1.0,       # skip CFG (one DiT forward per step)
            seed=0,
            use_gradient_checkpointing=use_ckpt,
            decode_video=decode,
        )

        latent = out["latent"]
        last = out["latent_last_frame"]
        print(f"[smoke] latent: shape={tuple(latent.shape)} requires_grad={latent.requires_grad}")
        print(f"[smoke] last_frame_latent: shape={tuple(last.shape)}")

        if decode:
            last_frame_pixels = out["last_frame"]
            print(f"[smoke] last_frame_pixels: shape={tuple(last_frame_pixels.shape)} "
                  f"requires_grad={last_frame_pixels.requires_grad}")
            target = torch.zeros_like(last_frame_pixels)
            loss = torch.nn.functional.mse_loss(last_frame_pixels, target)
        else:
            target = torch.zeros_like(last)
            loss = torch.nn.functional.mse_loss(last, target)

        print(f"[smoke] loss={loss.item():.4f}")
        loss.backward()

        assert context.grad is not None, "context.grad is None — graph broke."
        gnorm = context.grad.norm().item()
        gmax = context.grad.abs().max().item()
        print(f"[smoke] context.grad: norm={gnorm:.4e} max|g|={gmax:.4e}")
        assert gnorm > 0, "context.grad is all zeros — no gradient flowed."

    print("\n[smoke] PASS — gradients flow through i2v_diff in all configurations.")


if __name__ == "__main__":
    main()
