"""Loss helpers for the embedding search study.

All losses operate on tensors in [-1, 1] (matching VAE decode output).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_mse(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    """MSE in VAE latent space. pred/target shape: [C, H, W] or [C, F, H, W]."""
    return F.mse_loss(pred_latent, target_latent)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Differentiable mean SSIM. img1/img2 in [-1, 1] with shape [3, H, W] or [B, 3, H, W].
    Returns a scalar in [-1, 1] (typically [0, 1] for natural images)."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    device, dtype = img1.device, img1.dtype

    g = _gaussian_window(window_size, sigma, device, dtype)
    window_2d = g[:, None] * g[None, :]
    C = img1.shape[1]
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, **kwargs) -> torch.Tensor:
    return 1.0 - ssim(img1, img2, **kwargs)


_LPIPS_MODEL = None


def lpips_loss(img1: torch.Tensor, img2: torch.Tensor, net: str = "vgg") -> torch.Tensor:
    """LPIPS perceptual loss. Imports lpips lazily; raises a clean error if missing.
    img1/img2 in [-1, 1] with shape [3, H, W] or [B, 3, H, W]."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips  # type: ignore
        except ImportError as e:
            raise ImportError(
                "lpips loss requested but `lpips` package is not installed. "
                "Run `pip install lpips`."
            ) from e
        _LPIPS_MODEL = lpips.LPIPS(net=net).to(img1.device).eval()
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad_(False)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    return _LPIPS_MODEL(img1, img2).mean()
