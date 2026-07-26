"""Loss weighting helpers for FLUX.1 Fill training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_flux_fill_loss_multiplier(
        inpaint_tensor: torch.Tensor,
        reference_latents: torch.Tensor,
        repaint_weight: float = 1.0,
        preserve_weight: float = 0.1,
) -> torch.Tensor:
    """Build a normalized latent-space loss multiplier from RGBA alpha.

    Dataset convention:
      alpha 0 = repaint
      alpha 1 = preserve

    Repaint pixels receive ``repaint_weight`` and preserved pixels receive
    ``preserve_weight``. Each sample is normalized to mean 1 so mask coverage
    does not silently change the effective learning rate.
    """
    if inpaint_tensor.ndim != 4 or inpaint_tensor.shape[1] != 4:
        raise ValueError(
            "FLUX.1 Fill loss weighting requires RGBA inpaint tensors with "
            "shape [batch, 4, height, width]."
        )
    if reference_latents.ndim != 4:
        raise ValueError(
            "FLUX.1 Fill loss weighting requires image latents with shape "
            "[batch, channels, height, width]."
        )
    if repaint_weight <= 0 or preserve_weight <= 0:
        raise ValueError("FLUX.1 Fill loss weights must both be greater than zero.")
    if repaint_weight <= preserve_weight:
        raise ValueError(
            "FLUX.1 Fill repaint loss weight must be greater than the "
            "preserve loss weight."
        )

    batch_size = reference_latents.shape[0]
    if inpaint_tensor.shape[0] != batch_size:
        if batch_size % inpaint_tensor.shape[0] != 0:
            raise ValueError(
                "FLUX.1 Fill mask batch does not match the latent batch."
            )
        inpaint_tensor = inpaint_tensor.repeat(
            batch_size // inpaint_tensor.shape[0], 1, 1, 1
        )

    repaint_mask = 1.0 - inpaint_tensor[:, 3:4].float().clamp(0.0, 1.0)
    repaint_mask = F.interpolate(
        repaint_mask,
        size=reference_latents.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    multiplier = preserve_weight + repaint_mask * (
        repaint_weight - preserve_weight
    )
    multiplier = multiplier.expand(
        -1, reference_latents.shape[1], -1, -1
    )

    # Normalize each sample independently. This retains the inside/outside
    # ratio while keeping the average gradient scale stable across mask sizes.
    sample_mean = multiplier.mean(dim=(1, 2, 3), keepdim=True)
    if not torch.isfinite(sample_mean).all() or (sample_mean <= 0).any():
        raise ValueError("FLUX.1 Fill produced an invalid loss multiplier.")
    multiplier = multiplier / sample_mean

    return multiplier.to(
        device=reference_latents.device,
        dtype=reference_latents.dtype,
    ).detach()
