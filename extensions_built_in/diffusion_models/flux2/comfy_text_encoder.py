"""Local ComfyUI text-encoder adapter for FLUX.2 Klein.

ComfyUI distributes Qwen3 as a single mixed-FP8 safetensors checkpoint.  That
format is intentionally loaded by ComfyUI's quantization-aware loader rather
than ``transformers.from_pretrained``.  Keeping this adapter in ai-toolkit
lets training consume the exact same durable model file as ComfyUI without a
second Hugging Face-format copy or any hub/cache fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch


class LocalComfyQwen3Encoder(torch.nn.Module):
    """Expose a ComfyUI Qwen checkpoint to the ai-toolkit Flux2 pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        comfyui_path: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path).resolve()
        comfy_root = Path(comfyui_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Local FLUX.2 text encoder does not exist: {checkpoint}"
            )
        if not (comfy_root / "comfy" / "sd.py").is_file():
            raise FileNotFoundError(
                f"ComfyUI loader is not available under: {comfy_root}"
            )

        comfy_root_str = str(comfy_root)
        if comfy_root_str not in sys.path:
            sys.path.insert(0, comfy_root_str)

        import comfy.sd  # imported lazily; ai-toolkit remains standalone

        self._dtype = dtype
        self._checkpoint_path = str(checkpoint)
        self._comfy_sd = comfy.sd
        self._clip = comfy.sd.load_clip(
            [self._checkpoint_path], clip_type=comfy.sd.CLIPType.FLUX2
        )
        # A zero-byte anchor supplies the standard Module device/dtype surface
        # without registering ComfyUI's independently managed model graph.
        self.register_buffer(
            "_device_anchor", torch.empty(0, dtype=dtype), persistent=False
        )

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    @torch.no_grad()
    def encode_prompts(
        self, prompts: Iterable[str], device: torch.device | str | None = None
    ) -> torch.Tensor:
        encoded = []
        for prompt in prompts:
            tokens = self._clip.tokenize(prompt)
            cond = self._clip.encode_from_tokens(tokens)
            encoded.append(cond.to(dtype=self._dtype))
        result = torch.cat(encoded, dim=0)
        return result.to(device=device) if device is not None else result

    def unload(self) -> None:
        """Release ComfyUI-managed VRAM after text embeddings are cached."""
        import comfy.model_management

        comfy.model_management.unload_model_and_clones(
            self._clip.patcher, all_devices=True
        )
        comfy.model_management.soft_empty_cache()

