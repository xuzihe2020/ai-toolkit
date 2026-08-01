import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from extensions_built_in.diffusion_models.flux2.flux2_klein_model import (
    Flux2KleinModel,
)


class Flux2LocalTextEncoderTests(unittest.TestCase):
    def test_strict_local_mode_never_calls_transformers_from_pretrained(self):
        fake = SimpleNamespace(
            model_config=SimpleNamespace(
                te_name_or_path="/workspace/comfyui_models/text_encoders/flux2/"
                "qwen_3_8b_fp8mixed.safetensors",
                model_kwargs={
                    "strict_local_models": True,
                    "comfyui_path": "/workspace/ComfyUI",
                },
            ),
            flux2_klein_te_path="Qwen/Qwen3-8B",
            torch_dtype=torch.bfloat16,
            print_and_status_update=Mock(),
        )
        local_encoder = Mock()
        with (
            patch(
                "extensions_built_in.diffusion_models.flux2."
                "flux2_klein_model.LocalComfyQwen3Encoder",
                return_value=local_encoder,
            ) as local_loader,
            patch(
                "extensions_built_in.diffusion_models.flux2."
                "flux2_klein_model.Qwen3ForCausalLM.from_pretrained"
            ) as hub_loader,
            patch(
                "extensions_built_in.diffusion_models.flux2."
                "flux2_klein_model.Qwen2Tokenizer.from_pretrained"
            ) as hub_tokenizer,
        ):
            encoder, tokenizer = Flux2KleinModel.load_te(fake)

        self.assertIs(local_encoder, encoder)
        self.assertIsNone(tokenizer)
        local_loader.assert_called_once_with(
            fake.model_config.te_name_or_path,
            "/workspace/ComfyUI",
            dtype=torch.bfloat16,
        )
        hub_loader.assert_not_called()
        hub_tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
