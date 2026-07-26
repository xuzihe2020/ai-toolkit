import unittest

import torch
from PIL import Image

from toolkit.flux_fill_loss import build_flux_fill_loss_multiplier
from toolkit.inpaint_mask_utils import normalize_inpaint_mask_image


class FluxFillLossTests(unittest.TestCase):
    def test_white_black_mask_normalizes_to_internal_alpha(self):
        mask = Image.new("L", (2, 1), 0)
        mask.putpixel((1, 0), 255)

        normalized = normalize_inpaint_mask_image(mask)

        self.assertEqual("RGBA", normalized.mode)
        self.assertEqual(255, normalized.getchannel("A").getpixel((0, 0)))
        self.assertEqual(0, normalized.getchannel("A").getpixel((1, 0)))

    def test_legacy_variable_alpha_mask_remains_supported(self):
        mask = Image.new("RGBA", (2, 1), (10, 20, 30, 255))
        mask.putpixel((1, 0), (10, 20, 30, 0))

        normalized = normalize_inpaint_mask_image(mask)

        alpha = normalized.getchannel("A")
        self.assertEqual([255, 0], [alpha.getpixel((x, 0)) for x in range(2)])

    def test_opaque_rgba_white_black_mask_uses_rgb_values(self):
        mask = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
        mask.putpixel((1, 0), (255, 255, 255, 255))

        normalized = normalize_inpaint_mask_image(mask)

        alpha = normalized.getchannel("A")
        self.assertEqual([255, 0], [alpha.getpixel((x, 0)) for x in range(2)])

    def test_alpha_drives_normalized_repaint_weighting(self):
        rgba = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
        rgba[:, 3] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        latents = torch.zeros((1, 16, 2, 2), dtype=torch.bfloat16)

        multiplier = build_flux_fill_loss_multiplier(
            rgba, latents, repaint_weight=1.0, preserve_weight=0.1
        )

        self.assertEqual(latents.shape, multiplier.shape)
        self.assertEqual(latents.dtype, multiplier.dtype)
        self.assertAlmostEqual(1.0, multiplier.float().mean().item(), places=2)
        repaint = multiplier[0, 0, 0, 0].float().item()
        preserve = multiplier[0, 0, 0, 1].float().item()
        self.assertAlmostEqual(10.0, repaint / preserve, places=1)

    def test_rejects_non_rgba_and_zero_preserve_weight(self):
        latents = torch.zeros((1, 16, 2, 2))
        with self.assertRaisesRegex(ValueError, "RGBA"):
            build_flux_fill_loss_multiplier(
                torch.zeros((1, 3, 16, 16)), latents
            )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            build_flux_fill_loss_multiplier(
                torch.zeros((1, 4, 16, 16)),
                latents,
                preserve_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
