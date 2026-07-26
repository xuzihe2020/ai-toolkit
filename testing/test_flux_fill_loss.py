import unittest

import torch

from toolkit.flux_fill_loss import build_flux_fill_loss_multiplier


class FluxFillLossTests(unittest.TestCase):
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
