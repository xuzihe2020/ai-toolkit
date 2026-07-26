"""Inpaint-mask normalization shared by dataset loaders."""

from __future__ import annotations

from PIL import Image, ImageOps


def normalize_inpaint_mask_image(image: Image.Image) -> Image.Image:
    """Return ai-toolkit's legacy internal RGBA representation.

    Canonical on-disk convention:
      grayscale/RGB white (255) = repaint
      grayscale/RGB black (0) = preserve
      intermediate gray = soft transition

    Backward compatibility:
      an RGBA image with non-constant alpha keeps the legacy convention
      alpha 0 = repaint and alpha 255 = preserve.

    Converting canonical masks to RGBA here lets existing inpaint consumers
    continue reading the alpha channel without changing their internal
    semantics.
    """
    if image.mode == "RGBA":
        alpha_min, alpha_max = image.getchannel("A").getextrema()
        if alpha_min != alpha_max:
            return image

    repaint_mask = image.convert("L")
    preserve_alpha = ImageOps.invert(repaint_mask)
    return Image.merge(
        "RGBA",
        (repaint_mask, repaint_mask, repaint_mask, preserve_alpha),
    )
