"""
Two-tier image compression for LLM inference and storage.

Compresses images at validation time, returning both:
- inference_image: 1568px max dimension, original format preserved
- storage_image: 512px max dimension, WebP at 75% quality

This enables multi-turn image context while optimizing for both
API token costs (inference tier) and storage efficiency (storage tier).
"""
import base64
import logging
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)

INFERENCE_MAX_DIMENSION = 1568  # API auto-resize ceiling — no point capping below this
STORAGE_MAX_DIMENSION = 512     # Aggressive compression for multi-turn context
STORAGE_WEBP_QUALITY = 75       # WebP quality for storage tier


@dataclass(frozen=True)
class CompressedImage:
    """Result of two-tier image compression."""

    inference_base64: str       # 1568px max (INFERENCE_MAX_DIMENSION), original format
    inference_media_type: str   # Preserves original format
    storage_base64: str         # 512px max, WebP
    storage_media_type: str     # Always "image/webp"
    original_size_bytes: int
    inference_size_bytes: int
    storage_size_bytes: int


def compress_image(image_bytes: bytes, original_media_type: str) -> CompressedImage:
    """
    Compress image to both inference and storage tiers.

    Args:
        image_bytes: Raw image bytes (already decoded from base64)
        original_media_type: Original MIME type (image/jpeg, image/png, etc.)

    Returns:
        CompressedImage with both tier variants

    Raises:
        ValueError: If image cannot be processed (malformed, unsupported format)
    """
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}") from e

    original_size = len(image_bytes)

    # Handle animated GIFs - extract first frame only
    if getattr(img, 'is_animated', False):
        img.seek(0)
        img = img.copy()
        logger.debug("Animated GIF detected - extracted first frame only")

    inference_format = _get_pil_format(original_media_type)

    inference_img = _resize_to_max_dimension(img, INFERENCE_MAX_DIMENSION)
    inference_bytes = _encode_image(inference_img, inference_format, original_media_type)
    inference_base64 = base64.b64encode(inference_bytes).decode('utf-8')

    storage_img = _resize_to_max_dimension(img, STORAGE_MAX_DIMENSION)

    # Convert to RGB for WebP (removes alpha channel if present)
    if storage_img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', storage_img.size, (255, 255, 255))
        if storage_img.mode == 'P':
            storage_img = storage_img.convert('RGBA')
        background.paste(storage_img, mask=storage_img.split()[-1])
        storage_img = background
    elif storage_img.mode != 'RGB':
        storage_img = storage_img.convert('RGB')

    storage_bytes = _encode_image(storage_img, 'WEBP', 'image/webp', quality=STORAGE_WEBP_QUALITY)
    storage_base64 = base64.b64encode(storage_bytes).decode('utf-8')

    result = CompressedImage(
        inference_base64=inference_base64,
        inference_media_type=original_media_type,
        storage_base64=storage_base64,
        storage_media_type="image/webp",
        original_size_bytes=original_size,
        inference_size_bytes=len(inference_bytes),
        storage_size_bytes=len(storage_bytes),
    )

    logger.debug(
        f"Image compression complete: "
        f"original={original_size:,}B -> "
        f"inference={len(inference_bytes):,}B ({inference_img.size[0]}x{inference_img.size[1]}), "
        f"storage={len(storage_bytes):,}B ({storage_img.size[0]}x{storage_img.size[1]})"
    )

    return result


def _get_pil_format(media_type: str) -> str:
    """Convert MIME type to PIL format string."""
    format_map = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }
    return format_map.get(media_type, "JPEG")


def _resize_to_max_dimension(img: Image.Image, max_dim: int) -> Image.Image:
    """Resize image so largest dimension is max_dim, preserving aspect ratio."""
    width, height = img.size

    if width <= max_dim and height <= max_dim:
        return img.copy()

    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _encode_image(
    img: Image.Image,
    fmt: str,
    media_type: str,
    quality: int = 95,
) -> bytes:
    """Encode PIL image to bytes."""
    buffer = BytesIO()
    save_kwargs: dict = {}

    if fmt == "JPEG":
        # JPEG needs RGB mode
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True
    elif fmt == "PNG":
        save_kwargs['optimize'] = True
    elif fmt == "WEBP":
        save_kwargs['quality'] = quality
        save_kwargs['method'] = 4  # Balanced compression speed
    elif fmt == "GIF":
        pass

    img.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()
