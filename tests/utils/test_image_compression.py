"""
Tests for image_compression.py two-tier compression.

Focus: Real contract guarantees for image compression utility.
"""
import base64
import pytest
from io import BytesIO

from PIL import Image

from utils.image_compression import (
    compress_image,
    CompressedImage,
    INFERENCE_MAX_DIMENSION,
    STORAGE_MAX_DIMENSION,
)


def create_test_image(width: int, height: int, format: str = "PNG", mode: str = "RGB") -> bytes:
    """Create a test image of specified dimensions and format."""
    img = Image.new(mode, (width, height), color=(255, 0, 0) if mode == "RGB" else (255, 0, 0, 128))
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def decode_and_get_size(base64_data: str) -> tuple[int, int]:
    """Decode base64 image and return (width, height)."""
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(BytesIO(img_bytes))
    return img.size


def decode_and_get_format(base64_data: str) -> str:
    """Decode base64 image and return format."""
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(BytesIO(img_bytes))
    return img.format


class TestInferenceTierDimensions:
    """Tests for inference tier (1200px max) dimension handling."""

    def test_small_image_not_resized(self):
        """CONTRACT: Images smaller than 1200px are not resized for inference."""
        image_bytes = create_test_image(800, 600)

        result = compress_image(image_bytes, "image/png")

        width, height = decode_and_get_size(result.inference_base64)
        assert (width, height) == (800, 600)

    def test_large_landscape_resizes_to_1200_width(self):
        """CONTRACT: Landscape images resize to 1200px width."""
        image_bytes = create_test_image(3000, 2000)

        result = compress_image(image_bytes, "image/jpeg")

        width, height = decode_and_get_size(result.inference_base64)
        assert width == INFERENCE_MAX_DIMENSION
        assert height == 800  # 2000 * (1200/3000) = 800

    def test_large_portrait_resizes_to_1200_height(self):
        """CONTRACT: Portrait images resize to 1200px height."""
        image_bytes = create_test_image(2000, 3000)

        result = compress_image(image_bytes, "image/jpeg")

        width, height = decode_and_get_size(result.inference_base64)
        assert height == INFERENCE_MAX_DIMENSION
        assert width == 800  # 2000 * (1200/3000) = 800


class TestStorageTierDimensions:
    """Tests for storage tier (512px max) dimension handling."""

    def test_storage_tier_always_512_max(self):
        """CONTRACT: Storage tier always has max dimension of 512px."""
        image_bytes = create_test_image(2000, 1500)

        result = compress_image(image_bytes, "image/png")

        width, height = decode_and_get_size(result.storage_base64)
        assert max(width, height) == STORAGE_MAX_DIMENSION

    def test_small_image_still_resized_for_storage(self):
        """CONTRACT: Images larger than 512px are resized for storage tier."""
        image_bytes = create_test_image(800, 600)

        result = compress_image(image_bytes, "image/png")

        width, height = decode_and_get_size(result.storage_base64)
        assert max(width, height) == STORAGE_MAX_DIMENSION


class TestAspectRatioPreservation:
    """Tests for aspect ratio preservation during resizing."""

    def test_preserves_4_to_1_aspect_ratio(self):
        """CONTRACT: 4:1 aspect ratio is preserved in both tiers."""
        image_bytes = create_test_image(4000, 1000)

        result = compress_image(image_bytes, "image/jpeg")

        # Check inference tier
        inf_w, inf_h = decode_and_get_size(result.inference_base64)
        inf_ratio = inf_w / inf_h
        assert abs(inf_ratio - 4.0) < 0.1

        # Check storage tier
        stor_w, stor_h = decode_and_get_size(result.storage_base64)
        stor_ratio = stor_w / stor_h
        assert abs(stor_ratio - 4.0) < 0.1

    def test_preserves_1_to_4_aspect_ratio(self):
        """CONTRACT: 1:4 aspect ratio is preserved in both tiers."""
        image_bytes = create_test_image(1000, 4000)

        result = compress_image(image_bytes, "image/jpeg")

        # Check inference tier
        inf_w, inf_h = decode_and_get_size(result.inference_base64)
        inf_ratio = inf_w / inf_h
        assert abs(inf_ratio - 0.25) < 0.1

        # Check storage tier
        stor_w, stor_h = decode_and_get_size(result.storage_base64)
        stor_ratio = stor_w / stor_h
        assert abs(stor_ratio - 0.25) < 0.1


class TestFormatHandling:
    """Tests for image format handling."""

    def test_inference_preserves_jpeg_format(self):
        """CONTRACT: Inference tier preserves JPEG format."""
        image_bytes = create_test_image(800, 600, format="JPEG")

        result = compress_image(image_bytes, "image/jpeg")

        assert result.inference_media_type == "image/jpeg"
        assert decode_and_get_format(result.inference_base64) == "JPEG"

    def test_inference_preserves_png_format(self):
        """CONTRACT: Inference tier preserves PNG format."""
        image_bytes = create_test_image(800, 600, format="PNG")

        result = compress_image(image_bytes, "image/png")

        assert result.inference_media_type == "image/png"
        assert decode_and_get_format(result.inference_base64) == "PNG"

    def test_storage_always_webp(self):
        """CONTRACT: Storage tier is always WebP regardless of input."""
        # Test with PNG input
        png_bytes = create_test_image(800, 600, format="PNG")
        result = compress_image(png_bytes, "image/png")

        assert result.storage_media_type == "image/webp"
        assert decode_and_get_format(result.storage_base64) == "WEBP"

    def test_storage_converts_jpeg_to_webp(self):
        """CONTRACT: Storage tier converts JPEG to WebP."""
        jpeg_bytes = create_test_image(800, 600, format="JPEG")

        result = compress_image(jpeg_bytes, "image/jpeg")

        assert result.storage_media_type == "image/webp"
        assert decode_and_get_format(result.storage_base64) == "WEBP"


class TestTransparencyHandling:
    """Tests for RGBA to RGB conversion for storage tier."""

    def test_rgba_converts_to_rgb_for_storage(self):
        """CONTRACT: RGBA images convert to RGB for WebP storage."""
        rgba_bytes = create_test_image(800, 600, format="PNG", mode="RGBA")

        result = compress_image(rgba_bytes, "image/png")

        # Storage should be valid WebP (RGB mode)
        img_bytes = base64.b64decode(result.storage_base64)
        img = Image.open(BytesIO(img_bytes))
        assert img.mode == "RGB"


class TestSizeReduction:
    """Tests for compression effectiveness."""

    def test_storage_smaller_than_original(self):
        """CONTRACT: Storage tier is smaller than original."""
        image_bytes = create_test_image(2000, 1500, format="PNG")

        result = compress_image(image_bytes, "image/png")

        assert result.storage_size_bytes < result.original_size_bytes

    def test_storage_smaller_than_inference(self):
        """CONTRACT: Storage tier is smaller than inference tier for large images."""
        image_bytes = create_test_image(2000, 1500, format="PNG")

        result = compress_image(image_bytes, "image/png")

        assert result.storage_size_bytes < result.inference_size_bytes


class TestErrorHandling:
    """Tests for error handling on invalid input."""

    def test_invalid_bytes_raises_valueerror(self):
        """CONTRACT: Invalid image bytes raise ValueError."""
        with pytest.raises(ValueError, match="Failed to open image"):
            compress_image(b"not an image", "image/jpeg")

    def test_empty_bytes_raises_valueerror(self):
        """CONTRACT: Empty bytes raise ValueError."""
        with pytest.raises(ValueError, match="Failed to open image"):
            compress_image(b"", "image/jpeg")


class TestAnimatedGifHandling:
    """Tests for animated GIF frame extraction."""

    def test_animated_gif_uses_first_frame(self):
        """CONTRACT: Animated GIFs extract first frame only."""
        # Create a simple animated GIF with 2 frames
        frames = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='blue'),
        ]
        buffer = BytesIO()
        frames[0].save(
            buffer,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        gif_bytes = buffer.getvalue()

        result = compress_image(gif_bytes, "image/gif")

        # Should produce valid output (first frame)
        assert result.inference_base64 is not None
        assert result.storage_base64 is not None

        # Verify we got an image (not animated)
        inf_bytes = base64.b64decode(result.inference_base64)
        inf_img = Image.open(BytesIO(inf_bytes))
        # GIF output should not be animated
        assert not getattr(inf_img, 'is_animated', False)


class TestCompressedImageDataclass:
    """Tests for CompressedImage dataclass structure."""

    def test_returns_compressed_image_dataclass(self):
        """CONTRACT: compress_image returns CompressedImage dataclass."""
        image_bytes = create_test_image(800, 600)

        result = compress_image(image_bytes, "image/png")

        assert isinstance(result, CompressedImage)

    def test_dataclass_has_all_required_fields(self):
        """CONTRACT: CompressedImage has all expected fields."""
        image_bytes = create_test_image(800, 600)

        result = compress_image(image_bytes, "image/png")

        assert hasattr(result, 'inference_base64')
        assert hasattr(result, 'inference_media_type')
        assert hasattr(result, 'storage_base64')
        assert hasattr(result, 'storage_media_type')
        assert hasattr(result, 'original_size_bytes')
        assert hasattr(result, 'inference_size_bytes')
        assert hasattr(result, 'storage_size_bytes')

    def test_size_fields_are_integers(self):
        """CONTRACT: Size fields are integers representing byte counts."""
        image_bytes = create_test_image(800, 600)

        result = compress_image(image_bytes, "image/png")

        assert isinstance(result.original_size_bytes, int)
        assert isinstance(result.inference_size_bytes, int)
        assert isinstance(result.storage_size_bytes, int)
        assert result.original_size_bytes > 0
        assert result.inference_size_bytes > 0
        assert result.storage_size_bytes > 0
