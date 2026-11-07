import io

import numpy as np
import pytest
from PIL import Image

from vehicle_recognition.preprocessing.image_io import ImageValidationError, UploadLimits, load_image_bytes


def _encode(img: Image.Image, fmt: str, **kw) -> bytes:
    b = io.BytesIO()
    img.save(b, format=fmt, **kw)
    return b.getvalue()


def test_valid_jpeg_and_png():
    im = Image.new("RGB", (120, 80), (10, 200, 30))
    for fmt in ("JPEG", "PNG"):
        out = load_image_bytes(_encode(im, fmt))
        assert out.rgb.shape == (80, 120, 3) and out.rgb.dtype == np.uint8 and out.format == fmt


def test_rgba_png_flattened_to_rgb():
    im = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    out = load_image_bytes(_encode(im, "PNG"))
    assert out.rgb.shape == (64, 64, 3)
    assert tuple(out.rgb[0, 0]) == (255, 255, 255)  # transparent -> white background


def test_grayscale_converted():
    out = load_image_bytes(_encode(Image.new("L", (64, 64), 128), "PNG"))
    assert out.rgb.shape == (64, 64, 3)


def test_exif_orientation_applied():
    im = Image.new("RGB", (100, 50), (0, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    out = load_image_bytes(_encode(im, "JPEG", exif=exif))
    assert out.exif_orientation == 6
    assert out.rgb.shape[:2] == (100, 50)  # height/width swapped after transpose


@pytest.mark.parametrize("data,code", [
    (b"", "empty"),
    (b"GIF89a" + b"\x00" * 100, "unsupported_format"),
    (b"<?php echo 1; ?>", "unsupported_format"),
])
def test_rejects_bad_signatures(data, code):
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(data)
    assert e.value.code == code


def test_extension_spoof_gif_content_rejected():
    data = _encode(Image.new("RGB", (64, 64)), "GIF")
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(data)
    assert e.value.code == "unsupported_format"


def test_truncated_jpeg_is_corrupt():
    data = _encode(Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)), "JPEG")
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(data[: len(data) // 3])
    assert e.value.code == "corrupt"


def test_too_large_bytes():
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(b"\xff\xd8\xff" + b"0" * 2048, UploadLimits(max_bytes=1024))
    assert e.value.code == "too_large"


def test_pixel_limit_checked_before_decode():
    # 7000x100 PNG is tiny on disk but exceeds the per-side limit
    data = _encode(Image.new("L", (7000, 100)), "PNG")
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(data)
    assert e.value.code == "too_many_pixels"


def test_decompression_bomb_pixel_count():
    data = _encode(Image.new("1", (5000, 5000)), "PNG")  # 25 MP, a few KB
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(data, UploadLimits(max_pixels=10_000_000))
    assert e.value.code == "too_many_pixels"


def test_too_small():
    with pytest.raises(ImageValidationError) as e:
        load_image_bytes(_encode(Image.new("RGB", (10, 10)), "PNG"))
    assert e.value.code == "too_small"
