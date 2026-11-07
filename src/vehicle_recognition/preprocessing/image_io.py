"""Hardened image loading for untrusted uploads (and dataset images).

Checks, in order: byte size -> file signature (JPEG/PNG magic bytes, not extension) -> header dimensions and
pixel count *before* full decode (decompression-bomb guard) -> structural verify -> full decode -> EXIF
orientation -> RGB. Nothing is written to disk.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class UploadLimits:
    max_bytes: int = _env_int("MAX_UPLOAD_MB", 10) * 1024 * 1024
    max_side: int = 6000
    max_pixels: int = _env_int("MAX_IMAGE_PIXELS", 25_000_000)
    min_side: int = 32


class ImageValidationError(ValueError):
    """Raised for any rejected input. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class LoadedImage:
    rgb: np.ndarray  # HxWx3 uint8, orientation-corrected
    original_size: tuple[int, int]  # (w, h) as stored
    format: str
    exif_orientation: int
    meta: dict = field(default_factory=dict)


def sniff_format(data: bytes) -> str | None:
    if data.startswith(JPEG_MAGIC):
        return "JPEG"
    if data.startswith(PNG_MAGIC):
        return "PNG"
    return None


def load_image_bytes(data: bytes, limits: UploadLimits | None = None) -> LoadedImage:
    limits = limits or UploadLimits()
    if not data:
        raise ImageValidationError("empty", "The uploaded file is empty.")
    if len(data) > limits.max_bytes:
        raise ImageValidationError(
            "too_large", f"File is {len(data) / 1e6:.1f} MB; the limit is {limits.max_bytes / 1024 / 1024:.0f} MB."
        )
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageValidationError("unsupported_format", "Only JPEG and PNG images are accepted (checked by file content).")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        old_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = limits.max_pixels
        try:
            try:
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != fmt:
                        raise ImageValidationError("unsupported_format", "File content does not match a JPEG/PNG image.")
                    w, h = im.size
                    if max(w, h) > limits.max_side or w * h > limits.max_pixels:
                        raise ImageValidationError(
                            "too_many_pixels",
                            f"Image is {w}x{h}; limits are {limits.max_side}px per side and {limits.max_pixels / 1e6:.0f} MP.",
                        )
                    if min(w, h) < limits.min_side:
                        raise ImageValidationError("too_small", f"Image is {w}x{h}; minimum side is {limits.min_side}px.")
                    im.verify()
                with Image.open(io.BytesIO(data)) as im:
                    try:
                        orient = int(im.getexif().get(0x0112, 1))
                    except Exception:  # noqa: BLE001
                        orient = 1
                    im.load()
                    im = ImageOps.exif_transpose(im)
                    if im.mode not in ("RGB",):
                        if im.mode in ("RGBA", "LA", "P") and ("transparency" in im.info or im.mode in ("RGBA", "LA")):
                            bg = Image.new("RGB", im.size, (255, 255, 255))
                            rgba = im.convert("RGBA")
                            bg.paste(rgba, mask=rgba.split()[-1])
                            im = bg
                        else:
                            im = im.convert("RGB")
                    arr = np.asarray(im, dtype=np.uint8).copy()
            except ImageValidationError:
                raise
            except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
                raise ImageValidationError("too_many_pixels", "Image exceeds the decoded pixel limit.") from e
            except Exception as e:  # noqa: BLE001
                raise ImageValidationError("corrupt", "The image could not be decoded (corrupt or truncated file).") from e
        finally:
            Image.MAX_IMAGE_PIXELS = old_limit
    return LoadedImage(rgb=arr, original_size=(w, h), format=fmt, exif_orientation=orient)


def load_dataset_image(path: str | os.PathLike) -> np.ndarray:
    """Trusted dataset image in *stored* orientation (annotations refer to stored pixels) as RGB uint8."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
