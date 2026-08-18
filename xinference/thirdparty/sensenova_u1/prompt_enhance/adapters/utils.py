"""Image encoding / decoding utilities for VLM."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def read_image_bytes(image: str | bytes) -> bytes:
    """Read raw image bytes from a path or return bytes unchanged.

    Args:
        image: File path to an image, or raw image bytes.

    Returns:
        bytes: Raw image bytes.

    Raises:
        FileNotFoundError: If image is a path and the file does not exist.
    """
    if isinstance(image, bytes):
        return image
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {image}")
    return path.read_bytes()


def detect_mime(data: bytes) -> str:
    """Infer MIME type from image magic bytes.

    Args:
        data: Raw image bytes (at least 8 bytes for PNG check).

    Returns:
        str: 'image/png', 'image/jpeg', or 'image/png' as fallback.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def detect_suffix(data: bytes) -> str:
    """Infer file suffix from image magic bytes.

    Args:
        data: Raw image bytes.

    Returns:
        str: '.png', '.jpg', or '.bin' as fallback.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    return ".bin"


def image_to_mime_and_bytes(image: str | bytes) -> tuple[str, bytes]:
    """Get MIME type and raw bytes; convert to PNG if format is not PNG/JPEG.

    Args:
        image: File path or raw image bytes.

    Returns:
        tuple[str, bytes]: (mime_type, raw_bytes). Unknown formats become PNG.
    """
    raw = read_image_bytes(image)
    mime = detect_mime(raw)
    if mime in ("image/png", "image/jpeg"):
        return mime, raw
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "image/png", buf.getvalue()


def image_to_base64(image: str | bytes) -> tuple[str, str]:
    """Encode image to MIME type and base64 string.

    Args:
        image: File path or raw image bytes.

    Returns:
        tuple[str, str]: (mime_type, base64_encoded_string).
    """
    mime, raw = image_to_mime_and_bytes(image)
    return mime, base64.b64encode(raw).decode("utf-8")


def image_to_data_url(image: str | bytes) -> str:
    """Build a data URL (data:mime;base64,...) for the image.

    Args:
        image: File path or raw image bytes.

    Returns:
        str: Data URL string.
    """
    mime, b64 = image_to_base64(image)
    return f"data:{mime};base64,{b64}"


def mask_secret(secret: str) -> str:
    """Mask a secret for logging (e.g. show first 6 and last 4 chars).

    Args:
        secret: Raw secret string.

    Returns:
        str: Masked string (e.g. 'abcdef...ghij' or all '*' if length <= 8).
    """
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:6]}...{secret[-4:]}"
