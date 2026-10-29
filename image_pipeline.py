"""
Robust Image Validation and Format Detection Module for Multimodal RAG.

Addresses:
- P1: Detection of actual MIME types via magic bytes (avoids hardcoded image/jpeg).
- Byte validation: Verifies integrity via Pillow and catches corrupted base64 payloads.
- Explicit handling of malformed or unsupported image formats.
"""

import base64
import io
import re
from dataclasses import dataclass
from typing import Optional, Union
from PIL import Image


@dataclass
class ValidatedImage:
    """Structured representation of a verified image."""
    raw_bytes: bytes
    mime_type: str        # e.g., 'image/png', 'image/jpeg', 'image/webp'
    format: str           # e.g., 'PNG', 'JPEG', 'WEBP'
    width: int
    height: int
    base64_data: str      # Clean base64 payload without data URI prefix
    is_valid: bool = True
    error_message: Optional[str] = None

    @property
    def data_uri(self) -> str:
        """Returns browser-compatible data URI for HTML rendering."""
        return f"data:{self.mime_type};base64,{self.base64_data}"


def detect_mime_from_magic_bytes(header: bytes) -> Optional[str]:
    """Detects image MIME type by inspecting the first 16 bytes."""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    elif header.startswith(b"RIFF") and b"WEBP" in header[:12]:
        return "image/webp"
    elif header.startswith(b"BM"):
        return "image/bmp"
    elif header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "image/tiff"
    return None


def sanitize_base64_payload(b64_string: str) -> str:
    """Strips data URI prefixes (e.g., 'data:image/png;base64,') and whitespace."""
    clean = b64_string.strip()
    if clean.startswith("data:"):
        match = re.search(r"base64,(.*)", clean)
        if match:
            clean = match.group(1)
    return clean


def validate_image_payload(image_source: Union[str, bytes]) -> ValidatedImage:
    """
    Validates, repairs, and detects format for an image payload.
    
    Accepts:
    - Base64 encoded string (with or without data:image prefix)
    - Raw image bytes
    
    Returns:
    - ValidatedImage object with true MIME type and dimensions.
    """
    raw_bytes: bytes = b""
    clean_b64: str = ""

    # 1. Decode to bytes
    try:
        if isinstance(image_source, str):
            clean_b64 = sanitize_base64_payload(image_source)
            # Add padding if stripped
            missing_padding = len(clean_b64) % 4
            if missing_padding:
                clean_b64 += "=" * (4 - missing_padding)
            raw_bytes = base64.b64decode(clean_b64)
        elif isinstance(image_source, bytes):
            raw_bytes = image_source
            clean_b64 = base64.b64encode(raw_bytes).decode("ascii")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_source)}")
    except Exception as e:
        return ValidatedImage(
            raw_bytes=b"",
            mime_type="application/octet-stream",
            format="UNKNOWN",
            width=0,
            height=0,
            base64_data="",
            is_valid=False,
            error_message=f"Failed to decode base64: {e}",
        )

    if not raw_bytes:
        return ValidatedImage(
            raw_bytes=b"",
            mime_type="application/octet-stream",
            format="UNKNOWN",
            width=0,
            height=0,
            base64_data="",
            is_valid=False,
            error_message="Image payload is empty (0 bytes)",
        )

    # 2. Detect MIME type via magic bytes
    mime_type = detect_mime_from_magic_bytes(raw_bytes[:16])

    # 3. Verify image integrity and dimensions using Pillow
    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            # verify() checks stream integrity
            pil_img.verify()
            
        # Re-open after verify() to read dimensions safely
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            width, height = pil_img.size
            img_format = pil_img.format or "UNKNOWN"
            
            # If magic bytes didn't catch it, fallback to Pillow format
            if not mime_type:
                fmt_lower = img_format.lower()
                mime_type = f"image/{fmt_lower}" if fmt_lower != "jpg" else "image/jpeg"

    except Exception as e:
        return ValidatedImage(
            raw_bytes=raw_bytes,
            mime_type=mime_type or "application/octet-stream",
            format="CORRUPT",
            width=0,
            height=0,
            base64_data=clean_b64,
            is_valid=False,
            error_message=f"Corrupt or malformed image: {e}",
        )

    return ValidatedImage(
        raw_bytes=raw_bytes,
        mime_type=mime_type,
        format=img_format,
        width=width,
        height=height,
        base64_data=clean_b64,
        is_valid=True,
    )

