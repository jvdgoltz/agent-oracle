"""Validate and translate payloads for embedded Codex sessions."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

_IMAGE_DATA_URL_RE = re.compile(
    r"^data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/]*={0,2})$"
)
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_IMAGE_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


def validated_image_data_url(body: dict[str, Any]) -> str | None:
    """Validate an optional browser image data URL before passing it to Codex."""
    value = body.get("image_data_url")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("image_data_url must be a base64 PNG, JPEG, WebP, or GIF under 10 MB")
    match = _IMAGE_DATA_URL_RE.fullmatch(value)
    if match is None:
        raise ValueError("image_data_url must be a base64 PNG, JPEG, WebP, or GIF under 10 MB")
    mime_type, encoded = match.groups()
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_data_url contains invalid Base64 data") from exc
    if not image_bytes or len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError("image_data_url must contain an image no larger than 10 MB")
    if mime_type == "image/webp":
        valid_type = image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
    else:
        valid_type = any(
            image_bytes.startswith(signature) for signature in _IMAGE_SIGNATURES[mime_type]
        )
    if not valid_type:
        raise ValueError("image_data_url content does not match its image type")
    return value


def source_messages(messages: list[Any], session_id: str) -> list[dict[str, Any]]:
    """Convert locally parsed messages into the session-detail API shape."""
    return [
        {
            "id": sequence,
            "session_id": session_id,
            "role": str(message.role),
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "seq": sequence,
            "is_thinking": int(message.is_thinking),
            "model": message.model,
            "is_system_instruction": int(message.is_system_instruction),
            "is_injected": int(message.is_injected),
        }
        for sequence, message in enumerate(messages)
    ]
