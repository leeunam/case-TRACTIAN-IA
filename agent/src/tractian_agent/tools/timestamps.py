"""Validação compartilhada de timestamps ISO 8601 com timezone."""
from __future__ import annotations

from datetime import datetime


def parse_aware_iso_timestamp(value: str) -> datetime:
    """Exige data, hora e offset (ou ``Z``) sem aceitar datetimes ingênuos."""
    if "T" not in value:
        raise ValueError("O timestamp deve conter data, hora e timezone ISO 8601.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("O timestamp deve usar ISO 8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("O timestamp deve informar timezone.")
    return parsed
