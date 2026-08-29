"""Comportamento probabilístico da API.

GETs de consulta podem retornar variações: complete | partial | inconclusive |
conflict | unavailable. O modo é controlado por `seed` (determinístico) ou, sem
seed, amostrado por uma distribuição fixa derivada de data/seed.json.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any


class Mode(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"


DEFAULT_DISTRIBUTION: dict[str, float] = {
    "complete": 0.60,
    "partial": 0.15,
    "inconclusive": 0.10,
    "conflict": 0.08,
    "unavailable": 0.07,
}


def _hash01(*parts: str) -> float:
    """Hash determinístico de partes → float em [0, 1). Não usa Math.random."""
    h = hashlib.sha256(":".join(parts).encode()).hexdigest()
    return int(h[:12], 16) / float(0xFFFFFFFFFFFF)


def resolve_mode(
    *,
    resource: str,
    category: str,
    seed: str | None,
    overrides: dict[str, dict[str, str]] | None,
    distribution: dict[str, float] | None,
) -> Mode:
    """Decide o modo de resposta para um recurso/categoria.

    - Seeds especiais (exceto para recursos com override fixo de cenário):
        `complete`    -> sempre Mode.COMPLETE (útil para explorar no Swagger)
        `degraded`    -> sempre Mode.PARTIAL
    - `overrides[resource][category]` vence para cenários fixos (G501, S420, ...).
    - Com `seed` comum: modo determinístico por hash(seed + resource + category) × distribuição.
    - Sem `seed`: modo determinístico por hash(resource + category) × distribuição
      (estável entre execuções, mas variando por recurso).
    """
    overrides = overrides or {}
    distribution = distribution or DEFAULT_DISTRIBUTION

    ov = overrides.get(resource, {})
    if category in ov:
        return Mode(ov[category])

    if seed == "complete":
        return Mode.COMPLETE
    if seed == "degraded":
        return Mode.PARTIAL
        return Mode(ov[category])

    key = f"{seed or 'noseed'}|{resource}|{category}"
    r = _hash01(key)
    cumulative = 0.0
    for mode_name, weight in distribution.items():
        cumulative += weight
        if r < cumulative:
            return Mode(mode_name)
    return Mode.COMPLETE


def envelope(data: Any, mode: Mode, notes: str | None = None) -> dict[str, Any]:
    """Monta o envelope {mode, notes, data}."""
    body: dict[str, Any] = {"mode": mode.value, "notes": notes, "data": data}
    return body
