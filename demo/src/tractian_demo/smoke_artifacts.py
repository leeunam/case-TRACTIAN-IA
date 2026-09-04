from __future__ import annotations

import json
from pathlib import Path


def write_smoke_artifact(output_path: Path, value: dict[str, object]) -> None:
    """Grava evidência sanitizada de forma atômica fora do Git."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
