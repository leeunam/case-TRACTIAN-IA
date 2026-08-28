"""Gera os dados separados de agent-input (visível ao agente) e eval (gabarito).

Princípio: o agente só vê o que um cliente/analista veria ao abrir o chamado — a dúvida textual
e o contexto de acesso. O gabarito (causa-raiz, trajetória esperada, comportamento esperado) é
consumido pelo processo de avaliação, depois da execução, e NUNCA injetado no contexto do agente.

O contrato e os cenários permanecem canônicos em ``docs/`` e não são duplicados.
Roda com `python -m package_material` (a partir de api/) — gera os JSONs derivados.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AGENT_INPUT_DIR = ROOT / "agent-input"
EVAL_DIR = ROOT / "eval"

# Campos do case: o que é INPUT (público ao agente) vs GABARITO (só avaliador)
INPUT_FIELDS = ("id", "ticket_id", "company_id", "user_id", "asset_id", "message")
GABARITO_FIELDS = ("id", "ticket_id", "root_question", "mode", "expected_path")


def _load_cases() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "cases.parquet")


def _to_native(val):
    if hasattr(val, "item"):
        try:
            val = val.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)  # expected_path vem como JSON-string
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def _row(row, fields):
    out = {}
    for f in fields:
        out[f] = _to_native(row[f])
    return out


def build_agent_input() -> None:
    """Pacote visível ao agente: cases.json com só os campos de input + contexto."""
    AGENT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_cases()
    cases = [_row(r, INPUT_FIELDS) for _, r in df.iterrows()]
    (AGENT_INPUT_DIR / "cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False)
    )

    print(f"  agent-input/cases.json  ({len(cases)} casos, só campos de input)")


def build_eval() -> None:
    """Gabarito estruturado: gera apenas ``expected-paths.json``."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_cases()
    expected = [_row(r, GABARITO_FIELDS) for _, r in df.iterrows()]
    (EVAL_DIR / "expected-paths.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False)
    )

    print(f"  eval/expected-paths.json  ({len(expected)} casos, gabarito)")


def main() -> None:
    print(f"Gerando pacotes em {ROOT}/")
    if not (DATA_DIR / "cases.parquet").exists():
        raise SystemExit("Dados não encontrados. Rode `python -m seed_data` primeiro.")
    build_agent_input()
    build_eval()
    print("Concluído.")


if __name__ == "__main__":
    main()
