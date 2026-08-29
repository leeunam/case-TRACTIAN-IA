"""Carrega os parquet em memória e fornece lookups. Leve: dados são didáticos e pequenos."""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _normalize(s: str) -> str:
    """Lowercase + remove acentos para busca tolerante."""
    nf = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nf if not unicodedata.combining(c)).lower()


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame]:
    if not (DATA_DIR / "companies.parquet").exists():
        raise RuntimeError(
            "Dados não encontrados em data/. Rode `python -m seed_data` (em api/) primeiro."
        )
    tables: dict[str, pd.DataFrame] = {}
    for p in DATA_DIR.glob("*.parquet"):
        tables[p.stem] = pd.read_parquet(p)
    return tables


def _json_col(row: Any, col: str) -> Any:
    """Lê coluna armazenada como string JSON."""
    val = row[col]
    if pd.isna(val) or val in ("", None):
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val  # já é lista/dict (pyarrow pode inferir)


def _to_native(val: Any) -> Any:
    """Converte tipos numpy/pandas para nativos Python (serializáveis em JSON)."""
    if val is None:
        return None
    # numpy escalares
    if hasattr(val, "item"):
        try:
            val = val.item()
        except (ValueError, AttributeError):
            pass
    # pandas NaN (float)
    if isinstance(val, float) and pd.isna(val):
        return None
    return val


def _row_to_dict(row: pd.Series, json_cols: tuple[str, ...] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col, val in row.items():
        if col in json_cols:
            out[col] = _json_col(row, col)
        else:
            out[col] = _to_native(val)
    return out


def _find(table: str, col: str, value: str) -> dict[str, Any] | None:
    df = _tables()[table]
    hits = df[df[col] == value]
    if hits.empty:
        return None
    return _row_to_dict(hits.iloc[0], _JSON_COLS.get(table, ()))


def _filter(table: str, col: str, value: str) -> list[dict[str, Any]]:
    df = _tables()[table]
    hits = df[df[col] == value]
    return [_row_to_dict(r, _JSON_COLS.get(table, ())) for _, r in hits.iterrows()]


# Colunas armazenadas como JSON-string no parquet
_JSON_COLS: dict[str, tuple[str, ...]] = {
    "users": ("permissions",),
    "assets": (),  # sem colunas JSON (bearing specs já são colunas escalares)
    "baselines": ("features",),
    "analyses": ("evidence", "limitations"),
    "models": ("coverage",),
    "knowledge": ("tags",),
    "spectra": ("peaks", "bands_missing"),
    "cases": ("expected_path",),
}


# ---- Contexto ----
def get_company(company_id: str) -> dict[str, Any] | None:
    return _find("companies", "id", company_id)


def list_assets_by_company(company_id: str) -> list[dict[str, Any]]:
    return _filter("assets", "company_id", company_id)


def get_user(user_id: str) -> dict[str, Any] | None:
    return _find("users", "id", user_id)


# ---- Ativos ----
def get_asset(asset_id: str) -> dict[str, Any] | None:
    return _find("assets", "id", asset_id)


def list_points(asset_id: str) -> list[dict[str, Any]]:
    return _filter("points", "asset_id", asset_id)


# ---- Análises ----
def list_analyses(asset_id: str, status: str | None = None) -> list[dict[str, Any]]:
    rows = _filter("analyses", "asset_id", asset_id)
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return rows


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    return _find("analyses", "id", analysis_id)


# ---- Baseline ----
def get_baseline(asset_id: str, point_id: str | None = None) -> dict[str, Any] | None:
    rows = _filter("baselines", "asset_id", asset_id)
    if point_id:
        rows = [r for r in rows if r.get("point_id") == point_id]
    return rows[0] if rows else None


# ---- Dados técnicos ----
def get_rms(asset_id: str, point_id: str | None = None) -> list[dict[str, Any]]:
    rows = _filter("rms", "asset_id", asset_id)
    if point_id:
        rows = [r for r in rows if r.get("point_id") == point_id]
    return rows


def get_spectrum(asset_id: str, point_id: str | None = None) -> dict[str, Any] | None:
    rows = _filter("spectra", "asset_id", asset_id)
    if point_id:
        rows = [r for r in rows if r.get("point_id") == point_id]
    return rows[0] if rows else None


def get_data_quality(asset_id: str, point_id: str | None = None) -> dict[str, Any] | None:
    rows = _filter("data_quality", "asset_id", asset_id)
    if point_id:
        rows = [r for r in rows if r.get("point_id") == point_id]
    return rows[0] if rows else None


# ---- Modelos ----
def get_model(model_id: str) -> dict[str, Any] | None:
    return _find("models", "id", model_id)


def list_models() -> list[dict[str, Any]]:
    return [_row_to_dict(r, _JSON_COLS["models"]) for _, r in _tables()["models"].iterrows()]


# ---- Conhecimento ----
def search_knowledge(q: str, type_: str | None = None) -> list[dict[str, Any]]:
    df = _tables()["knowledge"]
    ql = _normalize(q)
    mask = df["title"].map(_normalize).str.contains(ql) | df["body"].map(_normalize).str.contains(ql)
    hits = df[mask]
    if type_:
        hits = hits[hits["type"] == type_]
    return [_row_to_dict(r, _JSON_COLS["knowledge"]) for _, r in hits.iterrows()]


def get_knowledge_doc(doc_id: str) -> dict[str, Any] | None:
    return _find("knowledge", "id", doc_id)


# ---- Casos ----
def get_case(case_id: str) -> dict[str, Any] | None:
    return _find("cases", "id", case_id)


# ---- seed.json ----
@lru_cache(maxsize=1)
def seed_config() -> dict[str, Any]:
    return json.loads((DATA_DIR / "seed.json").read_text())
