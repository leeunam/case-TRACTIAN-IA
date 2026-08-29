"""Persistência SQLite para intenções idempotentes."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / ".run" / "idempotency.sqlite3"
RETENTION_DAYS = 7
DEFAULT_PROCESSING_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class IdempotencyRecord:
    idempotency_key: str
    user_id: str
    method: str
    endpoint: str
    payload_hash: str
    status: str
    response_status: int | None
    response_body: str | None
    created_at: str
    updated_at: str
    expires_at: str


@dataclass(frozen=True)
class ReservationResult:
    decision: str
    record: IdempotencyRecord


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"sha256:v1:{digest}"


class IdempotencyStore:
    def __init__(
        self,
        database_path: str | Path,
        *,
        processing_timeout_seconds: int | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.processing_timeout_seconds = (
            _processing_timeout_seconds_from_env()
            if processing_timeout_seconds is None
            else processing_timeout_seconds
        )
        if self.processing_timeout_seconds <= 0:
            raise ValueError("O timeout de processamento deve ser maior que zero.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_records (
                    idempotency_key TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_status INTEGER,
                    response_body TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    UNIQUE (user_id, method, endpoint, idempotency_key)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_idempotency_records_expires_at
                ON idempotency_records (expires_at)
                """
            )

    def find(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        method: str,
        endpoint: str,
    ) -> IdempotencyRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT idempotency_key, user_id, method, endpoint, payload_hash,
                       status, response_status, response_body,
                       created_at, updated_at, expires_at
                FROM idempotency_records
                WHERE user_id = ? AND method = ? AND endpoint = ?
                  AND idempotency_key = ?
                """,
                (user_id, method, endpoint, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return IdempotencyRecord(**dict(row))

    def reserve(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        method: str,
        endpoint: str,
        payload_hash: str,
    ) -> ReservationResult:
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(days=RETENTION_DAYS)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM idempotency_records WHERE expires_at <= ?",
                (created_at.isoformat(),),
            )
            row = connection.execute(
                """
                SELECT idempotency_key, user_id, method, endpoint, payload_hash,
                       status, response_status, response_body,
                       created_at, updated_at, expires_at
                FROM idempotency_records
                WHERE user_id = ? AND method = ? AND endpoint = ?
                  AND idempotency_key = ?
                """,
                (user_id, method, endpoint, idempotency_key),
            ).fetchone()

            if row is not None:
                record = IdempotencyRecord(**dict(row))
                if record.payload_hash != payload_hash:
                    return ReservationResult("payload_conflict", record)
                if record.status == "completed":
                    return ReservationResult("replay", record)
                if record.status == "processing":
                    last_update = datetime.fromisoformat(record.updated_at)
                    processing_deadline = last_update + timedelta(
                        seconds=self.processing_timeout_seconds
                    )
                    if created_at >= processing_deadline:
                        connection.execute(
                            """
                            UPDATE idempotency_records
                            SET status = 'uncertain', updated_at = ?
                            WHERE user_id = ? AND method = ? AND endpoint = ?
                              AND idempotency_key = ? AND status = 'processing'
                            """,
                            (
                                created_at.isoformat(),
                                user_id,
                                method,
                                endpoint,
                                idempotency_key,
                            ),
                        )
                        uncertain_record = replace(
                            record,
                            status="uncertain",
                            updated_at=created_at.isoformat(),
                        )
                        return ReservationResult(
                            "outcome_unknown",
                            uncertain_record,
                        )
                    return ReservationResult("in_progress", record)
                if record.status == "uncertain":
                    return ReservationResult("outcome_unknown", record)
                raise RuntimeError(
                    f"Estado idempotente não reconhecido: {record.status}"
                )

            connection.execute(
                """
                INSERT INTO idempotency_records (
                    idempotency_key, user_id, method, endpoint, payload_hash,
                    status, response_status, response_body,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, 'processing', NULL, NULL, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    user_id,
                    method,
                    endpoint,
                    payload_hash,
                    created_at.isoformat(),
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            record = IdempotencyRecord(
                idempotency_key=idempotency_key,
                user_id=user_id,
                method=method,
                endpoint=endpoint,
                payload_hash=payload_hash,
                status="processing",
                response_status=None,
                response_body=None,
                created_at=created_at.isoformat(),
                updated_at=created_at.isoformat(),
                expires_at=expires_at.isoformat(),
            )
            return ReservationResult("execute", record)

    def complete(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        method: str,
        endpoint: str,
        payload_hash: str,
        reservation_created_at: str,
        response_status: int,
        response_body: dict[str, Any],
    ) -> None:
        updated_at = datetime.now(timezone.utc)
        serialized_response = json.dumps(
            response_body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE idempotency_records
                SET status = 'completed', response_status = ?, response_body = ?,
                    updated_at = ?
                WHERE user_id = ? AND method = ? AND endpoint = ?
                  AND idempotency_key = ? AND payload_hash = ?
                  AND created_at = ? AND status = 'processing'
                """,
                (
                    response_status,
                    serialized_response,
                    updated_at.isoformat(),
                    user_id,
                    method,
                    endpoint,
                    idempotency_key,
                    payload_hash,
                    reservation_created_at,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Reserva idempotente não encontrada para conclusão.")

    def mark_uncertain(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        method: str,
        endpoint: str,
        payload_hash: str,
        reservation_created_at: str,
    ) -> None:
        updated_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE idempotency_records
                SET status = 'uncertain', updated_at = ?
                WHERE user_id = ? AND method = ? AND endpoint = ?
                  AND idempotency_key = ? AND payload_hash = ?
                  AND created_at = ? AND status = 'processing'
                """,
                (
                    updated_at.isoformat(),
                    user_id,
                    method,
                    endpoint,
                    idempotency_key,
                    payload_hash,
                    reservation_created_at,
                ),
            )


def _processing_timeout_seconds_from_env() -> int:
    raw_value = os.environ.get(
        "IDEMPOTENCY_PROCESSING_TIMEOUT_SECONDS",
        str(DEFAULT_PROCESSING_TIMEOUT_SECONDS),
    )
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "IDEMPOTENCY_PROCESSING_TIMEOUT_SECONDS deve ser um número inteiro."
        ) from exc


@lru_cache(maxsize=1)
def get_idempotency_store() -> IdempotencyStore:
    database_path = os.environ.get("IDEMPOTENCY_DB_PATH", str(DEFAULT_DB_PATH))
    return IdempotencyStore(database_path)
