from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Iterator
from uuid import uuid4

from tractian_demo.contracts import (
    AgentRunProjection,
    CaseEvent,
    CaseMessage,
    CreateCaseRequest,
    DemoCase,
    Execution,
    ExecutionStatus,
    DecisionCandidate,
    DecisionRequest,
    DecisionStatus,
    DeliveryStatus,
    OutboxEvent,
    Persona,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


class DemoRepository:
    """SQLite local com transações explícitas e projeções públicas fechadas."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def open(self, *, public_cases: Iterable[Mapping[str, object]] = ()) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=10
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        self._connection = connection
        self._migrate()
        self._seed_public_cases(public_cases)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("repository is not open")
        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _migrate(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
                    company_id TEXT NOT NULL, requester_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL, initial_message TEXT NOT NULL,
                    source_case_id TEXT, immutable INTEGER NOT NULL,
                    simulation_mode TEXT NOT NULL DEFAULT 'standard', seed TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_messages (
                    id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id),
                    persona_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                    idempotency_key TEXT, payload_hash TEXT, created_at TEXT NOT NULL,
                    UNIQUE(case_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id),
                    message_id TEXT NOT NULL UNIQUE REFERENCES case_messages(id),
                    status TEXT NOT NULL, provider TEXT, fallback_reason TEXT,
                    trace_id TEXT, error_code TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT, lease_expires_at TEXT,
                    resume_decision_id TEXT, resume_kind TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(id), ordinal INTEGER NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(case_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS decision_requests (
                    id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id),
                    execution_id TEXT NOT NULL REFERENCES executions(id),
                    company_id TEXT NOT NULL, audience TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
                    subject_digest TEXT NOT NULL, scope_json TEXT NOT NULL,
                    summary TEXT NOT NULL, required_permission TEXT, resume_kind TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL, resolved_at TEXT, resolved_by TEXT,
                    resolution_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS decision_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL REFERENCES decision_requests(id),
                    kind TEXT NOT NULL, actor_id TEXT, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outbox_events (
                    id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decision_requests(id),
                    audience TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
                    available_at TEXT NOT NULL, lease_owner TEXT, lease_expires_at TEXT,
                    external_id TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS delivery_attempts (
                    id TEXT PRIMARY KEY, outbox_id TEXT NOT NULL REFERENCES outbox_events(id),
                    status TEXT NOT NULL, error_code TEXT, external_id TEXT,
                    started_at TEXT NOT NULL, finished_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_decision_request_execution
                ON decision_requests(execution_id);
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(decision_requests)")
            }
            additions = {
                "company_id": "TEXT NOT NULL DEFAULT ''",
                "summary": "TEXT NOT NULL DEFAULT 'Decisão pendente'",
                "resume_kind": "TEXT NOT NULL DEFAULT 'technical_review'",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE decision_requests ADD COLUMN {name} {definition}"
                    )
            execution_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(executions)")
            }
            for name in ("resume_decision_id", "resume_kind"):
                if name not in execution_columns:
                    connection.execute(f"ALTER TABLE executions ADD COLUMN {name} TEXT")
            case_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(cases)")
            }
            if "simulation_mode" not in case_columns:
                connection.execute(
                    "ALTER TABLE cases ADD COLUMN simulation_mode TEXT NOT NULL DEFAULT 'standard'"
                )
            if "seed" not in case_columns:
                connection.execute("ALTER TABLE cases ADD COLUMN seed TEXT")
            connection.execute("PRAGMA user_version = 2")

    def _seed_public_cases(self, cases: Iterable[Mapping[str, object]]) -> None:
        now = _iso(utc_now())
        with self.transaction() as connection:
            for item in cases:
                connection.execute(
                    """INSERT OR IGNORE INTO cases
                    (id,ticket_id,company_id,requester_id,asset_id,initial_message,
                     source_case_id,immutable,simulation_mode,seed,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        item["id"],
                        item["ticket_id"],
                        item["company_id"],
                        item["user_id"],
                        item["asset_id"],
                        item["message"],
                        None,
                        1,
                        "standard",
                        None,
                        now,
                    ),
                )

    def create_case(self, request: CreateCaseRequest) -> DemoCase:
        now = utc_now()
        case_id = f"case_demo_{uuid4().hex}"
        with self.transaction() as connection:
            if request.source_case_id is not None:
                source = connection.execute(
                    "SELECT * FROM cases WHERE id = ? AND immutable = 1",
                    (request.source_case_id,),
                ).fetchone()
                if source is None:
                    raise KeyError("PUBLIC_CASE_NOT_FOUND")
                values = (
                    case_id,
                    source["ticket_id"],
                    request.company_id or source["company_id"],
                    request.requester_id or source["requester_id"],
                    request.asset_id or source["asset_id"],
                    request.message or source["initial_message"],
                    source["id"],
                    0,
                    request.simulation_mode,
                    request.seed,
                    _iso(now),
                )
            else:
                assert request.company_id and request.requester_id
                assert request.asset_id and request.message
                values = (
                    case_id,
                    f"DEMO-{uuid4().hex[:8].upper()}",
                    request.company_id,
                    request.requester_id,
                    request.asset_id,
                    request.message,
                    None,
                    0,
                    request.simulation_mode,
                    request.seed,
                    _iso(now),
                )
            connection.execute(
                """INSERT INTO cases
                (id,ticket_id,company_id,requester_id,asset_id,initial_message,
                 source_case_id,immutable,simulation_mode,seed,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> DemoCase:
        row = self.connection.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        if row is None:
            raise KeyError("CASE_NOT_FOUND")
        return self._case(row)

    def list_cases(self) -> tuple[DemoCase, ...]:
        rows = self.connection.execute(
            "SELECT * FROM cases ORDER BY immutable DESC, created_at DESC, id"
        ).fetchall()
        return tuple(self._case(row) for row in rows)

    @staticmethod
    def _case(row: sqlite3.Row) -> DemoCase:
        return DemoCase(
            id=row["id"],
            ticket_id=row["ticket_id"],
            company_id=row["company_id"],
            requester_id=row["requester_id"],
            asset_id=row["asset_id"],
            initial_message=row["initial_message"],
            source_case_id=row["source_case_id"],
            immutable=bool(row["immutable"]),
            created_at=_dt(row["created_at"]),
            simulation_mode=row["simulation_mode"],
            seed=row["seed"],
        )

    def enqueue_message(
        self,
        *,
        case_id: str,
        persona_id: str,
        content: str,
        idempotency_key: str,
        _fail_after_message_for_test: bool = False,
    ) -> tuple[CaseMessage, Execution]:
        payload_hash = hashlib.sha256(
            json.dumps(
                {"persona_id": persona_id, "content": content},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        now = utc_now()
        with self.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM cases WHERE id = ?", (case_id,)
                ).fetchone()
                is None
            ):
                raise KeyError("CASE_NOT_FOUND")
            previous = connection.execute(
                "SELECT * FROM case_messages WHERE case_id = ? AND idempotency_key = ?",
                (case_id, idempotency_key),
            ).fetchone()
            if previous is not None:
                if previous["payload_hash"] != payload_hash:
                    raise ValueError("IDEMPOTENCY_CONFLICT")
                execution = connection.execute(
                    "SELECT * FROM executions WHERE message_id = ?", (previous["id"],)
                ).fetchone()
                assert execution is not None
                return self._message(previous), self._execution(execution)
            message_id = f"msg_{uuid4().hex}"
            execution_id = f"exec_{uuid4().hex}"
            connection.execute(
                """INSERT INTO case_messages
                (id,case_id,persona_id,role,content,idempotency_key,payload_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    message_id,
                    case_id,
                    persona_id,
                    "user",
                    content,
                    idempotency_key,
                    payload_hash,
                    _iso(now),
                ),
            )
            if _fail_after_message_for_test:
                raise RuntimeError("injected transaction failure")
            connection.execute(
                """INSERT INTO executions
                (id,case_id,message_id,status,attempt,created_at,updated_at)
                VALUES (?,?,?,?,0,?,?)""",
                (
                    execution_id,
                    case_id,
                    message_id,
                    ExecutionStatus.QUEUED.value,
                    _iso(now),
                    _iso(now),
                ),
            )
            self._append_event_tx(
                connection,
                case_id,
                "execution.queued",
                {"execution_id": execution_id, "message_id": message_id},
                now,
            )
        return self.list_messages(case_id)[-1], self.get_execution(execution_id)

    def list_messages(self, case_id: str) -> tuple[CaseMessage, ...]:
        rows = self.connection.execute(
            "SELECT * FROM case_messages WHERE case_id = ? ORDER BY created_at, id",
            (case_id,),
        ).fetchall()
        return tuple(self._message(row) for row in rows)

    def get_message(self, message_id: str) -> CaseMessage:
        row = self.connection.execute(
            "SELECT * FROM case_messages WHERE id=?", (message_id,)
        ).fetchone()
        if row is None:
            raise KeyError("MESSAGE_NOT_FOUND")
        return self._message(row)

    @staticmethod
    def _message(row: sqlite3.Row) -> CaseMessage:
        return CaseMessage(
            id=row["id"],
            case_id=row["case_id"],
            persona_id=row["persona_id"],
            role=row["role"],
            content=row["content"],
            created_at=_dt(row["created_at"]),
        )

    def get_execution(self, execution_id: str) -> Execution:
        row = self.connection.execute(
            "SELECT * FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if row is None:
            raise KeyError("EXECUTION_NOT_FOUND")
        return self._execution(row)

    def list_executions(self, case_id: str) -> tuple[Execution, ...]:
        rows = self.connection.execute(
            "SELECT * FROM executions WHERE case_id = ? ORDER BY created_at, id",
            (case_id,),
        ).fetchall()
        return tuple(self._execution(row) for row in rows)

    @staticmethod
    def _execution(row: sqlite3.Row) -> Execution:
        return Execution(
            id=row["id"],
            case_id=row["case_id"],
            message_id=row["message_id"],
            status=ExecutionStatus(row["status"]),
            provider=row["provider"],
            fallback_reason=row["fallback_reason"],
            trace_id=row["trace_id"],
            error_code=row["error_code"],
            attempt=row["attempt"],
            lease_owner=row["lease_owner"],
            lease_expires_at=_dt(row["lease_expires_at"]),
            resume_decision_id=row["resume_decision_id"],
            resume_kind=row["resume_kind"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def claim_execution(
        self, *, worker_id: str, lease_seconds: int = 30
    ) -> Execution | None:
        now = utc_now()
        expiry = now + timedelta(seconds=lease_seconds)
        with self.transaction() as connection:
            row = connection.execute(
                """SELECT * FROM executions
                WHERE status = 'queued' OR
                  (status = 'running' AND lease_expires_at < ? AND error_code IS NULL)
                ORDER BY created_at, id LIMIT 1""",
                (_iso(now),),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE executions SET status='running', lease_owner=?, lease_expires_at=?,
                attempt=attempt+1, updated_at=? WHERE id=?""",
                (worker_id, _iso(expiry), _iso(now), row["id"]),
            )
            self._append_event_tx(
                connection,
                row["case_id"],
                "execution.running",
                {"execution_id": row["id"]},
                now,
            )
        return self.get_execution(row["id"])

    def complete_execution(
        self, execution_id: str, projection: AgentRunProjection
    ) -> Execution:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
            if row is None:
                raise KeyError("EXECUTION_NOT_FOUND")
            if row["status"] != ExecutionStatus.RUNNING.value:
                raise ValueError("EXECUTION_NOT_RUNNING")
            message_id = f"msg_{uuid4().hex}"
            connection.execute(
                """INSERT INTO case_messages
                (id,case_id,persona_id,role,content,created_at)
                VALUES (?,?,?,?,?,?)""",
                (
                    message_id,
                    row["case_id"],
                    "tractian_agent",
                    "assistant",
                    projection.assistant_message,
                    _iso(now),
                ),
            )
            connection.execute(
                """UPDATE executions SET status='completed', provider=?, fallback_reason=?,
                trace_id=?, lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE id=?""",
                (
                    projection.provider,
                    projection.fallback_reason,
                    projection.trace_id,
                    _iso(now),
                    execution_id,
                ),
            )
            self._append_event_tx(
                connection,
                row["case_id"],
                "tools.completed",
                {"tool_names": list(projection.tool_names)},
                now,
            )
            self._append_event_tx(
                connection,
                row["case_id"],
                "agent.completed",
                {
                    "execution_id": execution_id,
                    "decision": projection.decision,
                    "provider": projection.provider,
                    "fallback_reason": projection.fallback_reason,
                    "evidence_count": projection.evidence_count,
                    "limitation_count": projection.limitation_count,
                    "trace_id": projection.trace_id,
                },
                now,
            )
        return self.get_execution(execution_id)

    def wait_for_decision(
        self, execution_id: str, projection: AgentRunProjection
    ) -> DecisionRequest:
        candidate = projection.decision_candidate
        if candidate is None:
            raise ValueError("DECISION_CANDIDATE_REQUIRED")
        execution = self.get_execution(execution_id)
        case = self.get_case(execution.case_id)
        return self.create_decision(
            case=case,
            execution=execution,
            candidate=candidate,
            projection=projection,
        )

    def fail_execution(self, execution_id: str, *, error_code: str) -> Execution:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
            if row is None:
                raise KeyError("EXECUTION_NOT_FOUND")
            connection.execute(
                """UPDATE executions SET status='failed', error_code=?, lease_owner=NULL,
                lease_expires_at=NULL, updated_at=? WHERE id=?""",
                (error_code, _iso(now), execution_id),
            )
            self._append_event_tx(
                connection,
                row["case_id"],
                "agent.failed",
                {"execution_id": execution_id, "error_code": error_code},
                now,
            )
        return self.get_execution(execution_id)

    def create_decision(
        self,
        *,
        case: DemoCase,
        execution: Execution,
        candidate: DecisionCandidate,
        projection: AgentRunProjection | None = None,
    ) -> DecisionRequest:
        now = utc_now()
        if candidate.expires_at <= now:
            raise ValueError("DECISION_ALREADY_EXPIRED")
        decision_id = f"decision_{uuid4().hex}"
        scope_json = json.dumps(candidate.scope, sort_keys=True, separators=(",", ":"))
        subject_digest = f"sha256:v1:{hashlib.sha256(scope_json.encode()).hexdigest()}"
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM decision_requests WHERE execution_id=?", (execution.id,)
            ).fetchone()
            if existing is not None:
                return self._decision(existing)
            current = connection.execute(
                "SELECT status FROM executions WHERE id=?", (execution.id,)
            ).fetchone()
            if current is None or current["status"] != ExecutionStatus.RUNNING.value:
                raise ValueError("EXECUTION_NOT_RUNNING")
            if projection is not None:
                connection.execute(
                    """INSERT INTO case_messages
                    (id,case_id,persona_id,role,content,created_at) VALUES (?,?,?,?,?,?)""",
                    (
                        f"msg_agent_{execution.id}",
                        case.id,
                        "tractian_agent",
                        "assistant",
                        projection.assistant_message,
                        _iso(now),
                    ),
                )
                connection.execute(
                    """UPDATE executions SET provider=?,fallback_reason=?,trace_id=?,updated_at=?
                    WHERE id=?""",
                    (
                        projection.provider,
                        projection.fallback_reason,
                        projection.trace_id,
                        _iso(now),
                        execution.id,
                    ),
                )
            connection.execute(
                """INSERT INTO decision_requests
                (id,case_id,execution_id,company_id,audience,kind,status,subject_digest,
                 scope_json,summary,required_permission,resume_kind,expires_at,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    decision_id,
                    case.id,
                    execution.id,
                    case.company_id,
                    candidate.audience,
                    candidate.kind,
                    DecisionStatus.PENDING.value,
                    subject_digest,
                    scope_json,
                    candidate.summary,
                    candidate.required_permission,
                    candidate.resume_kind,
                    _iso(candidate.expires_at),
                    _iso(now),
                ),
            )
            connection.execute(
                "UPDATE executions SET status='waiting_human', lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE id=?",
                (_iso(now), execution.id),
            )
            self._append_event_tx(
                connection,
                case.id,
                "decision.requested",
                {
                    "decision_id": decision_id,
                    "audience": candidate.audience,
                    "kind": candidate.kind,
                },
                now,
            )
            if candidate.audience in {"tractian", "authority"}:
                notification_id = f"notification_{uuid4().hex}"
                payload = {
                    "decision_id": decision_id,
                    "category": candidate.kind,
                    "summary": candidate.summary,
                    "case_id": case.id,
                }
                connection.execute(
                    """INSERT INTO outbox_events
                    (id,decision_id,audience,status,payload,available_at,attempt,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,0,?,?)""",
                    (
                        notification_id,
                        decision_id,
                        candidate.audience,
                        DeliveryStatus.PENDING.value,
                        json.dumps(payload, sort_keys=True),
                        _iso(now),
                        _iso(now),
                        _iso(now),
                    ),
                )
        return self.get_decision(decision_id)

    def get_decision(self, decision_id: str) -> DecisionRequest:
        row = self.connection.execute(
            "SELECT * FROM decision_requests WHERE id=?", (decision_id,)
        ).fetchone()
        if row is None:
            raise KeyError("DECISION_NOT_FOUND")
        return self._decision(row)

    def list_decisions(self, persona: Persona) -> tuple[DecisionRequest, ...]:
        now = _iso(utc_now())
        rows = self.connection.execute(
            "SELECT * FROM decision_requests WHERE status='pending' AND expires_at>? ORDER BY created_at",
            (now,),
        ).fetchall()
        values = []
        for row in rows:
            allowed = (
                (
                    row["audience"] == "requester"
                    and persona.id == self.get_case(row["case_id"]).requester_id
                )
                or (row["audience"] == "tractian" and persona.profile == "tractian")
                or (
                    row["audience"] == "authority"
                    and persona.profile == "authority"
                    and persona.company_id == row["company_id"]
                    and row["required_permission"] in persona.permissions
                )
            )
            if allowed:
                values.append(self._decision(row))
        return tuple(values)

    def resolve_decision(
        self, decision_id: str, *, persona: Persona, resolution: str
    ) -> DecisionRequest:
        if resolution not in {"approve", "reject"}:
            raise ValueError("INVALID_RESOLUTION")
        now = utc_now()
        resolution_hash = hashlib.sha256(
            f"{persona.id}\0{resolution}".encode()
        ).hexdigest()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM decision_requests WHERE id=?", (decision_id,)
            ).fetchone()
            if row is None:
                raise KeyError("DECISION_NOT_FOUND")
            allowed = (
                (
                    row["audience"] == "requester"
                    and persona.id == self.get_case(row["case_id"]).requester_id
                )
                or (row["audience"] == "tractian" and persona.profile == "tractian")
                or (
                    row["audience"] == "authority"
                    and persona.profile == "authority"
                    and persona.company_id == row["company_id"]
                    and row["required_permission"] in persona.permissions
                )
            )
            if not allowed:
                raise PermissionError("DECISION_FORBIDDEN")
            if row["status"] != DecisionStatus.PENDING.value:
                if row["resolution_hash"] == resolution_hash:
                    return self._decision(row)
                raise ValueError("DECISION_ALREADY_RESOLVED")
            if _dt(row["expires_at"]) <= now:
                connection.execute(
                    "UPDATE decision_requests SET status='expired' WHERE id=?",
                    (decision_id,),
                )
                raise ValueError("DECISION_EXPIRED")
            status = (
                DecisionStatus.APPROVED
                if resolution == "approve"
                else DecisionStatus.REJECTED
            )
            connection.execute(
                """UPDATE decision_requests SET status=?,resolved_at=?,resolved_by=?,resolution_hash=?
                WHERE id=? AND status='pending'""",
                (status.value, _iso(now), persona.id, resolution_hash, decision_id),
            )
            connection.execute(
                "INSERT INTO decision_events(decision_id,kind,actor_id,payload,created_at) VALUES (?,?,?,?,?)",
                (
                    decision_id,
                    f"decision.{status.value}",
                    persona.id,
                    json.dumps({"resolution": resolution}),
                    _iso(now),
                ),
            )
            self._append_event_tx(
                connection,
                row["case_id"],
                f"decision.{status.value}",
                {"decision_id": decision_id},
                now,
            )
            if row["resume_kind"] != "acknowledgement":
                system_message_id = f"msg_{uuid4().hex}"
                resume_execution_id = f"exec_{uuid4().hex}"
                connection.execute(
                    """INSERT INTO case_messages
                    (id,case_id,persona_id,role,content,created_at) VALUES (?,?,?,?,?,?)""",
                    (
                        system_message_id,
                        row["case_id"],
                        persona.id,
                        "system",
                        f"Decisão {resolution} registrada.",
                        _iso(now),
                    ),
                )
                connection.execute(
                    """INSERT INTO executions
                    (id,case_id,message_id,status,attempt,resume_decision_id,resume_kind,created_at,updated_at)
                    VALUES (?,?,?,?,0,?,?,?,?)""",
                    (
                        resume_execution_id,
                        row["case_id"],
                        system_message_id,
                        ExecutionStatus.QUEUED.value,
                        decision_id,
                        row["resume_kind"],
                        _iso(now),
                        _iso(now),
                    ),
                )
                self._append_event_tx(
                    connection,
                    row["case_id"],
                    "execution.queued",
                    {"execution_id": resume_execution_id, "decision_id": decision_id},
                    now,
                )
        return self.get_decision(decision_id)

    @staticmethod
    def _decision(row: sqlite3.Row) -> DecisionRequest:
        return DecisionRequest(
            id=row["id"],
            case_id=row["case_id"],
            execution_id=row["execution_id"],
            company_id=row["company_id"],
            audience=row["audience"],
            kind=row["kind"],
            status=DecisionStatus(row["status"]),
            subject_digest=row["subject_digest"],
            summary=row["summary"],
            scope=json.loads(row["scope_json"]),
            required_permission=row["required_permission"],
            resume_kind=row["resume_kind"],
            expires_at=_dt(row["expires_at"]),
            created_at=_dt(row["created_at"]),
            resolved_at=_dt(row["resolved_at"]),
            resolved_by=row["resolved_by"],
        )

    def claim_outbox(
        self, *, worker_id: str, lease_seconds: int = 30
    ) -> OutboxEvent | None:
        now = utc_now()
        expiry = now + timedelta(seconds=lease_seconds)
        with self.transaction() as connection:
            row = connection.execute(
                """SELECT * FROM outbox_events WHERE
                (status='pending' AND available_at<=?) OR
                (status='delivering' AND lease_expires_at<?)
                ORDER BY created_at LIMIT 1""",
                (_iso(now), _iso(now)),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE outbox_events SET status='delivering',lease_owner=?,lease_expires_at=?,
                attempt=attempt+1,updated_at=? WHERE id=?""",
                (worker_id, _iso(expiry), _iso(now), row["id"]),
            )
        return self.get_outbox(row["id"])

    def get_outbox(self, outbox_id: str) -> OutboxEvent:
        row = self.connection.execute(
            "SELECT * FROM outbox_events WHERE id=?", (outbox_id,)
        ).fetchone()
        if row is None:
            raise KeyError("NOTIFICATION_NOT_FOUND")
        return OutboxEvent(
            id=row["id"],
            decision_id=row["decision_id"],
            audience=row["audience"],
            status=DeliveryStatus(row["status"]),
            payload=json.loads(row["payload"]),
            attempt=row["attempt"],
            lease_owner=row["lease_owner"],
            lease_expires_at=_dt(row["lease_expires_at"]),
            external_id=row["external_id"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def finish_outbox(
        self,
        outbox_id: str,
        *,
        status: DeliveryStatus,
        external_id: str | None = None,
        error_code: str | None = None,
    ) -> OutboxEvent:
        if status not in {
            DeliveryStatus.DELIVERED,
            DeliveryStatus.FAILED,
            DeliveryStatus.UNCERTAIN,
        }:
            raise ValueError("INVALID_DELIVERY_TERMINAL")
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM outbox_events WHERE id=?", (outbox_id,)
            ).fetchone()
            if row is None:
                raise KeyError("NOTIFICATION_NOT_FOUND")
            if row["status"] != DeliveryStatus.DELIVERING.value:
                raise ValueError("NOTIFICATION_NOT_DELIVERING")
            attempt_id = f"delivery_{uuid4().hex}"
            connection.execute(
                "INSERT INTO delivery_attempts(id,outbox_id,status,error_code,external_id,started_at,finished_at) VALUES (?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    outbox_id,
                    status.value,
                    error_code,
                    external_id,
                    row["updated_at"],
                    _iso(now),
                ),
            )
            connection.execute(
                "UPDATE outbox_events SET status=?,external_id=?,lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE id=?",
                (status.value, external_id, _iso(now), outbox_id),
            )
        return self.get_outbox(outbox_id)

    def retry_outbox(self, outbox_id: str) -> OutboxEvent:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM outbox_events WHERE id=?", (outbox_id,)
            ).fetchone()
            if row is None:
                raise KeyError("NOTIFICATION_NOT_FOUND")
            if row["status"] not in {
                DeliveryStatus.FAILED.value,
                DeliveryStatus.UNCERTAIN.value,
            }:
                raise ValueError("NOTIFICATION_RETRY_FORBIDDEN")
            connection.execute(
                "UPDATE outbox_events SET status='pending',available_at=?,external_id=NULL,updated_at=? WHERE id=?",
                (_iso(now), _iso(now), outbox_id),
            )
        return self.get_outbox(outbox_id)

    def append_event(
        self, case_id: str, kind: str, payload: Mapping[str, object]
    ) -> CaseEvent:
        now = utc_now()
        with self.transaction() as connection:
            event_id = self._append_event_tx(connection, case_id, kind, payload, now)
        row = self.connection.execute(
            "SELECT * FROM case_events WHERE id=?", (event_id,)
        ).fetchone()
        assert row is not None
        return self._event(row)

    @staticmethod
    def _append_event_tx(
        connection: sqlite3.Connection,
        case_id: str,
        kind: str,
        payload: Mapping[str, object],
        now: datetime,
    ) -> int:
        ordinal = connection.execute(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM case_events WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
        cursor = connection.execute(
            "INSERT INTO case_events(case_id,ordinal,kind,payload,created_at) VALUES (?,?,?,?,?)",
            (
                case_id,
                ordinal,
                kind,
                json.dumps(dict(payload), sort_keys=True),
                _iso(now),
            ),
        )
        return int(cursor.lastrowid)

    def list_events(self, case_id: str, *, after_id: int = 0) -> tuple[CaseEvent, ...]:
        rows = self.connection.execute(
            "SELECT * FROM case_events WHERE case_id=? AND id>? ORDER BY id",
            (case_id, after_id),
        ).fetchall()
        return tuple(self._event(row) for row in rows)

    @staticmethod
    def _event(row: sqlite3.Row) -> CaseEvent:
        return CaseEvent(
            id=row["id"],
            case_id=row["case_id"],
            ordinal=row["ordinal"],
            kind=row["kind"],
            payload=json.loads(row["payload"]),
            created_at=_dt(row["created_at"]),
        )
