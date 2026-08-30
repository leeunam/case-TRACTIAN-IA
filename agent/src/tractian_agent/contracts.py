from __future__ import annotations

from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

PayloadT = TypeVar("PayloadT")
ArgumentsT = TypeVar("ArgumentsT", bound=BaseModel)
IdempotencyKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=255, pattern=r"^\S+$"),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identity(StrictModel):
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")


class SupportRequest(StrictModel):
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    ticket_id: str = Field(min_length=1, pattern=r"^\S+$")
    asset_id: str | None = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")
    identity: Identity


class ToolCall(StrictModel, Generic[ArgumentsT]):
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: ArgumentsT


class ActionReceipt(StrictModel):
    accepted: bool
    action_id: str = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")


class ResponseMode(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"


class ApiErrorCategory(str, Enum):
    API = "api"
    SERVER = "server"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    INVALID_RESPONSE = "invalid_response"


class ApiResult(StrictModel, Generic[PayloadT]):
    ok: Literal[True] = True
    status_code: int = Field(ge=200, le=299)
    data: PayloadT
    mode: ResponseMode | None = None
    notes: str | None = None


class ApiError(StrictModel):
    ok: Literal[False] = False
    category: ApiErrorCategory
    code: str = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")
    status_code: int | None = Field(default=None, ge=100, le=599)


ApiOutcome = ApiResult[PayloadT] | ApiError
