from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DemoSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_path: Path = PROJECT_ROOT / ".run/demo.sqlite3"
    checkpoint_path: Path = PROJECT_ROOT / ".run/demo-agent-checkpoints.sqlite3"
    public_cases_path: Path = PROJECT_ROOT / "agent-input/cases.json"
    industrial_api_url: str = "http://127.0.0.1:8000"
    public_app_url: str = "http://127.0.0.1:5173"
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    primary_provider: str = Field(default="groq", pattern=r"^(groq|nvidia-nim)$")
    fallback_provider: str = Field(default="nvidia-nim", pattern=r"^(groq|nvidia-nim)$")
    planner_model: str = "openai/gpt-oss-20b"
    writer_model: str = "openai/gpt-oss-20b"
    slack_tractian_channel: str | None = None
    slack_authority_channel: str | None = None
    slack_access_token_configured: bool = False

    @model_validator(mode="after")
    def different_providers(self) -> "DemoSettings":
        if self.primary_provider == self.fallback_provider:
            raise ValueError("primary e fallback devem ser providers diferentes")
        return self

    @classmethod
    def from_env(cls, environment: Mapping[str, str]) -> "DemoSettings":
        origins = environment.get(
            "DEMO_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
        )
        return cls(
            database_path=Path(
                environment.get(
                    "DEMO_DATABASE_PATH", cls.model_fields["database_path"].default
                )
            ),
            checkpoint_path=Path(
                environment.get(
                    "DEMO_CHECKPOINT_PATH", cls.model_fields["checkpoint_path"].default
                )
            ),
            public_cases_path=Path(
                environment.get(
                    "DEMO_PUBLIC_CASES_PATH",
                    cls.model_fields["public_cases_path"].default,
                )
            ),
            industrial_api_url=environment.get(
                "INDUSTRIAL_API_URL", "http://127.0.0.1:8000"
            ),
            public_app_url=environment.get("PUBLIC_APP_URL", "http://127.0.0.1:5173"),
            allowed_origins=tuple(
                item.strip() for item in origins.split(",") if item.strip()
            ),
            primary_provider=environment.get("DEMO_PRIMARY_PROVIDER", "groq"),
            fallback_provider=environment.get("DEMO_FALLBACK_PROVIDER", "nvidia-nim"),
            planner_model=environment.get("DEMO_PLANNER_MODEL", "openai/gpt-oss-20b"),
            writer_model=environment.get("DEMO_WRITER_MODEL", "openai/gpt-oss-20b"),
            slack_tractian_channel=environment.get("SLACK_TRACTIAN_CHANNEL_ID"),
            slack_authority_channel=environment.get("SLACK_AUTHORITY_CHANNEL_ID"),
            slack_access_token_configured=bool(
                environment.get("SLACK_MCP_ACCESS_TOKEN")
            ),
        )
