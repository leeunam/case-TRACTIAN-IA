from __future__ import annotations

from tractian_demo.contracts import DecisionCandidate


_REQUIRED_PERMISSION = {
    "reprocess_analysis": "action_low",
    "request_specialist_analysis": "action_low",
    "update_asset_criticality": "action_high",
    "request_model_retraining": "action_high",
    "escalate_case": "escalate",
}


def route_decision(
    *,
    action: str | None,
    requester_permissions: frozenset[str],
    technical_review: bool = False,
) -> DecisionCandidate:
    """Matriz pura: não confunde revisão TRACTIAN com autorização empresarial."""
    from datetime import datetime, timedelta, timezone

    expiry = datetime.now(timezone.utc) + timedelta(hours=24)
    if technical_review:
        return DecisionCandidate(
            audience="tractian",
            kind="technical_review",
            summary="Revisar o bloqueio técnico do gate.",
            scope={},
            resume_kind="technical_review",
            expires_at=expiry,
        )
    if action is None or action not in _REQUIRED_PERMISSION:
        raise ValueError("ação não roteável")
    required = _REQUIRED_PERMISSION[action]
    if action in {
        "request_specialist_analysis",
        "request_model_retraining",
        "escalate_case",
    }:
        audience = "tractian"
        kind = "technical_review"
        resume_kind = (
            "confirmation" if required in requester_permissions else "delegated_action"
        )
    elif required in requester_permissions:
        audience = "requester"
        kind = "action_confirmation"
        resume_kind = "confirmation"
    else:
        audience = "authority"
        kind = "action_authorization"
        resume_kind = "delegated_action"
    return DecisionCandidate(
        audience=audience,
        kind=kind,
        summary=f"Decidir a ação {action}.",
        scope={"action": action},
        required_permission=required,
        resume_kind=resume_kind,
        expires_at=expiry,
    )
