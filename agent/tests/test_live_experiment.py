import asyncio

import httpx
import pytest

from tractian_agent.evaluation.contracts import BenchmarkInput
from tractian_agent.evaluation.live_experiment import _fetch_user_profile


def _case() -> BenchmarkInput:
    return BenchmarkInput(
        id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        company_id="comp_aurora",
        user_id="usr_lucas",
        asset_id="asset_B204",
        message="O que significa BPFO?",
    )


def test_live_runtime_fetches_permissions_from_trusted_api_boundary() -> None:
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-user-id"] == "usr_lucas"
            return httpx.Response(
                200,
                json={
                    "id": "usr_lucas",
                    "company_id": "comp_aurora",
                    "name": "Lucas",
                    "role": "mechanic",
                    "permissions": ["read", "action_low"],
                },
            )

        async with httpx.AsyncClient(
            base_url="https://simulator.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await _fetch_user_profile(client, _case())

    profile = asyncio.run(scenario())

    assert profile.permissions == frozenset({"read", "action_low"})


def test_live_runtime_rejects_cross_company_identity() -> None:
    async def scenario():
        async with httpx.AsyncClient(
            base_url="https://simulator.test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "id": "usr_lucas",
                        "company_id": "comp_other",
                        "name": "Lucas",
                        "role": "mechanic",
                        "permissions": ["read"],
                    },
                )
            ),
        ) as client:
            return await _fetch_user_profile(client, _case())

    with pytest.raises(ValueError, match="diverge"):
        asyncio.run(scenario())
