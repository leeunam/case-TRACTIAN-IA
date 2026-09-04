from __future__ import annotations

import httpx

from tractian_demo.contracts import DeliveryStatus
from tractian_demo.repository import DemoRepository
from tractian_demo.slack_mcp import SlackMcpClient, SlackMcpProtocolError


class SlackDeliveryWorker:
    def __init__(
        self,
        repository: DemoRepository,
        client: SlackMcpClient,
        *,
        worker_id: str,
        public_app_url: str,
        tractian_channel: str,
        authority_channel: str,
    ) -> None:
        self._repository = repository
        self._client = client
        self._worker_id = worker_id
        self._public_app_url = public_app_url.rstrip("/")
        self._channels = {"tractian": tractian_channel, "authority": authority_channel}

    async def run_once(self) -> bool:
        item = self._repository.claim_outbox(worker_id=self._worker_id)
        if item is None:
            return False
        channel = self._channels[item.audience]
        text = (
            f"[{item.payload['category']}] {item.payload['summary']}\n"
            f"Decida na central: {self._public_app_url}/?decision={item.decision_id}"
        )
        try:
            external_id = await self._client.send_message(channel_id=channel, text=text)
        except SlackMcpProtocolError as error:
            uncertain = str(error) == "SLACK_EXTERNAL_ID_MISSING"
            self._repository.finish_outbox(
                item.id,
                status=DeliveryStatus.UNCERTAIN if uncertain else DeliveryStatus.FAILED,
                error_code=str(error),
            )
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            self._repository.finish_outbox(
                item.id,
                status=(
                    DeliveryStatus.UNCERTAIN
                    if status == 429 or status >= 500
                    else DeliveryStatus.FAILED
                ),
                error_code=f"SLACK_HTTP_{status}",
            )
        except (httpx.TimeoutException, httpx.RequestError):
            self._repository.finish_outbox(
                item.id,
                status=DeliveryStatus.UNCERTAIN,
                error_code="SLACK_DELIVERY_AMBIGUOUS",
            )
        else:
            self._repository.finish_outbox(
                item.id, status=DeliveryStatus.DELIVERED, external_id=external_id
            )
        return True
