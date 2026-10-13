from __future__ import annotations

import xml.sax.saxutils
from typing import Any

import httpx

from config import Settings


class HandoffTool:
    """Warm-transfer an active Twilio call through a conference, or report mock mode."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def transfer(self, reason: str) -> dict[str, Any]:
        configured = all(
            (
                self.settings.twilio_account_sid,
                self.settings.twilio_auth_token,
                self.settings.twilio_call_sid,
                self.settings.twilio_from_number,
                self.settings.handoff_number,
            )
        )
        if not configured:
            return {
                "mode": "mock",
                "transferred": False,
                "reason": reason,
                "message": "Handoff requested in mock mode; no call was transferred.",
            }

        account = self.settings.twilio_account_sid
        call_sid = self.settings.twilio_call_sid
        conference = f"support-{call_sid}"
        room = xml.sax.saxutils.escape(conference)
        twiml = f"<Response><Dial><Conference>{room}</Conference></Dial></Response>"
        api = f"https://api.twilio.com/2010-04-01/Accounts/{account}"
        auth = (account, self.settings.twilio_auth_token)
        async with httpx.AsyncClient(timeout=15.0, auth=auth) as client:
            agent_call = await client.post(
                f"{api}/Calls.json",
                data={
                    "To": self.settings.handoff_number,
                    "From": self.settings.twilio_from_number,
                    "Twiml": twiml,
                },
            )
            agent_call.raise_for_status()
            caller_call = await client.post(
                f"{api}/Calls/{call_sid}.json",
                data={"Twiml": twiml},
            )
            caller_call.raise_for_status()
        return {
            "mode": "live",
            "transferred": True,
            "agent_call_sid": agent_call.json().get("sid"),
            "reason": reason,
        }
