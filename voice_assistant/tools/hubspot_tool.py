from __future__ import annotations

import uuid
from typing import Any

import httpx

from config import Settings


class HubSpotTool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://api.hubapi.com"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {settings.hubspot_access_token}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )

    async def search_contact(self, query: str) -> dict[str, Any]:
        if self.settings.hubspot_mock or not self.settings.hubspot_access_token:
            return {
                "mode": "mock",
                "contacts": [],
                "message": f"No mock contact matched '{query}'.",
            }
        response = await self.client.post(
            "/crm/v3/objects/contacts/search",
            json={
                "query": query,
                "properties": ["email", "firstname", "lastname", "phone", "company"],
                "limit": 5,
            },
        )
        response.raise_for_status()
        return response.json()

    async def create_ticket(
        self, title: str, description: str, contact_email: str = ""
    ) -> dict[str, Any]:
        if self.settings.hubspot_mock or not self.settings.hubspot_access_token:
            return {
                "mode": "mock",
                "id": f"mock-{uuid.uuid4().hex[:8]}",
                "title": title,
                "message": "Ticket recorded in mock mode; nothing was sent to HubSpot.",
            }
        response = await self.client.post(
            "/crm/v3/objects/tickets",
            json={
                "properties": {
                    "subject": title,
                    "content": description,
                    "hs_pipeline": "0",
                    "hs_pipeline_stage": "1",
                }
            },
        )
        response.raise_for_status()
        ticket = response.json()
        if contact_email:
            contact_response = await self.search_contact(contact_email)
            results = contact_response.get("results", [])
            if results:
                await self.client.put(
                    f"/crm/v4/objects/tickets/{ticket['id']}/associations/contacts/{results[0]['id']}",
                    json=[{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 16}],
                )
        return {"mode": "live", "id": ticket.get("id"), "title": title}

    async def close(self) -> None:
        await self.client.aclose()
