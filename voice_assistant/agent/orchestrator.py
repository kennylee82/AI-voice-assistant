from __future__ import annotations

import asyncio
import json
import re
from enum import Enum, auto
from typing import Any, Callable

from openai import AsyncOpenAI

from config import Settings
from agent.prompts import SYSTEM_PROMPT
from guards.confirmation import ConfirmationGate, ConfirmationState, PendingAction
from tools.handoff_tool import HandoffTool
from tools.hubspot_tool import HubSpotTool
from tools.rag_engine import RagEngine


class AgentState(Enum):
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    STOPPED = auto()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search company support and product information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_customer",
            "description": "Find a HubSpot contact by name, email, or phone.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a customer support ticket after asking for confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "contact_email": {"type": "string"},
                },
                "required": ["title", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Request a warm transfer to a human support agent after confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


class VoiceOrchestrator:
    def __init__(
        self,
        settings: Settings,
        audio=None,
        tts=None,
        on_text: Callable[[str], None] | None = None,
    ):
        self.settings = settings
        self.audio = audio
        self.tts = tts
        self.on_text = on_text
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.hubspot = HubSpotTool(settings)
        self.handoff = HandoffTool(settings)
        self.rag = RagEngine()
        self.confirmation = ConfirmationGate()
        self.state = AgentState.LISTENING
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def turn(self, user_text: str) -> str:
        state, pending = self.confirmation.resolve(user_text)
        if state is ConfirmationState.PENDING:
            response = "Please say yes to confirm, or no to cancel."
            await self._deliver(response)
            return response
        if state is ConfirmationState.DECLINED:
            response = "Understood. I did not make that change."
            await self._deliver(response)
            self._remember(user_text, response)
            return response
        if state is ConfirmationState.CONFIRMED and pending is not None:
            result = await self._execute_confirmed(pending)
            response = self._format_action_result(pending.name, result)
            await self._deliver(response)
            self._remember(user_text, response)
            return response

        self.state = AgentState.THINKING
        self.messages.append({"role": "user", "content": user_text})
        final_text = ""
        for _ in range(4):
            content, tool_calls = await self._stream_completion()
            final_text += content
            if not tool_calls:
                break

            self.messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            confirmation_requested = False
            for call in tool_calls:
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                name = call["name"]
                if name in {"create_support_ticket", "transfer_to_human"}:
                    message = self.confirmation.request(name, arguments)
                    result: dict[str, Any] = {"confirmation_required": True}
                    confirmation_requested = True
                else:
                    result = await self._execute_tool(name, arguments)
                    message = json.dumps(result, ensure_ascii=True)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": message,
                    }
                )
                if confirmation_requested:
                    break
            if confirmation_requested:
                response = message
                await self._deliver(response)
                final_text += response
                break
        self._remember(user_text, final_text, include_user=False)
        self.state = AgentState.LISTENING
        return final_text

    async def _stream_completion(self) -> tuple[str, list[dict[str, str]]]:
        response = await self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[self.messages[0], *self.messages[-12:]],
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.4,
            stream=True,
        )
        content_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        speech_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        speaker_task = asyncio.create_task(self._deliver_queued(speech_queue))
        speech_buffer = ""
        try:
            async for event in response:
                delta = event.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    speech_buffer += delta.content
                    complete, speech_buffer = self._split_speech(speech_buffer)
                    for sentence in complete:
                        await speech_queue.put(sentence)
                for tool_delta in delta.tool_calls or []:
                    call = calls.setdefault(
                        tool_delta.index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if tool_delta.id:
                        call["id"] = tool_delta.id
                    if tool_delta.function and tool_delta.function.name:
                        call["name"] += tool_delta.function.name
                    if tool_delta.function and tool_delta.function.arguments:
                        call["arguments"] += tool_delta.function.arguments
            if speech_buffer.strip():
                await speech_queue.put(speech_buffer.strip())
            await speech_queue.put(None)
            await speaker_task
        except BaseException:
            speaker_task.cancel()
            await asyncio.gather(speaker_task, return_exceptions=True)
            raise
        return "".join(content_parts).strip(), [calls[index] for index in sorted(calls)]

    async def _deliver_queued(self, speech_queue: asyncio.Queue[str | None]) -> None:
        while (sentence := await speech_queue.get()) is not None:
            await self._deliver(sentence)

    @staticmethod
    def _split_speech(text: str) -> tuple[list[str], str]:
        sentences: list[str] = []
        start = 0
        for match in re.finditer(r"[.!?](?:\s+|$)", text):
            end = match.end()
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        remainder = text[start:]
        if len(remainder) > 180 and " " in remainder:
            split_at = remainder.rfind(" ", 0, 160)
            sentences.append(remainder[:split_at].strip())
            remainder = remainder[split_at + 1 :]
        return sentences, remainder

    async def _deliver(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.audio is not None and self.tts is not None:
            self.state = AgentState.SPEAKING
            await self.audio.play(self.tts.stream(text))
        elif self.on_text is not None:
            self.on_text(text)
        else:
            print(text, flush=True)

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_knowledge_base":
            results = self.rag.search(arguments.get("query", ""), self.settings.rag_top_k)
            return {"results": results}
        if name == "search_customer":
            return await self.hubspot.search_contact(arguments.get("query", ""))
        return {"error": f"Unknown tool: {name}"}

    async def _execute_confirmed(self, action: PendingAction) -> dict[str, Any]:
        if action.name == "create_support_ticket":
            return await self.hubspot.create_ticket(
                title=action.arguments.get("title", "Customer support request"),
                description=action.arguments.get("description", ""),
                contact_email=action.arguments.get("contact_email", ""),
            )
        if action.name == "transfer_to_human":
            return await self.handoff.transfer(action.arguments.get("reason", ""))
        return {"error": "Unsupported confirmed action"}

    @staticmethod
    def _format_action_result(name: str, result: dict[str, Any]) -> str:
        if name == "create_support_ticket":
            if result.get("mode") == "mock":
                return f"In mock mode, I recorded ticket {result.get('id')}; it was not sent to HubSpot."
            return f"Your support ticket is created. The reference number is {result.get('id')}."
        if result.get("transferred"):
            return "The transfer is underway. Please stay on the line."
        return "The transfer was simulated in mock mode, so the call has not moved to an agent."

    def _remember(
        self, user_text: str, response: str, include_user: bool = True
    ) -> None:
        if include_user:
            self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": response})
        if len(self.messages) > 25:
            self.messages = [self.messages[0], *self.messages[-24:]]

    async def close(self) -> None:
        self.state = AgentState.STOPPED
        await self.hubspot.close()
        await self.client.close()
