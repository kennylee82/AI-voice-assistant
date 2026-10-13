from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ConfirmationState(Enum):
    NONE = auto()
    PENDING = auto()
    CONFIRMED = auto()
    DECLINED = auto()


@dataclass(frozen=True)
class PendingAction:
    name: str
    arguments: dict[str, Any]


class ConfirmationGate:
    """Holds one user-approved action until an explicit yes or no is heard."""

    def __init__(self) -> None:
        self.pending: PendingAction | None = None

    def request(self, name: str, arguments: dict[str, Any]) -> str:
        self.pending = PendingAction(name=name, arguments=arguments)
        if name == "create_support_ticket":
            return "I can create that support ticket. Would you like me to submit it?"
        return "I can transfer you to a human support agent. Should I start the transfer?"

    def resolve(self, text: str) -> tuple[ConfirmationState, PendingAction | None]:
        if self.pending is None:
            return ConfirmationState.NONE, None

        answer = re.sub(r"[^a-z']+", " ", text.lower()).strip()
        if answer in {
            "yes", "yes please", "yeah", "yeah please", "yep", "sure",
            "please do", "confirm", "do it",
        }:
            action, self.pending = self.pending, None
            return ConfirmationState.CONFIRMED, action
        if answer in {"no", "nope", "dont", "do not", "cancel", "stop"}:
            self.pending = None
            return ConfirmationState.DECLINED, None
        return ConfirmationState.PENDING, None
