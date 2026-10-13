from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    vector: dict[int, float]


DEFAULT_DOCUMENTS = {
    "account_access": "If you cannot sign in, use the password reset link on the sign-in page. The reset link expires after 30 minutes. If you no longer have access to your email, contact support so an agent can verify the account.",
    "billing": "Subscriptions renew on the date shown in Account Settings. To cancel, open Account Settings, choose Plan, then select Cancel subscription. Access remains active through the paid billing period. Refunds are reviewed by the billing team; do not promise a refund.",
    "support_hours": "Customer support is available Monday through Friday, 9 AM to 5 PM Pacific Time, excluding public holidays. For urgent issues outside these hours, submit a support ticket and the on-call team will review it.",
    "data_export": "To export your workspace data, open Settings, choose Data, and select Request export. An email notification is sent when the archive is ready. Export links expire after 7 days.",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _vectorize(text: str) -> dict[int, float]:
    counts: dict[int, float] = {}
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
        index = int.from_bytes(digest, "little") % 512
        counts[index] = counts.get(index, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm:
        counts = {index: value / norm for index, value in counts.items()}
    return counts


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


class RagEngine:
    """A small deterministic hashed-vector index for speech-sized passages."""

    def __init__(self, documents: dict[str, str] | None = None, max_tokens: int = 120):
        self.max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding("o200k_base")
        self.chunks = [
            Chunk(source=source, text=part, vector=_vectorize(part))
            for source, text in (documents or DEFAULT_DOCUMENTS).items()
            for part in self._split(text)
        ]

    def _split(self, text: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if self.token_count(candidate) <= self.max_tokens:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if self.token_count(word) <= self.max_tokens:
                current = word
                continue
            fragment = ""
            for character in word:
                if self.token_count(fragment + character) > self.max_tokens:
                    chunks.append(fragment)
                    fragment = character
                else:
                    fragment += character
            current = fragment
        if current:
            chunks.append(current)
        return chunks

    def token_count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str | float]]:
        query_vector = _vectorize(query)
        ranked = sorted(
            (
                (_cosine(query_vector, chunk.vector), chunk)
                for chunk in self.chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            {"source": chunk.source, "text": chunk.text, "score": score}
            for score, chunk in ranked[: max(0, top_k)]
            if score > 0
        ]
