from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets

from config import Settings


class ElevenLabsTTSClient:
    """Streams 24 kHz PCM with Eleven v4 Turbo over Text-to-Dialogue WebSocket."""

    def __init__(self, settings: Settings):
        if not settings.elevenlabs_voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is required for speech output")
        self.settings = settings

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        query = urlencode(
            {
                "model_id": self.settings.elevenlabs_tts_model_id,
                "output_format": "pcm_24000",
            }
        )
        url = f"wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?{query}"
        voice_id = self.settings.elevenlabs_voice_id

        async with websockets.connect(url, max_size=2**20, ping_interval=20) as socket:
            await socket.send(
                json.dumps(
                    {
                        "voices": [voice_id],
                        "xi_api_key": self.settings.elevenlabs_api_key,
                    }
                )
            )
            await socket.send(
                json.dumps(
                    {
                        "inputs": [
                            {"text": text, "voice_id": voice_id, "new_turn": False}
                        ]
                    }
                )
            )
            await socket.send(json.dumps({"close_socket": True}))

            async for raw_message in socket:
                message = json.loads(raw_message)
                if message.get("error"):
                    raise RuntimeError(str(message["error"]))
                audio = message.get("audio")
                if audio:
                    yield base64.b64decode(audio)
                if message.get("is_final"):
                    break

