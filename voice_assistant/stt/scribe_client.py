from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable

from elevenlabs import (
    AsyncElevenLabs,
    AudioFormat,
    CommitStrategy,
    RealtimeAudioOptions,
    RealtimeEvents,
)

from config import Settings


TranscriptHandler = Callable[[str], Awaitable[None]]


class ScribeClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)

    async def run(
        self,
        audio_queue: asyncio.Queue[bytes],
        on_partial: Callable[[str], None],
        on_committed: TranscriptHandler,
    ) -> None:
        audio_format = AudioFormat(f"pcm_{self.settings.audio_sample_rate}")
        connection = await self.client.speech_to_text.realtime.connect(
            RealtimeAudioOptions(
                model_id="scribe_v2_realtime",
                audio_format=audio_format,
                sample_rate=self.settings.audio_sample_rate,
                commit_strategy=CommitStrategy.VAD,
                vad_silence_threshold_secs=0.45,
                min_speech_duration_ms=120,
            )
        )

        stopped = asyncio.Event()
        errors: list[str] = []
        callback_tasks: set[asyncio.Task] = set()

        def event_text(event) -> str:
            if isinstance(event, dict):
                return str(event.get("text", ""))
            return str(getattr(event, "text", ""))

        def handle_partial(event) -> None:
            on_partial(event_text(event))

        def handle_committed(event) -> None:
            text = event_text(event).strip()
            if text:
                task = asyncio.create_task(on_committed(text))
                callback_tasks.add(task)
                task.add_done_callback(callback_tasks.discard)

        def handle_error(error) -> None:
            errors.append(str(error))
            stopped.set()

        connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, handle_partial)
        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, handle_committed)
        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS, handle_committed)
        connection.on(RealtimeEvents.ERROR, handle_error)
        connection.on(RealtimeEvents.AUTH_ERROR, handle_error)
        connection.on(RealtimeEvents.CLOSE, stopped.set)

        sender = asyncio.create_task(
            self._send_audio(connection, audio_queue), name="scribe-audio-sender"
        )
        stop_waiter = asyncio.create_task(stopped.wait(), name="scribe-stop-waiter")
        try:
            done, _ = await asyncio.wait(
                {sender, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
            if sender in done:
                sender.result()
            if errors:
                raise RuntimeError(errors[-1])
        finally:
            for task in (sender, stop_waiter):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, stop_waiter, return_exceptions=True)
            await connection.close()
            if callback_tasks:
                await asyncio.gather(*callback_tasks, return_exceptions=True)

    @staticmethod
    async def _send_audio(connection, audio_queue: asyncio.Queue[bytes]) -> None:
        while True:
            audio = await audio_queue.get()
            await connection.send(
                {"audio_base_64": base64.b64encode(audio).decode("ascii")}
            )
