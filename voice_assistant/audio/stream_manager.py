from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator

import numpy as np

from config import Settings


class AudioStreamManager:
    """Bridges sounddevice's real-time callbacks to bounded asyncio queues."""

    def __init__(
        self, settings: Settings, audio_queue: asyncio.Queue[bytes], loop: asyncio.AbstractEventLoop
    ):
        self.settings = settings
        self.audio_queue = audio_queue
        self.loop = loop
        self.barge_in_event = asyncio.Event()
        self._speaker_queue: queue.Queue[bytes] = queue.Queue(maxsize=24)
        self._pending_playback = b""
        self._speaking = False
        self._input_stream = None
        self._output_stream = None

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @staticmethod
    def _device_id(value: str | None) -> int | None:
        if value and value.isdigit():
            return int(value)
        return None

    def start(self) -> None:
        import sounddevice as sd

        blocksize = max(1, self.settings.audio_sample_rate * self.settings.audio_chunk_ms // 1000)
        self._output_stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
            device=self._device_id(self.settings.audio_output_device),
            callback=self._speaker_callback,
            blocksize=1024,
        )
        self._input_stream = sd.InputStream(
            samplerate=self.settings.audio_sample_rate,
            channels=1,
            dtype="int16",
            device=self._device_id(self.settings.audio_input_device),
            blocksize=blocksize,
            callback=self._microphone_callback,
        )
        self._output_stream.start()
        self._input_stream.start()

    def _microphone_callback(self, indata, frames, timing, status) -> None:
        if status:
            self.loop.call_soon_threadsafe(print, f"Audio input: {status}")
        audio = indata.copy().tobytes()
        if self._speaking:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0) if samples.size else 0.0
            if rms >= self.settings.barge_in_rms_threshold:
                self.loop.call_soon_threadsafe(self.barge_in_event.set)
        self.loop.call_soon_threadsafe(self._enqueue_microphone_audio, audio)

    def _enqueue_microphone_audio(self, audio: bytes) -> None:
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            pass

    def _speaker_callback(self, outdata, frames, timing, status) -> None:
        output = outdata[:, 0]
        output.fill(0)
        written = 0
        while written < frames:
            if not self._pending_playback:
                try:
                    self._pending_playback = self._speaker_queue.get_nowait()
                except queue.Empty:
                    break
            samples = np.frombuffer(self._pending_playback, dtype=np.int16)
            count = min(frames - written, samples.size)
            output[written : written + count] = samples[:count]
            written += count
            self._pending_playback = samples[count:].tobytes()

    async def play(self, chunks: AsyncIterator[bytes]) -> None:
        self._speaking = True
        self.barge_in_event.clear()
        pcm_remainder = b""
        try:
            async for chunk in chunks:
                if self.barge_in_event.is_set():
                    break
                chunk = pcm_remainder + chunk
                aligned_length = len(chunk) - (len(chunk) % 2)
                pcm_remainder = chunk[aligned_length:]
                chunk = chunk[:aligned_length]
                while chunk and not self.barge_in_event.is_set():
                    try:
                        self._speaker_queue.put(chunk, timeout=0.05)
                        break
                    except queue.Full:
                        await asyncio.sleep(0)
            while (
                (not self._speaker_queue.empty() or self._pending_playback)
                and not self.barge_in_event.is_set()
            ):
                await asyncio.sleep(0.01)
            if not self.barge_in_event.is_set():
                await asyncio.sleep(0.04)
        finally:
            self._speaking = False

    def cancel_playback(self) -> None:
        while True:
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                break
        self._pending_playback = b""
        self._speaking = False

    def close(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                stream.stop()
                stream.close()
