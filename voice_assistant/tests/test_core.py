import unittest
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

from guards.confirmation import ConfirmationGate, ConfirmationState
from elevenlabs import RealtimeEvents
from stt.scribe_client import ScribeClient
from tools.rag_engine import RagEngine
from tts.eleven_v4_client import ElevenLabsTTSClient


class ConfirmationGateTests(unittest.TestCase):
    def test_action_requires_explicit_yes(self):
        gate = ConfirmationGate()
        gate.request("create_support_ticket", {"title": "Login issue"})

        state, action = gate.resolve("maybe later")
        self.assertEqual(state, ConfirmationState.PENDING)
        self.assertIsNone(action)

        state, action = gate.resolve("yes, please")
        self.assertEqual(state, ConfirmationState.CONFIRMED)
        self.assertIsNotNone(action)
        self.assertEqual(action.arguments["title"], "Login issue")

    def test_no_discards_pending_action(self):
        gate = ConfirmationGate()
        gate.request("transfer_to_human", {"reason": "urgent issue"})

        state, action = gate.resolve("no")
        self.assertEqual(state, ConfirmationState.DECLINED)
        self.assertIsNone(action)
        self.assertEqual(gate.resolve("yes")[0], ConfirmationState.NONE)


class RagEngineTests(unittest.TestCase):
    def test_chunks_stay_under_token_limit_and_retrieve_relevant_note(self):
        rag = RagEngine({"billing": " ".join(["billing"] * 260)}, max_tokens=120)

        self.assertTrue(all(len(chunk.text.split()) <= 120 for chunk in rag.chunks))
        self.assertTrue(all(rag.token_count(chunk.text) <= 120 for chunk in rag.chunks))
        self.assertEqual(rag.search("billing", top_k=1)[0]["source"], "billing")

    def test_empty_query_returns_no_results(self):
        self.assertEqual(RagEngine().search("", top_k=3), [])


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def __aiter__(self):
        for message in self.messages:
            yield json.dumps(message)


class ScribeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_sdk_vad_events_and_sends_audio(self):
        class FakeConnection:
            def __init__(self):
                self.handlers = {}
                self.audio_sent = asyncio.Event()
                self.payloads = []

            def on(self, event, callback):
                self.handlers[event] = callback

            async def send(self, payload):
                self.payloads.append(payload)
                self.audio_sent.set()

            async def close(self):
                self.handlers[RealtimeEvents.CLOSE]()

        import asyncio

        connection = FakeConnection()
        connect = AsyncMock(return_value=connection)
        fake_sdk = SimpleNamespace(
            speech_to_text=SimpleNamespace(
                realtime=SimpleNamespace(connect=connect)
            )
        )
        settings = SimpleNamespace(
            elevenlabs_api_key="test-key", audio_sample_rate=16000
        )
        audio_queue = asyncio.Queue()
        audio_queue.put_nowait(b"pcm audio")
        partials = []
        committed = []

        async def handle_committed(text):
            committed.append(text)

        with patch("stt.scribe_client.AsyncElevenLabs", return_value=fake_sdk):
            client = ScribeClient(settings)
            run_task = asyncio.create_task(
                client.run(audio_queue, partials.append, handle_committed)
            )
            await connection.audio_sent.wait()
            connection.handlers[RealtimeEvents.PARTIAL_TRANSCRIPT]({"text": "Hel"})
            connection.handlers[RealtimeEvents.COMMITTED_TRANSCRIPT]({"text": "Hello"})
            connection.handlers[RealtimeEvents.CLOSE]()
            await run_task

        options = connect.await_args.args[0]
        self.assertEqual(options["model_id"], "scribe_v2_realtime")
        self.assertEqual(options["commit_strategy"].value, "vad")
        self.assertEqual(partials, ["Hel"])
        self.assertEqual(committed, ["Hello"])
        self.assertEqual(
            connection.payloads,
            [{"audio_base_64": base64.b64encode(b"pcm audio").decode("ascii")}],
        )


class ElevenLabsTTSClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_v4_turbo_pcm_from_dialogue_websocket(self):
        pcm_frame = b"\x01\x00\x02\x00"
        socket = FakeWebSocket(
            [
                {"audio": base64.b64encode(pcm_frame).decode("ascii")},
                {"is_final": True},
            ]
        )
        settings = SimpleNamespace(
            elevenlabs_voice_id="voice-test",
            elevenlabs_api_key="key-test",
            elevenlabs_tts_model_id="eleven_v4_turbo",
        )
        client = ElevenLabsTTSClient(settings)

        with patch("tts.eleven_v4_client.websockets.connect", return_value=socket) as connect:
            chunks = [chunk async for chunk in client.stream("Hello there.")]

        query = parse_qs(urlsplit(connect.call_args.args[0]).query)
        self.assertEqual(query["model_id"], ["eleven_v4_turbo"])
        self.assertEqual(query["output_format"], ["pcm_24000"])
        self.assertEqual(chunks, [pcm_frame])
        self.assertEqual(socket.sent[0]["voices"], ["voice-test"])
        self.assertEqual(socket.sent[0]["xi_api_key"], "key-test")
        self.assertEqual(socket.sent[1]["inputs"][0]["text"], "Hello there.")
        self.assertEqual(socket.sent[2], {"close_socket": True})


if __name__ == "__main__":
    unittest.main()
