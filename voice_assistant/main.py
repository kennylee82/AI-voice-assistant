from __future__ import annotations

import argparse
import asyncio
import sys

from config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time voice support assistant")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Run the local mock demo")
    mode.add_argument("--text", action="store_true", help="Use text input, without microphone audio")
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print available local audio devices and exit",
    )
    return parser.parse_args()


async def run_demo() -> None:
    from tools.rag_engine import RagEngine

    rag = RagEngine()
    print("Local demo mode. Ask about sign-in, billing, support hours, or data export.")
    print("Type 'quit' to exit. No API requests or audio devices are used.\n")
    while True:
        try:
            question = await asyncio.to_thread(input, "You: ")
        except EOFError:
            return
        if question.strip().lower() in {"quit", "exit"}:
            return
        matches = rag.search(question, top_k=1)
        if matches:
            print(f"Assistant: {matches[0]['text']}\n")
        else:
            print("Assistant: I couldn't find that in the local sample support notes.\n")


async def run_text_mode() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env before using --text mode.")

    from agent.orchestrator import AgentState, VoiceOrchestrator

    assistant = VoiceOrchestrator(settings, on_text=lambda text: print(f"Assistant: {text}"))
    print("Text mode. Type 'quit' to exit. HubSpot stays mocked by default.\n")
    try:
        while True:
            try:
                text = await asyncio.to_thread(input, "You: ")
            except EOFError:
                return
            if text.strip().lower() in {"quit", "exit"}:
                return
            if text.strip():
                await assistant.turn(text)
    finally:
        await assistant.close()


async def run_voice_mode() -> None:
    if not settings.realtime_ready:
        raise RuntimeError(
            "Live voice mode needs OPENAI_API_KEY, ELEVENLABS_API_KEY, and "
            "ELEVENLABS_VOICE_ID in .env. Use --demo to try the local sample."
        )

    from agent.orchestrator import VoiceOrchestrator
    from audio.stream_manager import AudioStreamManager
    from stt.scribe_client import ScribeClient
    from tts.eleven_v4_client import ElevenLabsTTSClient

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=48)
    audio = AudioStreamManager(settings, audio_queue, loop)
    tts = ElevenLabsTTSClient(settings)
    assistant = VoiceOrchestrator(settings, audio=audio, tts=tts)
    active_turn: asyncio.Task | None = None

    def show_partial(text: str) -> None:
        print(f"\rYou: {text}", end="", flush=True)

    async def on_committed(text: str) -> None:
        nonlocal active_turn
        print(f"\rYou: {text}                           ")
        if active_turn is not None and not active_turn.done():
            active_turn.cancel()
            audio.cancel_playback()
            await asyncio.gather(active_turn, return_exceptions=True)
        active_turn = asyncio.create_task(run_turn(text), name="assistant-turn")

    async def run_turn(text: str) -> None:
        try:
            await assistant.turn(text)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            assistant.state = AgentState.LISTENING
            print(f"\nAssistant turn failed: {error}", file=sys.stderr, flush=True)

    async def interrupt_watcher() -> None:
        while True:
            await audio.barge_in_event.wait()
            if active_turn is not None and not active_turn.done():
                active_turn.cancel()
                audio.cancel_playback()
                await asyncio.gather(active_turn, return_exceptions=True)
            audio.barge_in_event.clear()

    try:
        audio.start()
        print("Listening. Speak naturally; interrupt playback by speaking. Press Ctrl+C to stop.\n")
        tasks = {
            asyncio.create_task(
                ScribeClient(settings).run(audio_queue, show_partial, on_committed),
                name="scribe-realtime",
            ),
            asyncio.create_task(interrupt_watcher(), name="barge-in-watcher"),
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if active_turn is not None and not active_turn.done():
            active_turn.cancel()
            await asyncio.gather(active_turn, return_exceptions=True)
        audio.close()
        await assistant.close()


def main() -> int:
    args = parse_args()
    if args.list_audio_devices:
        try:
            import sounddevice as sd
        except ImportError as error:
            print(f"Install requirements first: {error}", file=sys.stderr)
            return 1
        print(sd.query_devices())
        return 0

    try:
        if args.demo:
            asyncio.run(run_demo())
        elif args.text:
            asyncio.run(run_text_mode())
        else:
            asyncio.run(run_voice_mode())
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
    except Exception as error:
        print(f"Assistant could not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
