# Voice Assistant

An asynchronous, event-driven customer-support voice assistant. It streams microphone audio to ElevenLabs Scribe v2 Realtime through the official Python SDK, sends committed turns through GPT-4o with streamed function calling, and streams Eleven v4 Turbo 24 kHz PCM speech to the speaker over the Text-to-Dialogue WebSocket. HubSpot and Twilio operations are guarded by explicit verbal confirmation; integrations default to mock behavior where credentials are absent.

## Requirements

- Python 3.9 or later
- Microphone and speaker for live voice mode
- OpenAI and ElevenLabs API credentials for live voice mode
- HubSpot private-app token for live CRM operations (optional)
- An active Twilio call SID, caller ID, and agent number for live conference handoff (optional)

Scribe uses `scribe_v2_realtime` with SDK-managed WebSocket events and VAD commits. TTS defaults to `eleven_v4_turbo`, which uses the Text-to-Dialogue WebSocket; it is not the standard HTTP `/text-to-speech/{voice_id}/stream` endpoint. Set `ELEVENLABS_TTS_MODEL_ID` to another compatible model only when using the matching transport. The configured voice must be available to your ElevenLabs account.

## Setup

```sh
cd voice_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, and `ELEVENLABS_VOICE_ID` in `.env` for live voice mode. Never commit `.env` or place credentials in source code.

## Run

```sh
python main.py                 # Live microphone and speaker mode
python main.py --text          # OpenAI agent using terminal text input
python main.py --demo          # Local sample RAG, no API keys or audio
python main.py --list-audio-devices
```

On macOS, grant microphone access to the terminal or VS Code host running Python. Set `AUDIO_INPUT_DEVICE` and `AUDIO_OUTPUT_DEVICE` to numeric IDs when the defaults are not right; inspect devices with the command above. Capture uses mono signed 16-bit PCM at 16 kHz. Playback requests mono signed 16-bit PCM at 24 kHz. `BARGE_IN_RMS_THRESHOLD` tunes local interruption detection; speaker echo may require headphones or a higher threshold.

## Integrations and safety

- The in-memory hashed-vector retriever has no model download and splits source text at 120 GPT-4o tokens. Replace the sample support notes in `tools/rag_engine.py` with approved company content.
- `HUBSPOT_MOCK=true` is the default. Set it to `false` and configure `HUBSPOT_ACCESS_TOKEN` to create/search real CRM records. Ticket creation requires a clear yes after the assistant reads its confirmation prompt.
- Without complete Twilio settings, handoff is reported as simulated and no call is placed. Live conference transfer requires `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_CALL_SID`, `TWILIO_FROM_NUMBER`, and `HANDOFF_NUMBER`; it moves the active caller leg into a conference with the agent leg.
- Audio and transcripts are sent to the configured providers in live mode. Review their retention and privacy terms before handling customer data. Do not use the sample support content as production policy.

## Checks

```sh
python -m unittest discover -s tests -v
python -m compileall -q .
```
