The Python voice assistant project is in `voice_assistant/`.

- Follow the architecture and setup notes in `voice_assistant/README.md`.
- Keep provider credentials in `voice_assistant/.env`; never commit secrets.
- Keep HubSpot and Twilio side effects behind the verbal confirmation gate.
- Run checks from `voice_assistant/` with `python -m unittest discover -s tests -v` and `python -m compileall -q .`.
- Preserve async I/O and bounded buffering in audio, STT, LLM, and TTS paths.
