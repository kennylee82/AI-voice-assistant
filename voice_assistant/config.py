from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_tts_model_id: str = os.getenv(
        "ELEVENLABS_TTS_MODEL_ID", "eleven_v4_turbo"
    )
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    hubspot_access_token: str = os.getenv("HUBSPOT_ACCESS_TOKEN", "")
    hubspot_mock: bool = os.getenv("HUBSPOT_MOCK", "true").lower() == "true"
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_call_sid: str = os.getenv("TWILIO_CALL_SID", "")
    twilio_from_number: str = os.getenv("TWILIO_FROM_NUMBER", "")
    handoff_number: str = os.getenv("HANDOFF_NUMBER", "")
    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_chunk_ms: int = int(os.getenv("AUDIO_CHUNK_MS", "20"))
    audio_input_device: str | None = os.getenv("AUDIO_INPUT_DEVICE") or None
    audio_output_device: str | None = os.getenv("AUDIO_OUTPUT_DEVICE") or None
    barge_in_rms_threshold: float = float(os.getenv("BARGE_IN_RMS_THRESHOLD", "0.035"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))

    @property
    def realtime_ready(self) -> bool:
        return bool(
            self.openai_api_key
            and self.elevenlabs_api_key
            and self.elevenlabs_voice_id
        )


settings = Settings()
