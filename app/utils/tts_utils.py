import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from app.schemas import MAX_TTS_CHARS, TTS_RETRIES
from typing import Generator

load_dotenv()
class TTSUtils:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.xi = ElevenLabs(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def media_type_for(fmt: str) -> str:
        if fmt.startswith("mp3"):
            return "audio/mpeg"
        if fmt.startswith("wav"):
            return "audio/wav"
        # PCM is raw; some browsers won’t play it directly.
        return "audio/L16"

    @staticmethod
    def trim_for_tts(text: str) -> tuple[str, bool]:
        if len(text) <= MAX_TTS_CHARS:
            return text, False
        return text[:MAX_TTS_CHARS], True

    def tts_stream_once(
        self,
        text: str,
        voice_id: str,
        tts_model_id: str,
        output_format: str,
    ):
        """Single attempt to open a TTS stream from ElevenLabs."""
        if not self.xi:
            raise ValueError("ELEVENLABS_API_KEY not set or invalid.")
        return self.xi.text_to_speech.convert(
            voice_id=voice_id,
            model_id=tts_model_id,
            text=text,
            output_format=output_format,
        )
    def iter_tts_with_retries(
        self,
        text: str,
        voice_id: str,
        tts_model_id: str,
        output_format: str,
        retries: int = TTS_RETRIES,
    ) -> Generator[bytes, None, None]:
        """
        Yields audio chunks from ElevenLabs, retrying the *initial* stream creation
        a few times for resilience. (Mid-stream failures are surfaced.)
        """
        attempt = 0
        while attempt <= retries:
            try:
                stream = self.tts_stream_once(text, voice_id, tts_model_id, output_format)
                for chunk in stream:
                    if isinstance(chunk, bytes):
                        yield chunk
                return  # completed successfully
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    raise e