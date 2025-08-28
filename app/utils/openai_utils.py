import io
import os
# from app.main import OPENAI_API_KEY
from app.schemas import ChatRequest, OPENAI_TOKEN_LIMIT
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

# OpenAI client with timeouts/retries
class OpenAIClient:
    def __init__(self, api_key: str = None, timeout: int = 30, max_retries: int = 2):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries
        )

    async def get_llm_reply(self, payload: ChatRequest) -> str:
        """Call OpenAI in a threadpool to avoid blocking the event loop."""
        def _call():
            resp = self.client.chat.completions.create(
                model=payload.model,
                messages=[m.model_dump() for m in payload.messages],
                temperature=getattr(payload, "temperature", 0.2),
                max_tokens=OPENAI_TOKEN_LIMIT
            )
            return (resp.choices[0].message.content or "").strip()
        return await run_in_threadpool(_call)

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio using OpenAI Whisper."""
        def _call():
            # Note: The client library expects a file-like object.
            # We use io.BytesIO to wrap our in-memory bytes.
            # The 'file' argument is a tuple of (filename, file-like-object, mimetype).
            audio_file = ("audio.webm", io.BytesIO(audio_bytes), "audio/webm")
            resp = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            return (resp.text or "").strip()
        return await run_in_threadpool(_call)