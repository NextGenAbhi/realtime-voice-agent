from pydantic import BaseModel, Field, constr
from typing import List, Optional, Literal, Annotated

Role = Literal["user", "assistant", "system"]
# --- TTS CONFIG & RESILIENCE ---
MAX_TTS_CHARS = 500                 # guard against very long TTS texts
TTS_RETRIES = 2                      # number of retry attempts for TTS
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
OPENAI_TOKEN_LIMIT = 200                   # model context length limit

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    # session_id: Annotated[str, constr(strip_whitespace=True, min_length=1)] = Field(
    #     ..., description="client chosen conversation id"
    # )
    messages: List[Message] = Field(
        ..., description="list of messages comprising the conversation so far"
    )
    model: str = "gpt-4o-mini"
    max_context: int = 12

class ChatResponse(BaseModel):
    reply: str
    model: str

class TTSChatRequest(ChatRequest):  # inherits your fields
    voice_id: Optional[str] = DEFAULT_VOICE_ID
    tts_model_id: Optional[str] = DEFAULT_TTS_MODEL
    output_format: Optional[
        Literal["mp3_44100_128", "mp3_22050_32", "wav_44100", "pcm_16000"]
    ] = DEFAULT_OUTPUT_FORMAT
