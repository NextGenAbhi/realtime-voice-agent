import os
import base64
import json
from urllib.parse import quote
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError, APITimeoutError
from elevenlabs.client import ElevenLabs

# Direct OpenAI client for TTS/audio endpoints
# openai_client_direct = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from app.schemas import (
    ChatRequest,
    ChatResponse,
    TTSChatRequest,
    DEFAULT_VOICE_ID,
    DEFAULT_TTS_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    TTS_RETRIES,
)
from app.utils.openai_utils import OpenAIClient
from app.utils.tts_utils import TTSUtils


load_dotenv()
# Constants
openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
tts_client = TTSUtils(api_key=os.getenv("ELEVENLABS_API_KEY"))

app = FastAPI()

app.mount("/realtime", StaticFiles(directory="app/static/realtime", html=True), name="realtime")

# @app.get("/favicon.ico", include_in_schema=False)
# async def favicon():
#     return FileResponse("app/static/favicon.ico")

@app.post("/v1/agent/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        """
        JSON response only. No audio. Non-blocking OpenAI call via run_in_threadpool.
        """
        reply = await openai_client.get_llm_reply(payload)
        if not reply:
            raise HTTPException(status_code=502, detail="Empty reply from model.")
        return ChatResponse(reply=reply, model=payload.model)
    except (OpenAIError, APITimeoutError) as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/agent/chat/stream-audio")
async def chat_stream_audio(payload: TTSChatRequest):
    """
    Streams ElevenLabs audio chunks directly to the browser.
    Content-Type reflects the requested format (mp3/wav/pcm).
    """
    if tts_client.xi is None:
        # Keep JSON endpoint usable even if TTS isn’t configured.
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured; audio streaming unavailable.")

    try:
        # 1) Get model reply (guard empty)
        reply = await openai_client.get_llm_reply(payload)
        if not reply:
            raise HTTPException(status_code=502, detail="Empty reply from model.")

        # 2) Trim very long text for TTS (resilience)
        tts_text, truncated = tts_client.trim_for_tts(reply)

        # 3) Build a generator that yields TTS bytes, with retries on initial connect
        def gen():
            yield from tts_client.iter_tts_with_retries(
                text=tts_text,
                voice_id=payload.voice_id or DEFAULT_VOICE_ID,
                tts_model_id=payload.tts_model_id or DEFAULT_TTS_MODEL,
                output_format=payload.output_format or DEFAULT_OUTPUT_FORMAT,
                retries=TTS_RETRIES,
            )

        headers = {
            # Expose a short preview; FE can show it while audio plays.
            "x-agent-text": quote(tts_text[:1024]),
            "x-text-truncated": "true" if truncated else "false",
        }

        media_type = tts_client.media_type_for(payload.output_format or DEFAULT_OUTPUT_FORMAT)
        return StreamingResponse(gen(), media_type=media_type, headers=headers)

    except (OpenAIError, APITimeoutError) as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")
    except Exception as e:
        # surfaces ElevenLabs errors incl. retry exhaustion
        print("Error in /v1/agent/chat/stream-audio:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if 'audio' in message:
                audio_base64 = message['audio']
                audio_bytes = base64.b64decode(audio_base64)
                
                # Transcribe audio to text
                user_text = await openai_client.transcribe_audio(audio_bytes)

                if user_text and user_text.strip():
                    # Get assistant's response
                    chat_request = ChatRequest(messages=[{"role": "user", "content": user_text}])
                    assistant_text = await openai_client.get_llm_reply(chat_request)
                    
                    # Generate audio for assistant's response
                    response = openai_client.client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=assistant_text,
                        response_format="mp3"
                    )
                    assistant_audio_bytes = response.read()
                    assistant_audio_base64 = base64.b64encode(assistant_audio_bytes).decode('utf-8')

                    # Send user's text, assistant's text and audio to UI
                    await websocket.send_text(json.dumps({
                        "user_text": user_text,
                        "assistant_text": assistant_text,
                        "assistant_audio": assistant_audio_base64
                    }))
                else:
                    # Optionally, send a message back to the UI indicating silence was detected
                    await websocket.send_text(json.dumps({
                        "status": "silence_detected"
                    }))
                # Send a final message to indicate the end of the audio stream
                # await websocket.send_text(json.dumps({"audio_stream_end": True}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        error_message = f"Error in websocket: {str(e)}"
        print(error_message)
        # Truncate the reason to avoid ProtocolError
        reason = (error_message[:120] + '...') if len(error_message) > 123 else error_message
        await websocket.close(code=1011, reason=reason)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
        await websocket.close(code=1011, reason=str(e))