import logging
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket
from faster_whisper import WhisperModel
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper_server")

# Initialize the model
model_size = os.getenv("WHISPER_MODEL", "distil-medium.en")
device = "cpu"  # Change to "cuda" if you want to use GPU
model = WhisperModel(model_size, device=device, compute_type="int8")

logger.info(f"Loaded Faster Whisper model: {model_size} on {device}")

async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            logger.info("Received audio data")

            # Save the received audio data to a file
            audio_file_path = "received_audio.wav"
            with open(audio_file_path, "wb") as f:
                f.write(data)
            logger.info(f"Audio data saved to {audio_file_path}")

            # Transcribe the audio file
            segments, info = model.transcribe(audio_file_path, beam_size=8, language="en", vad_filter=True)
            transcription = " ".join(segment.text for segment in segments)
            logger.info(f"Transcription: {transcription}")

            # Send the transcription back to the client
            await websocket.send_text(transcription)
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")
    finally:
        if not websocket.client_state.closed:
            await websocket.close()
        logger.info("WebSocket connection closed")

app = Starlette()

# Add the WebSocket route
app.add_websocket_route("/ws", transcribe_audio)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
