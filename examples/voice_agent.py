#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import sys
import time

# Add parent directory to path for local development (must be before other imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

# Load environment variables from examples/.env.local (must be before plugin imports)
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env.local"))

# LiveKit imports - plugins must be imported on main thread
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import silero
from livekit.plugins import openai as lk_openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Local plugins
from local_livekit_plugins import FasterWhisperSTT, PiperTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("voice-agent")

# =============================================================================
# Configuration
# =============================================================================

USE_LOCAL = os.getenv("USE_LOCAL", "false").lower() == "true"

# Local pipeline settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")



# =============================================================================
# Agent Definition
# =============================================================================

class VoiceAssistant(Agent):
    """A simple voice assistant that responds to user queries."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            Keep your responses concise and conversational - aim for 1-2 sentences.
            Be friendly and natural in your speech patterns."""
        )


# =============================================================================
# Pipeline Factories
# =============================================================================

def create_local_session() -> AgentSession:

    logger.info("=" * 60)
    logger.info("STARTING LOCAL PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  STT: FasterWhisper ({WHISPER_MODEL} on {WHISPER_DEVICE})")
    logger.info(f"  LLM: Ollama ({OLLAMA_MODEL})")
    logger.info(f"  TTS: Piper (CUDA: {PIPER_USE_CUDA})")
    logger.info("=" * 60)

    turn_detector = MultilingualModel()

    if not PIPER_MODEL_PATH:
        raise ValueError(
            "PIPER_MODEL_PATH not set. Download a voice model from:\n"
            "https://huggingface.co/rhasspy/piper-voices"
        )

    return AgentSession(
        stt=FasterWhisperSTT(
            model_size=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="int8"  #float16
                if WHISPER_DEVICE == "cpu" 
                else "int8",
        ),
        llm=lk_openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
        ),
        tts=PiperTTS(
            model_path=PIPER_MODEL_PATH,
            use_cuda=PIPER_USE_CUDA,
        ),
        vad=silero.VAD.load(),
        turn_detection=turn_detector,
    )

''' if cloud pipline is needed'''
'''def create_cloud_session() -> AgentSession:
    """
    Create an AgentSession using cloud STT/LLM/TTS services.

    Requirements:
        - DEEPGRAM_API_KEY
        - OPENAI_API_KEY
        - CARTESIA_API_KEY
    """
    logger.info("=" * 60)
    logger.info("STARTING CLOUD PIPELINE")
    logger.info("=" * 60)
    logger.info("  STT: Deepgram Nova-2")
    logger.info("  LLM: OpenAI GPT-4o-mini")
    logger.info("  TTS: Cartesia Sonic")
    logger.info("=" * 60)

    return AgentSession(
        stt="deepgram/nova-2",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic",
        vad=silero.VAD.load(),
    )'''


# =============================================================================
# Agent Entrypoint
# =============================================================================

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for the voice agent."""

    logger.info(f"Joining room: {ctx.room.name}")
    await ctx.connect()   #connect agent to LiveKit server and room

    # Create session based on configuration
    session = create_local_session() if USE_LOCAL else create_cloud_session()

    # ==========================================================================
    # Round-trip latency tracking
    # ==========================================================================
    # Measures time from user speech transcribed to agent starting to speak.
    # This captures: LLM processing + TTS first byte (STT already done).

    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

    # ==========================================================================

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        room_input_options=RoomInputOptions(),
    )

    # Send initial greeting
    await session.generate_reply(
        instructions="Greet the user and let them know you're ready to help."
    )

    logger.info("Agent ready - listening for speech...")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
