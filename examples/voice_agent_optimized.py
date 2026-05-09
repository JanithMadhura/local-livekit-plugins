#!/usr/bin/env python3
"""
Optimized Low-Latency Voice Agent
==================================

Real-time voice AI with minimal latency:
- Streaming LLM responses (start TTS before full response)
- Streaming TTS (start audio playback from first sentence)
- Pipelined processing (concurrent synthesis + playback)
- Expected latency: 1.5-3 seconds (vs 6-7 seconds in standard mode)

Key Optimizations:
1. LLM streaming: Begin TTS synthesis as tokens arrive (not waiting for full response)
2. TTS streaming: Begin audio playback from first sentence (not waiting for full synthesis)
3. Aggressive speed: 1.5x speech rate reduces both perceived and actual latency
4. Optimized models: Smaller/faster models where possible

Usage:
    uv run examples/voice_agent_optimized.py console
"""

from __future__ import annotations

import logging
import os
import sys
import time
import asyncio

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from livekit.agents.llm import ChatContext

# Load environment variables
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env.local"))

# LiveKit imports - plugins must be at module level
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import silero
from livekit.plugins import openai as lk_openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Local plugins - use streaming TTS
from local_livekit_plugins import FasterWhisperSTT
from local_livekit_plugins.piper_tts_streaming import PiperTTSStreaming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("voice-agent-optimized")


# =============================================================================
# Configuration - Optimized for Latency
# =============================================================================

USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"

# STT: Use smaller/faster Whisper model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # "base" is faster than "medium"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # float16 for GPU, int8 for CPU

# TTS: Streaming with 1.5x speed for lower latency
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_SPEED = float(os.getenv("PIPER_SPEED", "1.0"))  # 50% faster speech
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"

# LLM: Use faster model if available
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")  # phi3, neural-chat, tinyllama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


# =============================================================================
# Agent Definition - Optimized Prompts
# =============================================================================

class OptimizedVoiceAssistant(Agent):
    """
    Voice assistant optimized for real-time conversation.
    
    Instructions designed to:
    - Produce shorter responses (faster to synthesize)
    - Be naturally conversational (works well at 1.5x speed)
    - Use simple sentence structure (clearer at accelerated speed)
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a realtime voice assistant.

                STRICT RULES:
                - Maximum 5 words
                - One sentence only
                - No explanations
                - No repetition
                - No extra details
                - No safety messages
                - Stop immediately after answering
                """

        )


# =============================================================================
# Pipeline Factory - Optimized Configuration
# =============================================================================

def create_optimized_local_session() -> AgentSession:
    """Create a low-latency local processing pipeline."""

    logger.info("=" * 70)
    logger.info("OPTIMIZED LOW-LATENCY PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  STT: FasterWhisper ({WHISPER_MODEL} on {WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})")
    logger.info(f"  LLM: Ollama ({OLLAMA_MODEL}) - streaming enabled")
    logger.info(f"  TTS: PiperTTSStreaming (speed={PIPER_SPEED}x, streaming=True)")
    logger.info("=" * 70)
    logger.info("Expected latency: 1.5-3 seconds (vs 6-7 seconds standard)")
    logger.info("=" * 70)

    turn_detector = MultilingualModel()
    stt_vad = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.2,
        prefix_padding_duration=0.1,
        max_buffered_speech=1.5,
    )

    if not PIPER_MODEL_PATH:
        raise ValueError(
            "PIPER_MODEL_PATH not set. Download a voice model from:\n"
            "https://huggingface.co/rhasspy/piper-voices"
        )

    return AgentSession(
        # STT: Streaming-capable whisper with VAD-based chunking
        stt=FasterWhisperSTT(
            model_size=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,  # Quantized for speed
            streaming=True,
            vad=stt_vad,
        ),
        # LLM: Streaming is automatic with LiveKit agents
        llm=lk_openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        ), 
        # TTS: Uses internal sentence-level streaming for low-latency synthesis
        # (not LiveKit's streaming interface, but achieves same effect)
        tts=PiperTTSStreaming(
            model_path=PIPER_MODEL_PATH,
            use_cuda=PIPER_USE_CUDA,
            speed=PIPER_SPEED,  # 1.5 = 50% faster
            streaming=False,    # Uses internal sentence-level streaming (compatible)
        ),
        vad=stt_vad,
        turn_detection=turn_detector,
    )


# =============================================================================
# Agent Entrypoint
# =============================================================================

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for optimized voice agent."""

    logger.info(f"Joining room: {ctx.room.name}")
    await ctx.connect()

    # Create optimized session
    session = create_optimized_local_session()

    # =========================================================================
    # Latency Tracking - Now Shows Streaming Benefit
    # =========================================================================

    _transcription_time: float | None = None
    _chunk_count = 0
    _first_audio_time: float | None = None

    # Transcript stabilization state
    _partial_transcript = ""
    _last_speech_time = 0.0
    _user_speaking = False
    _eou_probability = 0.0
    _agent_speaking = False
    _current_speech_handle = None
    _interrupt_requested = False

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        nonlocal _chunk_count
        nonlocal _first_audio_time
        nonlocal _partial_transcript

        _transcription_time = time.perf_counter()
        _chunk_count = 0
        _first_audio_time = None

        text = ev.transcript.strip()

        if text:
            # Merge partial transcript progressively
            _partial_transcript = text

        logger.info(f"PARTIAL: {_partial_transcript}")

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        nonlocal _user_speaking
        nonlocal _agent_speaking
        nonlocal _partial_transcript
        nonlocal _current_speech_handle
        nonlocal _interrupt_requested
        logger.info(f"Agent speaking state: {_agent_speaking}")

        _user_speaking = True

        # User interrupted assistant speech
        if _agent_speaking:
            
            logger.info("User interruption detected")

            _interrupt_requested = True

            # Clear previous transcript state
            _partial_transcript = ""

            # Stop assistant speaking state
            _agent_speaking = False

            # Cancel current speech
            if _current_speech_handle:
                try:
                    _current_speech_handle.cancel()
                    logger.info("Current speech cancelled")
                except Exception as e:
                    logger.warning(f"Speech cancel failed: {e}")

            logger.debug("User started speaking")

    @session.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        nonlocal _user_speaking, _last_speech_time
        _user_speaking = False
        _last_speech_time = time.perf_counter()
        logger.debug("User stopped speaking")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        nonlocal _first_audio_time
        nonlocal _agent_speaking
        nonlocal _partial_transcript

        if ev.new_state == "speaking":
            _agent_speaking = True

            # Reset transcript state when AI starts speaking
            _partial_transcript = ""

            if _transcription_time is not None:
                _first_audio_time = time.perf_counter()

                latency_ms = (
                    _first_audio_time - _transcription_time
                ) * 1000

                logger.info(
                    f"AUDIO LATENCY: {latency_ms:.0f}ms"
                )

                _transcription_time = None

        else:
            _agent_speaking = False

    @session.on("agent_state_changed")
    def on_state_change(ev):
        if ev.new_state == "speaking":
            print("TTS started - LLM streaming is working!")

    async def stream_llm_to_tts(session, user_text):
        """
        Stream Ollama tokens directly into TTS sentence-by-sentence.
        """
        generated_chars = 0
        tts_queue = asyncio.Queue()

        nonlocal _agent_speaking
        nonlocal _interrupt_requested

        logger.info(f"Streaming response for: {user_text}")

        async def tts_worker():

            while True:

                chunk = await tts_queue.get()

                if chunk is None:
                    break

                try:
                    await session.say(chunk)
                except Exception as e:
                    logger.warning(f"TTS worker error: {e}")

                tts_queue.task_done()

        # Start TTS worker
        worker_task = asyncio.create_task(tts_worker())

        ctx = ChatContext()
        ctx.append(
            role="system",
            text="Reply under 5 words."
        )

        ctx.append(
            role="user",
            text=user_text
        )

        stream = session.llm.chat(chat_ctx=ctx)

        current_sentence = ""

        async for chunk in stream:

            # interruption support
            if _interrupt_requested:
                logger.info("LLM stream interrupted")
                break

            token = chunk.delta.content or ""

            generated_chars += len(token)

            # hard cutoff
            if generated_chars > 20:
                logger.info("Generation cutoff reached")
                break

            if not token:
                continue

            current_sentence += token

            # FORCE ultra-fast streaming
            if len(current_sentence.strip()) >= 8:

                chunk_text = current_sentence.strip()

                logger.info(f"EARLY TTS: {chunk_text}")

                _agent_speaking = True

                await tts_queue.put(chunk_text)

                current_sentence = ""
                
            '''if len(current_sentence) > 30:
                should_emit = True

            logger.info(f"TOKEN: {token}") '''

            '''# sentence boundary detection
            words = current_sentence.split()

            should_emit = (
                len(current_sentence) > 8
                or any(p in token for p in [".", "!", "?"])
            )

            if should_emit:

                chunk_text = current_sentence.strip()

                if chunk_text:

                    logger.info(f"TTS CHUNK: {chunk_text}")

                    _agent_speaking = True
                    chunk_text = chunk_text[:50]
                    await tts_queue.put(chunk_text)
                    

                current_sentence = "" '''

        # leftover partial sentence
        if (
            len(current_sentence.strip()) > 3
            and not _interrupt_requested
        ):
            await tts_queue.put(current_sentence.strip())

        # Keep only recent conversation turns
        if hasattr(session, "_chat_ctx"):
            session._chat_ctx.messages = session._chat_ctx.messages[-2:]

        await tts_queue.put(None)

        await worker_task
        
        _agent_speaking = False
    # =========================================================================
    async def transcript_monitor():
        nonlocal _partial_transcript
        nonlocal _user_speaking
        nonlocal _last_speech_time

        while True:
            await asyncio.sleep(0.1)

            if not _partial_transcript:
                continue

            silence_duration = time.perf_counter() - _last_speech_time

            # Finalize only if:
            # - user stopped speaking
            # - enough silence passed
            should_finalize = (
                not _user_speaking
                and silence_duration > 1.0
                and len(_partial_transcript.split()) > 2
            )

            if should_finalize:
                final_text = _partial_transcript.strip()

                logger.info(f"FINALIZED: {final_text}")

                _partial_transcript = ""

                nonlocal _current_speech_handle
                nonlocal _interrupt_requested

                _interrupt_requested = False 

                await stream_llm_to_tts(session, final_text)

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=OptimizedVoiceAssistant(),
        room_input_options=RoomInputOptions(),
    )

    await session.say("Hello, ready.")

    asyncio.create_task(transcript_monitor())

    # Send optimized greeting
    

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
