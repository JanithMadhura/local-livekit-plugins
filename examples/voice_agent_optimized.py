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
from livekit.agents import AgentSession, Agent, RoomInputOptions, StopResponse
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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")  # tiny.en is fastest for English
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "float16" if WHISPER_DEVICE == "cuda" else "int8",
)
WHISPER_MAX_AUDIO_SECONDS = float(os.getenv("WHISPER_MAX_AUDIO_SECONDS", "2.0"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"
FINALIZE_SILENCE_SECONDS = float(os.getenv("FINALIZE_SILENCE_SECONDS", "0.6"))
MIN_FINAL_TRANSCRIPT_CHARS = int(os.getenv("MIN_FINAL_TRANSCRIPT_CHARS", "2"))

# TTS: Streaming with 1.5x speed for lower latency
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_SPEED = float(os.getenv("PIPER_SPEED", "1.0"))  # 50% faster speech
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"

# LLM: Use faster model if available
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")  # use `ollama list` for installed names
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "20"))
EARLY_TTS_WORDS = max(1, int(os.getenv("EARLY_TTS_WORDS", "3")))
MAX_TTS_WORDS = max(EARLY_TTS_WORDS, int(os.getenv("MAX_TTS_WORDS", "8")))


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
                - Maximum 6 words
                - One sentence only
                - No apologies
                - No preamble
                - No explanations
                - No repetition
                - No extra details
                - No safety messages
                - Stop immediately after answering
                """

        )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Stop LiveKit's automatic reply; the manual stream handles TTS."""
        raise StopResponse()


# =============================================================================
# Pipeline Factory - Optimized Configuration
# =============================================================================

def create_optimized_local_session() -> AgentSession:
    """Create a low-latency local processing pipeline."""

    logger.info("=" * 70)
    logger.info("OPTIMIZED LOW-LATENCY PIPELINE")
    logger.info("=" * 70)
    logger.info(
        f"  STT: FasterWhisper ({WHISPER_MODEL} on {WHISPER_DEVICE}, "
        f"{WHISPER_COMPUTE_TYPE}, max_audio={WHISPER_MAX_AUDIO_SECONDS}s)"
    )
    logger.info(f"  LLM: Ollama ({OLLAMA_MODEL}) - manual streaming, max_tokens={OLLAMA_MAX_TOKENS}")
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
            vad_filter=WHISPER_VAD_FILTER,
            streaming=True,
            vad=stt_vad,
            max_audio_seconds=WHISPER_MAX_AUDIO_SECONDS,
        ),
        # LLM: Streaming is automatic with LiveKit agents
        llm=lk_openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
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
                    _current_speech_handle.interrupt(force=True)
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
        emitted_words = 0
        tts_queue = asyncio.Queue()

        nonlocal _agent_speaking
        nonlocal _current_speech_handle
        nonlocal _interrupt_requested

        logger.info(f"Streaming response for: {user_text}")

        async def tts_worker():
            nonlocal _current_speech_handle

            while True:

                chunk = await tts_queue.get()

                if chunk is None:
                    break

                if _interrupt_requested:
                    logger.info(f"Dropping queued TTS chunk after interruption: {chunk}")
                    tts_queue.task_done()
                    continue

                try:
                    _current_speech_handle = session.say(
                        chunk,
                        add_to_chat_ctx=False,
                    )
                    await _current_speech_handle
                except Exception as e:
                    logger.warning(f"TTS worker error: {e}")
                finally:
                    _current_speech_handle = None

                tts_queue.task_done()

        # Start TTS worker
        worker_task = asyncio.create_task(tts_worker())

        ctx = ChatContext()
        ctx.add_message(
            role="system",
            content=(
                "Reply in 3-6 words. One sentence. "
                "No apologies. No preamble. If unclear, say: Please repeat."
            )
        )

        ctx.add_message(
            role="user",
            content=user_text
        )

        current_sentence = ""
        stop_streaming = False
        first_chunk_time: float | None = None

        try:
            async with session.llm.chat(
                chat_ctx=ctx,
                extra_kwargs={"max_tokens": OLLAMA_MAX_TOKENS},
            ) as stream:
                async for chunk in stream:

                    # interruption support
                    if _interrupt_requested:
                        logger.info("LLM stream interrupted")
                        break

                    if not chunk.delta or not chunk.delta.content:
                        continue

                    token = chunk.delta.content
                    generated_chars += len(token)

                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                        if _transcription_time is not None:
                            logger.info(
                                "LLM TTFT: %.0fms",
                                (first_chunk_time - _transcription_time) * 1000,
                            )

                    logger.info(
                        "LLM RAW CHUNK: %d chars: %r",
                        len(token),
                        token[:120],
                    )

                    current_sentence += token

                    # Ollama/LiveKit deltas can be large. Split them here so TTS
                    # sees tiny chunks even when the upstream stream is buffered.
                    parts = current_sentence.split()

                    # Do not wait for the full word chunk when the model has
                    # already produced a clean short acknowledgement.
                    if (
                        parts
                        and len(parts) < EARLY_TTS_WORDS
                        and current_sentence.strip().endswith((".", "!", "?", ",", ";", ":"))
                        and emitted_words < MAX_TTS_WORDS
                        and not _interrupt_requested
                    ):
                        remaining_words = MAX_TTS_WORDS - emitted_words
                        words_to_emit = min(len(parts), remaining_words)
                        chunk_text = " ".join(parts[:words_to_emit])

                        logger.info(f"EARLY TTS: {chunk_text}")

                        _agent_speaking = True

                        await tts_queue.put(chunk_text)

                        emitted_words += words_to_emit
                        parts = parts[words_to_emit:]

                    words = current_sentence.strip().split()

                    should_emit = (
                        len(words) >= EARLY_TTS_WORDS
                        and (
                            token.endswith((" ", ".", "!", "?", ","))
                            or len(words) >= MAX_TTS_WORDS
                        )
                    )

                    if (
                        should_emit
                        and emitted_words < MAX_TTS_WORDS
                        and not _interrupt_requested
                    ):

                        chunk_text = current_sentence.strip()

                        logger.info(f"EARLY TTS: {chunk_text}")

                        _agent_speaking = True

                        await tts_queue.put(chunk_text)

                        emitted_words += len(words)

                        current_sentence = ""

                    current_sentence = " ".join(parts)

                    if emitted_words >= MAX_TTS_WORDS and not stop_streaming:
                        logger.info("Local word cutoff reached")
                        stop_streaming = True

                    if stop_streaming:
                        break
        except Exception as e:
            logger.exception(f"LLM stream failed: {e}")

        # leftover partial sentence
        leftover_text = current_sentence.strip()
        if (
            any(char.isalnum() for char in leftover_text)
            and not _interrupt_requested
            and emitted_words < MAX_TTS_WORDS
        ):
            remaining_words = MAX_TTS_WORDS - emitted_words
            chunk_text = " ".join(leftover_text.split()[:remaining_words])

            if chunk_text:
                logger.info(f"EARLY TTS: {chunk_text}")
                _agent_speaking = True
                await tts_queue.put(chunk_text)

        logger.info(
            "LLM stream complete: %d raw chars, %d emitted words",
            generated_chars,
            emitted_words,
        )

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
            transcript = _partial_transcript.strip()
            has_speech_text = (
                len(transcript) >= MIN_FINAL_TRANSCRIPT_CHARS
                and any(char.isalnum() for char in transcript)
            )

            # Finalize only if:
            # - user stopped speaking
            # - enough silence passed
            should_finalize = (
                not _user_speaking
                and silence_duration > FINALIZE_SILENCE_SECONDS
                and has_speech_text
            )

            if should_finalize:
                final_text = transcript

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
