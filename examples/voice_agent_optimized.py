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
import random

from transformers import pipeline as hf_pipeline

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
from local_livekit_plugins import GrokSTT, GrokTTS
from local_livekit_plugins.piper_tts_streaming import PiperTTSStreaming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("voice-agent-optimized")

# =============================================================================
# Emotion Classifier
# =============================================================================
emotion_classifier = hf_pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=-1,  # CPU; change to 0 for GPU
)

intent_classifier = hf_pipeline(
    "text-classification",
    model="Wyona/message-classification-question-other-smalltalk-modified",
    top_k=1,
    device=-1,
)

def detect_intent(text: str) -> str:
    """Detect intent: question / smalltalk / other"""
    try:
        result = intent_classifier(text)
        label = result[0][0]["label"].lower()
        score = result[0][0]["score"]
        logger.info(f"Intent: {label} (confidence: {score:.2f})")
        if score < 0.65:
            return "other"
        return label
    except Exception as e:
        logger.warning(f"Intent detection failed: {e}")
        return "other"

def detect_emotion(text: str) -> str:
    try:
        result = emotion_classifier(text)
        label = result[0][0]["label"].lower()
        score = result[0][0]["score"]

        logger.info(f"Emotion: {label} (confidence: {score:.2f})")

        if score < 0.70:
            return "neutral"

        # Map GoEmotions 28 labels down to your 7 filler categories
        positive = {"joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"}
        negative_sad = {"sadness", "grief", "remorse", "disappointment"}
        negative_angry = {"anger", "annoyance", "disapproval"}
        scared = {"fear", "nervousness"}
        surprised = {"surprise"}

        if label in positive:
            return "joy"
        elif label in negative_sad:
            return "sadness"
        elif label in negative_angry:
            return "anger"
        elif label in scared:
            return "fear"
        elif label in surprised:
            return "surprise"
        else:
            return "neutral"

    except Exception as e:
        logger.warning(f"Emotion detection failed: {e}")
        return "neutral"
        
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("local_livekit_plugins").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.WARNING)


# =============================================================================
# Configuration - Optimized for Latency
# =============================================================================

USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"

# STT: Use smaller/faster Whisper model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-small.en")  # tiny.en is fastest for English
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "float16" if WHISPER_DEVICE == "cuda" else "int8",
)
WHISPER_MAX_AUDIO_SECONDS = float(os.getenv("WHISPER_MAX_AUDIO_SECONDS", "0.0"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"
FINALIZE_SILENCE_SECONDS = float(os.getenv("FINALIZE_SILENCE_SECONDS", "1.5"))
MIN_FINAL_TRANSCRIPT_CHARS = int(os.getenv("MIN_FINAL_TRANSCRIPT_CHARS", "2"))

# TTS: Streaming with 1.5x speed for lower latency
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_SPEED = float(os.getenv("PIPER_SPEED", "1.0"))  # 50% faster speech
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"

# LLM: Use faster model if available
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")  # use `ollama list` for installed names
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "25"))
EARLY_TTS_WORDS = max(1, int(os.getenv("EARLY_TTS_WORDS", "2")))
MAX_TTS_WORDS = max(EARLY_TTS_WORDS, int(os.getenv("MAX_TTS_WORDS", "20")))


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
        """Normalize the final user message, invoke the streaming handler, then stop the default reply.

        This runs the filler immediately and starts LLM streaming without waiting for the transcript monitor.
        """
        # Extract text from the message (may be string/object/list)
        raw = new_message.content if hasattr(new_message, "content") else new_message

        if isinstance(raw, list):
            parts: list[str] = []
            for item in raw:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "content" in item:
                    parts.append(str(item.get("content", "")))
                elif hasattr(item, "content"):
                    parts.append(str(item.content))
                else:
                    parts.append(str(item))
            user_text = " ".join(p for p in (p.strip() for p in parts) if p)
        else:
            user_text = str(raw).strip()

        handler = getattr(self.session, "_stream_llm_to_tts", None)
        # Optionally require end-of-utterance confirmation from the turn detector
        eou_threshold = float(os.getenv("EOU_THRESHOLD", "0.05"))
        if handler and user_text:
            try:
                td = getattr(self.session, "turn_detection", None)
                if td is not None:
                    # Debug: inspect what is being passed to the turn detector.
                    # Attempt to extract ChatContext messages (role/content) so we
                    # can compare the exact input the model sees.
                    try:
                        logger.debug("Turn detector context repr: %r", turn_ctx)

                        msgs = None
                        if hasattr(turn_ctx, "items"):
                            msgs = turn_ctx.items

                        if msgs is not None:
                            for i, m in enumerate(msgs):
                                try:
                                    role = getattr(m, "role", None)
                                    text = None
                                    if hasattr(m, "text_content"):
                                        text = m.text_content
                                    elif hasattr(m, "content"):
                                        # content is a list; join strings for logging
                                        try:
                                            text_parts = [c for c in m.content if isinstance(c, str)]
                                            text = "\n".join(text_parts) if text_parts else None
                                        except Exception:
                                            text = repr(m.content)

                                    logger.debug("TD msg %d: role=%r text=%r", i, role, text)
                                except Exception:
                                    logger.debug("TD msg %d raw: %r", i, m)
                        else:
                            # Fallback: log string conversion (trimmed)
                            try:
                                s = str(turn_ctx)
                                logger.debug("turn_ctx str (trim): %r", s[:200])
                            except Exception:
                                logger.debug("Unable to stringify turn_ctx")
                    except Exception:
                        logger.debug("Unable to introspect turn_ctx for TD input")

                    try:
                        # Ensure the turn-detector sees the final user message.
                        # Some callers pass a ChatContext that only contains the assistant
                        # state; append the user's final transcript so the model has
                        # both assistant+user context (matching the plugin's own
                        # internal calls that produced the higher score).
                        try:
                            probe_ctx = turn_ctx.copy()
                        except Exception:
                            probe_ctx = ChatContext()

                        try:
                            probe_ctx.add_message(role="user", content=user_text)
                        except Exception:
                            # best-effort: if add_message isn't available, ignore
                            pass

                        # Initial check
                        prob = await td.predict_end_of_turn(probe_ctx, timeout=1.0)
                        logger.info("EOU prob: %.3f (threshold: %.3f)", prob, eou_threshold)

                        # If below threshold, poll a few times (short window) to wait
                        # for a more complete transcript or updated model state.
                        if prob < eou_threshold:
                            poll_timeout = float(os.getenv("EOU_POLL_TIMEOUT", "3.0"))
                            poll_interval = float(os.getenv("EOU_POLL_INTERVAL", "0.1"))
                            start_t = time.perf_counter()
                            logger.debug("EOU below threshold, polling up to %.2fs", poll_timeout)
                            while time.perf_counter() - start_t < poll_timeout:
                                await asyncio.sleep(poll_interval)
                                try:
                                    # try to refresh probe_ctx with any updated context
                                    try:
                                        updated = turn_ctx.copy()
                                        probe_ctx = updated
                                        probe_ctx.add_message(role="user", content=user_text)
                                    except Exception:
                                        pass

                                    prob = await td.predict_end_of_turn(probe_ctx, timeout=0.5)
                                    logger.info("EOU poll prob: %.3f", prob)
                                    if prob >= eou_threshold:
                                        break
                                except Exception as e:
                                    logger.debug("EOU poll failed: %s", e)

                            if prob < eou_threshold:
                                logger.info("EOU below threshold after polling; forcing response due to timeout")
                                # Do NOT raise StopResponse here — the poll window
                                # expired but we still want to respond. Continue
                                # to the streaming handler below.
                    except StopResponse:
                        raise
                    except Exception as e:
                        logger.warning("EOU check failed, proceeding: %s", e)

                await handler(self.session, user_text)
            except StopResponse:
                # expected control-flow to stop the default reply
                raise
            except Exception:
                logger.exception("stream handler failed in on_user_turn_completed")

        # Prevent LiveKit's default reply; we've handled the turn.
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
        min_speech_duration=0.3,
        min_silence_duration=1.8,
        prefix_padding_duration=0.3,
        max_buffered_speech=8.0,
        activation_threshold=0.20,
    )

    if not PIPER_MODEL_PATH:
        raise ValueError(
            "PIPER_MODEL_PATH not set. Download a voice model from:\n"
            "https://huggingface.co/rhasspy/piper-voices"
        )

    return AgentSession(
        # STT: Streaming-capable whisper with VAD-based chunking
        # stt=FasterWhisperSTT(
        #    model_size=WHISPER_MODEL,
        #    device=WHISPER_DEVICE,
        #    compute_type=WHISPER_COMPUTE_TYPE,  # Quantized for speed
        #    vad_filter=WHISPER_VAD_FILTER,
        #    streaming=False,  # Enable streaming mode
        #    vad=stt_vad,
        #    max_audio_seconds=WHISPER_MAX_AUDIO_SECONDS,
        # ), 

        stt=GrokSTT(
            api_key=os.getenv("GROK_API_KEY_STT"),
            model="grok-1",
            language="en",
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

        # tts=GrokTTS(
        #    api_key=os.getenv("GROK_API_KEY_TTS"),
        #    voice="eve",
        #    language="en",
        #    sample_rate=24000,
        # ),
        vad=stt_vad,
        turn_detection=turn_detector,
        # Endpointing and interruption tuning (floats/ints expected)
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
        min_interruption_duration=0.5,
        min_interruption_words=0,
        # Disable speculative preemptive generation and automatic interruptions
        preemptive_generation=False,
        allow_interruptions=False,
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
    _stt_start_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        nonlocal _stt_start_time
        nonlocal _chunk_count
        nonlocal _first_audio_time

        _transcription_time = time.perf_counter()
        _chunk_count = 0
        _first_audio_time = None

        text = ev.transcript.strip()
        if _stt_start_time is None:
            _stt_start_time = time.perf_counter()

        if text:
            logger.info(f"PARTIAL: {text}")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        nonlocal _first_audio_time

        if ev.new_state == "speaking":
            if _transcription_time is not None:
                _first_audio_time = time.perf_counter()

                latency_ms = (
                    _first_audio_time - _transcription_time
                ) * 1000

                logger.info(
                    f"AUDIO LATENCY: {latency_ms:.0f}ms"
                )

                _transcription_time = None


    @session.on("agent_state_changed")
    def on_state_change(ev):
        if ev.new_state == "speaking":
            print("TTS started - LLM streaming is working!")

    @session.on("metrics_collected")
    def on_metrics(ev):
        metrics = ev.metrics if hasattr(ev, "metrics") else ev

        # LiveKit turn detector metrics
        if hasattr(metrics, "eou_probability"):
            logger.info(
                f"EOU UPDATED: {metrics.eou_probability:.3f}"
            )

    async def stream_llm_to_tts(session, user_text):
        """
        Stream Ollama tokens directly into TTS sentence-by-sentence.
        """
        generated_chars = 0
        emitted_words = 0
        first_llm_token_received = False
        tts_queue = asyncio.Queue()

        # Internal interrupt flag — set by the tts_worker on cancellation
        _interrupt_requested = False
        _agent_speaking = False
        _current_speech_handle = None

        logger.info(f"Streaming response for: {user_text}")

        # fillers = {
        #     "question": [
        #         "Good question.",
        #         "Interesting.",
        #         "Let me think.",
        #     ],

        #     "explanation": [
        #         "Alright.",
        #         "Okay.",
        #         "Let me explain.",
        #     ],

        #     "default": [
        #         "Okay.",
        #         "Sure.",
        #     ]
        # }

        # def detect_filler_type(user_text: str):

        #     text = user_text.lower()

        #     # explanation / learning
        #     if any(word in text for word in [
        #         "explain",
        #         "why",
        #         "how",
        #         "teach",
        #         "what is",
        #     ]):
        #         return "explanation"

        #     # question
        #     if "?" in text:
        #         return "question"

        #     return "default"

        EMOTION_FILLERS = {
            # Emotion-based
            "joy":          ["Great!", "Love that!", "Awesome!"],
            "sadness":      ["I understand.", "That sounds hard.", "I hear you."],
            "anger":        ["Fair point.", "I get that.", "Let me help."],
            "fear":         ["It's okay.", "No worries.", "You're safe."],
            "surprise":     ["Wow!", "No way!", "That's wild!"],

            # Intent-based (for neutral emotion)
            "smalltalk":    ["Hey there.", "Hi there!"],
            "question":     ["Interesting question.", "Let me think about that.", "Let me think for a moment."],
            "other":        ["Alright.", "Got it.", "Sure."],
        }


        def get_emotion_filler(user_text: str) -> str:
            # Stage 1 — check emotion
            emotion = detect_emotion(user_text)
            logger.info(f"DETECTED EMOTION: {emotion}")

            if emotion != "neutral":
                return random.choice(EMOTION_FILLERS.get(emotion, EMOTION_FILLERS["other"]))

            # Stage 2 — greeting check BEFORE intent classifier
            text = user_text.lower().strip()
            greetings = ["how are you", "how are u", "hey", "hi ", "hello", "what's up", "whats up"]
            if any(g in text for g in greetings):
                logger.info("DETECTED INTENT: smalltalk (greeting)")
                return random.choice(EMOTION_FILLERS["smalltalk"])

            # Stage 3 — intent classifier
            intent = detect_intent(user_text)
            logger.info(f"DETECTED INTENT: {intent}")
            return random.choice(EMOTION_FILLERS.get(intent, EMOTION_FILLERS["other"]))
                # async def play_filler():

        #     try:

        #         filler_type = detect_filler_type(user_text)

        #         selected_filler = random.choice(
        #             fillers[filler_type]
        #         )

        #         logger.info(
        #             f"Selected filler: {selected_filler}"
        #         )

        #         await session.say(
        #             selected_filler,
        #             add_to_chat_ctx=False,
        #         )

        #     except Exception as e:

        #         logger.warning(f"Filler playback failed: {e}")

        def clean_llm_output(text: str) -> str:
            """Strip phi3 training artifacts and markdown from LLM output."""
            import re
            # Cut at the first --- or ** (phi3 instruction bleed)
            text = re.split(r'---|^\*\*|\*\*', text)[0]
            # Remove surrounding quotes
            text = text.strip().strip('"').strip("'")
            # Remove markdown bold/italic/headers
            text = re.sub(r'[*#`]+', '', text)
            # Collapse whitespace
            text = ' '.join(text.split())
            return text.strip()

        async def play_filler():
            try:
                selected_filler = get_emotion_filler(user_text)
                logger.info(f"Selected filler: {selected_filler}")
                # Put filler as FIRST item in tts_queue
                # tts_worker will speak it before anything else
                await tts_queue.put(selected_filler)

                await asyncio.sleep(0.05)
            except Exception as e:
                logger.warning(f"Filler playback failed: {e}")

        # delayed_filler removed — filler is now triggered exclusively
        # when the first LLM token arrives (EARLY_TTS word-count gate).

        # Event set when the filler finishes playing.
        # The LLM loop awaits this before releasing any LLM-generated chunks,
        # so the filler always plays fully before the real response starts.
        filler_done_event = asyncio.Event()
        filler_done_event.set()  # default: no filler pending, nothing to wait for

        async def tts_worker():
            first_chunk = True

            while True:
                chunk = await tts_queue.get()

                if chunk is None:
                    tts_queue.task_done()
                    break

                if _interrupt_requested:
                    logger.info(f"Dropping TTS chunk: {chunk}")
                    tts_queue.task_done()
                    # Unblock LLM loop even on interrupt
                    filler_done_event.set()
                    first_chunk = False
                    continue

                try:

                    logger.info(f"TTS SPEAKING: {chunk}")
                    handle = session.say(chunk, add_to_chat_ctx=False)
                    _current_speech_handle = handle
                    await handle  # wait for THIS chunk to fully finish
                    await asyncio.sleep(0.05)  # tiny gap between chunks
                except Exception as e:
                    logger.warning(f"TTS worker error: {e}")
                finally:
                    _current_speech_handle = None
                    # Signal filler done after the FIRST chunk (the filler) finishes.
                    # Subsequent LLM chunks are already unblocked at this point.
                    if first_chunk:
                        filler_done_event.set()
                        first_chunk = False

                tts_queue.task_done()

            # Only clear agent_speaking once the ENTIRE queue has drained.
            _agent_speaking = False

        # Detect filler synchronously FIRST (it's a regular function, not async).
        # This means selected_filler is known immediately — no blocking needed —
        # so we can embed it in the system prompt AND queue it for TTS at the same time.
        selected_filler = get_emotion_filler(user_text)
        logger.info(f"Selected filler: {selected_filler}")

        # Build ctx NOW, while filler is queued but not yet playing.
        # The LLM stream won't open until after this, so there's no race.
        ctx = ChatContext()
        ctx.add_message(
            role="system",
            content=(
                "Reply naturally in one short sentence. "
                "Maximum 10 words. "
                "Never give long explanations. "
                f'You already said "{selected_filler}" as an acknowledgement — '
                "do NOT repeat it or start with any greeting. "
                "Go straight to answering."
            )
        )

        # Start TTS worker and queue filler — runs concurrently with LLM stream below.
        # filler_done_event ensures LLM chunks don't enter the queue until filler finishes.
        worker_task = asyncio.create_task(tts_worker())
        filler_done_event.clear()
        _agent_speaking = True
        await tts_queue.put(selected_filler)
        # No blocking wait here — the filler plays concurrently while the LLM warms up.
        # tts_worker will set filler_done_event when filler finishes, which gates
        # any LLM chunks from entering the queue via the in-loop filler_done_event.wait().

        ctx.add_message(
            role="user",
            content=user_text
        )

        current_sentence = ""
        stop_streaming = False
        first_chunk_time: float | None = None

        # This manual streaming loop allows us to push tokens into TTS as they arrive, without waiting for the full response.
        filler_played = True   # already played eagerly above

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

                    if not first_llm_token_received:

                        first_llm_token_received = True

                    generated_chars += len(token)

                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                        if _transcription_time is not None:
                            logger.info(
                                "LLM TTFT: %.0fms",
                                (first_chunk_time - _transcription_time) * 1000,
                            )

                    #logger.info(
                    #    "LLM RAW CHUNK: %d chars: %r",
                    #    len(token),
                    #    token[:120],
                    #)

                    current_sentence += token

                    # Ollama/LiveKit deltas can be large. Split them here so TTS
                    # sees tiny chunks even when the upstream stream is buffered.
                    parts = current_sentence.split()
                    words = current_sentence.strip().split()

                    # Filler is now played eagerly before the LLM stream opens,
                    # so no in-loop trigger needed. filler_played=True already.

                    # Gate: wait for filler to finish before pushing any LLM chunk.
                    # This is a no-op if the filler already finished (event already set).
                    # Only blocks on very fast LLMs where tokens arrive before filler ends.
                    if not filler_done_event.is_set():
                        await filler_done_event.wait()

                    # Do not wait for the full word chunk when the model has
                    # already produced a clean short acknowledgement.
                    if (
                        parts
                        and len(parts) < EARLY_TTS_WORDS
                        and current_sentence.strip().endswith((".", "!", "?", ";", ":"))
                        and emitted_words < MAX_TTS_WORDS
                        and not _interrupt_requested
                    ):
                        remaining_words = MAX_TTS_WORDS - emitted_words
                        words_to_emit = min(len(parts), remaining_words)
                        chunk_text = clean_llm_output(" ".join(parts[:words_to_emit]))

                        if chunk_text:
                            logger.info(f"EARLY TTS: {chunk_text}")
                            _agent_speaking = True
                            await tts_queue.put(chunk_text)
                            await asyncio.sleep(0.15)
                            if emitted_words >= MAX_TTS_WORDS:
                                logger.info("Reached soft word target")
                            emitted_words += words_to_emit
                            parts = parts[words_to_emit:]

                    words = current_sentence.strip().split()

                    should_emit = (
                        len(words) >= EARLY_TTS_WORDS
                        and (
                            token.endswith((" ", ".", "!", "?"))
                            or len(words) >= MAX_TTS_WORDS
                        )
                    )

                    if (
                        should_emit
                        and emitted_words < MAX_TTS_WORDS
                        and not _interrupt_requested
                    ):

                        chunk_text = clean_llm_output(current_sentence.strip())

                        if chunk_text:
                            logger.info(f"EARLY TTS: {chunk_text}")
                            _agent_speaking = True
                            await tts_queue.put(chunk_text)
                            if emitted_words >= MAX_TTS_WORDS:
                                logger.info("Reached soft word target")
                            emitted_words += len(words)

                        current_sentence = ""
                        parts = []

                    current_sentence = " ".join(parts)

        except Exception as e:
            logger.exception(f"LLM stream failed: {e}")

        # leftover partial sentence
        leftover_text = clean_llm_output(current_sentence.strip())

        # ignore tiny leftovers
        if len(leftover_text.split()) < 2:
            leftover_text = ""

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
                if emitted_words >= MAX_TTS_WORDS:
                    logger.info("Reached soft word target")

        logger.info(
            "LLM stream complete: %d raw chars, %d emitted words",
            generated_chars,
            emitted_words,
        )

        await tts_queue.put(None)

        await worker_task
        
        _agent_speaking = False

    # Register the streaming handler so the Agent class can reach it via session attribute
    session._stream_llm_to_tts = stream_llm_to_tts

    # =========================================================================
    ## NOTE: The following WAV injection code is for testing and demonstration purposes.
    ## It simulates a user speaking from a pre-recorded audio file, allowing us to test the full pipeline (STT -> LLM -> TTS) with controlled input. In a
    '''async def inject_wav_audio(session, wav_path):

        import wave
        import numpy as np
        from livekit.rtc import AudioFrame

        logger.info(f"Injecting WAV audio: {wav_path}")

        wf = wave.open(wav_path, "rb")

        sample_rate = wf.getframerate()
        channels = wf.getnchannels()

        chunk_ms = 500
        chunk_size = int(sample_rate * chunk_ms / 1000)

        collected = []

        while True:

            frames = wf.readframes(chunk_size)

            if not frames:
                break

            audio_np = np.frombuffer(
                frames,
                dtype=np.int16
            )

            collected.append(audio_np)

            full_audio = np.concatenate(collected)

            frame = AudioFrame(
                data=full_audio.tobytes(),
                sample_rate=sample_rate,
                num_channels=channels,
                samples_per_channel=len(full_audio),
            )

            try:

                result = await session.stt.recognize(frame)

                if result.alternatives:

                    text = result.alternatives[0].text.strip()

                    if text:

                        logger.info(
                            f"WAV TRANSCRIPT: {text}"
                        )

                        await stream_llm_to_tts(
                            session,
                            text
                        )

                        break

            except Exception as e:
                logger.warning(f"WAV inject error: {e}")

            await asyncio.sleep(chunk_ms / 1000)

        logger.info("WAV injection complete")  '''
    #==================================================================================================


    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=OptimizedVoiceAssistant(),
        room_input_options=RoomInputOptions(),
    )

    await session.say("Hello, ready.")

    #=========================================
    '''wav_path = os.path.join(
        os.path.dirname(__file__),
        "test.wav"
    )

    await inject_wav_audio(session, wav_path)'''
    #=====================================================================================

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