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
        #stt=FasterWhisperSTT(
        #    model_size=WHISPER_MODEL,
        #    device=WHISPER_DEVICE,
        #    compute_type=WHISPER_COMPUTE_TYPE,  # Quantized for speed
        #    vad_filter=WHISPER_VAD_FILTER,
        #    streaming=False,  # Enable streaming mode
        #   vad=stt_vad,
        #    max_audio_seconds=WHISPER_MAX_AUDIO_SECONDS,
        #), 

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

    # Transcript stabilization state
    _partial_transcript = ""
    _last_speech_time = time.perf_counter()
    _user_speaking = False
    _eou_probability = 0.0
    _agent_speaking = False
    _current_speech_handle = None
    _interrupt_requested = False

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        nonlocal _stt_start_time
        nonlocal _chunk_count
        nonlocal _first_audio_time
        nonlocal _partial_transcript

        # ignore assistant echo
        

        _transcription_time = time.perf_counter()
        _chunk_count = 0
        _first_audio_time = None

        text = ev.transcript.strip()
        if _stt_start_time is None:
            _stt_start_time = time.perf_counter()

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
        logger.info("VAD DETECTED SPEECH")

        _user_speaking = True

        # User interrupted assistant speech
        # only interrupt on REAL transcript
        if (
            _agent_speaking
            and len(_partial_transcript.strip().split()) >= 2
        ):

            logger.info("Real user interruption detected")

            _interrupt_requested = True

            # cancel current speech
            if _current_speech_handle:
                try:
                    _current_speech_handle.interrupt(force=True)
                    logger.info("Current speech cancelled")
                except Exception as e:
                    logger.warning(f"Speech cancel failed: {e}")

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
        first_llm_token_received = False
        tts_queue = asyncio.Queue()

        nonlocal _agent_speaking
        nonlocal _current_speech_handle
        nonlocal _interrupt_requested


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
            "question":     ["Interesting question. Let me think for a second."],
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

        async def tts_worker():
            nonlocal _current_speech_handle

            while True:
                chunk = await tts_queue.get()

                if chunk is None:
                    tts_queue.task_done()
                    break

                if _interrupt_requested:
                    logger.info(f"Dropping TTS chunk: {chunk}")
                    tts_queue.task_done()
                    continue

                try:
                    logger.info(f"TTS SPEAKING: {chunk}")
                    handle = session.say(chunk, add_to_chat_ctx=False)
                    _current_speech_handle = handle
                    await handle  # ← wait for THIS chunk to fully finish
                    await asyncio.sleep(0.05)  # tiny gap between chunks
                except Exception as e:
                    logger.warning(f"TTS worker error: {e}")
                finally:
                    _current_speech_handle = None
                    _agent_speaking = False

                tts_queue.task_done()

        # Start TTS worker
        worker_task = asyncio.create_task(tts_worker())

        ctx = ChatContext()
        ctx.add_message(
            role="system",
            content=(
                "Reply naturally in one short sentence. "
                "Maximum 10 words. "
                "Never give long explanations."
            )
        )

        ctx.add_message(
            role="user",
            content=user_text
        )

        current_sentence = ""
        stop_streaming = False
        first_chunk_time: float | None = None

        # This manual streaming loop allows us to push tokens into TTS as they arrive, without waiting for the full response.
        filler_played = False

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

                    # Trigger filler on very first LLM token, before any TTS emission.
                    if (
                        not filler_played
                        and not emitted_words
                        and len(words) >= 1
                        and not _interrupt_requested
                    ):
                        filler_played = True
                        logger.info("Speaking filler FIRST")

                        selected_filler = get_emotion_filler(user_text)

                        logger.info(f"Selected filler: {selected_filler}")

                        # Send filler through SAME TTS worker
                        await tts_queue.put(selected_filler)

                        # Give filler time to play first
                        await asyncio.sleep(0.3)# Give filler some breathing room before TTS starts

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


    async def transcript_monitor():
        nonlocal _partial_transcript
        nonlocal _user_speaking
        nonlocal _last_speech_time
        nonlocal _stt_start_time

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

                if _stt_start_time is not None:

                    stt_ms = (
                        time.perf_counter() - _stt_start_time
                    ) * 1000

                    print(f"\nSTT TIME: {stt_ms:.0f}ms")

                    _stt_start_time = None

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

    #=========================================
    '''wav_path = os.path.join(
        os.path.dirname(__file__),
        "test.wav"
    )

    await inject_wav_audio(session, wav_path)'''
    #=====================================================================================

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