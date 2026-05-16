from __future__ import annotations

import asyncio
import io
import wave
import numpy as np
import websockets
import json
import logging

from livekit.agents import stt, APIConnectOptions, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit import rtc

logger = logging.getLogger(__name__)


class GrokSTT(stt.STT):

    def __init__(
        self,
        api_key: str,
        model: str = "grok-1",
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,        # ← NOW streaming
                interim_results=False,  # ← sends partials as user speaks
            )
        )

        self.api_key = api_key
        self._model = model
        self.language = language

    async def _recognize_impl(self, *args, **kwargs):
        raise NotImplementedError(
            "Non-streaming recognition is not supported"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions | None = None,
    ) -> "GrokSTTStream":
        """LiveKit calls this to get a streaming session."""
        lang = language if language is not NOT_GIVEN else self.language
        return GrokSTTStream(
            stt=self,
            language=lang,
            conn_options=conn_options or APIConnectOptions(),
        )


class GrokSTTStream(stt.SpeechStream):
    """
    Streaming STT session.

    Flow:
    1. LiveKit pushes AudioFrames via push_frame()
    2. We forward raw PCM bytes to xAI WebSocket continuously
    3. xAI sends back transcript.partial events as user speaks
    4. When xAI sends speech_final=True we emit FINAL_TRANSCRIPT
    5. LiveKit triggers the LLM immediately — no VAD delay
    """

    def __init__(
        self,
        *,
        stt: GrokSTT,
        language: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(
            stt=stt,
            conn_options=conn_options,
        )
        self._stt = stt
        self._language = language

    async def _run(self) -> None:
        """Main streaming loop — runs for the lifetime of the session."""

        ws_url = (
            f"wss://api.x.ai/v1/stt"
        )

        try:
            async with websockets.connect(
                ws_url,
                additional_headers={
                    "Authorization": f"Bearer {self._stt.api_key}"
                },
                ping_interval=20,
                ping_timeout=10,
            ) as ws:

                logger.info("GrokSTT WebSocket connected")

                # Run sender and receiver concurrently
                await asyncio.gather(
                    self._send_audio(ws),
                    self._recv_transcripts(ws),
                )

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"GrokSTT WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"GrokSTT stream error: {e}")

    async def _send_audio(self, ws) -> None:
        """
        Read AudioFrames from LiveKit and send PCM bytes to xAI.
        LiveKit pushes frames via the input channel (_input_ch).
        """
        try:
            async for frame in self._input_ch:

                # End of stream signal
                if frame is None:
                    logger.info("GrokSTT: end of audio stream")
                    await ws.send(json.dumps({"type": "audio.done"}))
                    break

                if not isinstance(frame, rtc.AudioFrame):
                    continue

                # Resample to 16kHz if needed (xAI expects 16kHz)
                audio_data = np.frombuffer(frame.data, dtype=np.int16)

                if frame.sample_rate != 16000:
                    # Simple resample by decimation/interpolation
                    ratio = 16000 / frame.sample_rate
                    new_length = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                    audio_data = np.interp(
                        indices,
                        np.arange(len(audio_data)),
                        audio_data
                    ).astype(np.int16)

                # Send raw PCM bytes
                await ws.send(audio_data.tobytes())

        except Exception as e:
            logger.error(f"GrokSTT send error: {e}")

    async def _recv_transcripts(self, ws) -> None:
        """
        Receive transcript events from xAI and emit to LiveKit.
        """
        try:
            async for message in ws:
                result = json.loads(message)
                event_type = result.get("type", "")

                if event_type == "transcript.partial":
                    text = result.get("text", "").strip()
                    is_final = result.get("is_final", False)
                    speech_final = result.get("speech_final", False)

                    if not text:
                        continue

                    logger.debug(f"GrokSTT partial: '{text}' "
                                f"is_final={is_final} "
                                f"speech_final={speech_final}")

                    if speech_final:
                        # User finished speaking — emit final transcript
                        # LiveKit will immediately trigger LLM
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        start_time=0,
                                        end_time=0,
                                        language=self._language,
                                    )
                                ],
                            )
                        )
                        logger.info(f"GrokSTT FINAL: '{text}'")

                    elif is_final or text:
                        # Interim result — show as user speaks
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        start_time=0,
                                        end_time=0,
                                        language=self._language,
                                    )
                                ],
                            )
                        )

                elif event_type == "transcript.done":
                    text = result.get("text", "").strip()
                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        start_time=0,
                                        end_time=0,
                                        language=self._language,
                                    )
                                ],
                            )
                        )

                elif event_type == "error":
                    logger.error(f"GrokSTT error from xAI: {result}")

        except websockets.exceptions.ConnectionClosedError:
            pass
        except Exception as e:
            logger.error(f"GrokSTT recv error: {e}")