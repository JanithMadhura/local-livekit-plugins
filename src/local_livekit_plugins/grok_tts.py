from __future__ import annotations

import json
import base64
import numpy as np
import websockets

from livekit.agents import tts
from livekit import rtc

import asyncio



class GrokTTS(tts.TTS):

    def __init__(
        self,
        api_key: str,
        voice: str = "eve",
        language: str = "en",
        sample_rate: int = 24000,
    ):

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self.api_key = api_key
        self.voice = voice
        self.language = language
        self._sample_rate = sample_rate

    def stream(self, conn_options=None):
        return GrokTTSStream(
            self,
            conn_options=conn_options,
        )

    def synthesize(
        self,
        text: str,
        conn_options=None,
    ):
        stream = self.stream()

        stream.push_text(text)
        stream.end_input()

        return stream

class GrokTTSStream(tts.SynthesizeStream):
    def __init__(
        self,
        tts_instance: GrokTTS,
        conn_options=None,
    ):

        super().__init__(
            tts=tts_instance,
            conn_options=conn_options,
        )

        self._tts = tts_instance

    async def _collect_audio(self, ws, output_emitter):
        """Receive audio chunks until audio.done."""
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            event_type = data.get("type", "")

            if event_type == "audio.delta":
                audio_bytes = base64.b64decode(data["delta"])
                output_emitter.push(audio_bytes)

            elif event_type == "audio.done":
                output_emitter.end_segment()
                output_emitter.flush()
                break

            elif event_type == "error":
                print("TTS ERROR:", data)
                break

    async def _run(self, output_emitter):

        ws_url = (
            f"wss://api.x.ai/v1/tts"
            f"?voice={self._tts.voice}"
            f"&language={self._tts.language}"
            f"&codec=pcm"
            f"&sample_rate={self._tts._sample_rate}"
            f"&optimize_streaming_latency=1"
        )

        output_emitter.initialize(
            request_id="grok_tts",
            sample_rate=self._tts._sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        segment_counter = 0
        segment_started = False

        async with websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {self._tts.api_key}"},
            ping_interval=20,
            ping_timeout=10,
        ) as ws:

            async for text in self._input_ch:

                # String token — stream it to xAI immediately
                if isinstance(text, str):
                    text_str = text.strip()
                    if not text_str:
                        continue

                    if not segment_started:
                        segment_id = f"segment_{segment_counter}"
                        segment_counter += 1
                        output_emitter.start_segment(segment_id=segment_id)
                        segment_started = True

                    await ws.send(json.dumps({"type": "text.delta", "delta": text}))

                    # Close as soon as sentence ends — don't wait for FlushSentinel
                    if text_str in [".", "!", "?"] or text_str.endswith((".", "!", "?")):
                        await ws.send(json.dumps({"type": "text.done"}))
                        segment_started = False
                        await self._collect_audio(ws, output_emitter)

                # WORD BY WORD STEAMING
                # if isinstance(text, str):
                #     text_str = text.strip()
                #     if not text_str:
                #         continue

                #     if not segment_started:
                #         segment_id = f"segment_{segment_counter}"
                #         segment_counter += 1
                #         output_emitter.start_segment(segment_id=segment_id)
                #         segment_started = True

                #     # Send the word
                #     await ws.send(json.dumps({"type": "text.delta", "delta": text}))

                #     # If this token starts with a space OR is punctuation = previous word is complete
                #     # Send text.done after every complete word
                #     next_token_starts_new_word = text.startswith(" ") or text_str in [".", "!", "?"]
                    
                #     if next_token_starts_new_word:
                #         await ws.send(json.dumps({"type": "text.done"}))
                #         segment_started = False
                #         await self._collect_audio(ws, output_emitter)
                        
                #         # Start fresh segment for next word
                #         segment_id = f"segment_{segment_counter}"
                #         segment_counter += 1
                #         output_emitter.start_segment(segment_id=segment_id)
                #         segment_started = True

                # FlushSentinel — close the text stream, collect audio
                elif "FlushSentinel" in str(type(text)):

                    if not segment_started:
                        # Nothing was buffered, skip
                        continue

                    # Tell xAI we're done sending text
                    await ws.send(json.dumps({"type": "text.done"}))
                    segment_started = False

                    # Now receive the audio
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        event_type = data.get("type", "")

                        if event_type == "audio.delta":
                            audio_bytes = base64.b64decode(data["delta"])
                            output_emitter.push(audio_bytes)

                        elif event_type == "audio.done":
                            output_emitter.end_segment()
                            output_emitter.flush()
                            break

                        elif event_type == "error":
                            print("TTS ERROR:", data)
                            break
                    
            