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

        async with websockets.connect(
            ws_url,
            additional_headers={
                "Authorization": f"Bearer {self._tts.api_key}"
            }
        ) as ws:

            buffer = ""

            async for text in self._input_ch:

                print("INPUT_CHUNK:", text)
                print("TYPE:", type(text))

                # Flush => synthesize current buffered text
                if "FlushSentinel" in str(type(text)):

                    final_text = buffer.strip()
                    buffer = ""

                    if not final_text:
                        continue

                    print("FINAL TEXT:", final_text)

                    segment_id = "segment_0"

                    output_emitter.start_segment(
                        segment_id=segment_id
                    )

                    # Send text to xAI
                    await ws.send(json.dumps({
                        "type": "text.delta",
                        "delta": final_text,
                    }))

                    await ws.send(json.dumps({
                        "type": "text.done"
                    }))

                    # Receive streamed audio for THIS text batch
                    while True:

                        print("WAITING FOR TTS...")

                        msg = await ws.recv()

                        data = json.loads(msg)

                        event_type = data.get("type", "")

                        # AUDIO CHUNK
                        if event_type == "audio.delta":

                            print("AUDIO DELTA RECEIVED")

                            audio_bytes = base64.b64decode(data["delta"])

                            pcm = np.frombuffer(audio_bytes, dtype=np.int16)

                            print("PCM SAMPLES:", len(pcm))

                            FRAME_SIZE = 240

                            for i in range(0, len(pcm), FRAME_SIZE):

                                chunk = pcm[i:i + FRAME_SIZE]

                                if len(chunk) == 0:
                                    continue

                                #print("SENDING FRAME:", len(chunk))

                                frame = rtc.AudioFrame(
                                    data=chunk.tobytes(),
                                    sample_rate=self._tts._sample_rate,
                                    num_channels=1,
                                    samples_per_channel=len(chunk),
                                )

                                self._event_ch.send_nowait(
                                    tts.SynthesizedAudio(
                                        request_id="grok_tts",
                                        segment_id="0",
                                        frame=frame,
                                        delta_text="",
                                    )
                                )

                        # END OF THIS SYNTHESIS
                        elif event_type == "audio.done":

                            print("AUDIO DONE")

                            output_emitter.end_segment()

                            output_emitter.flush()

                            break

                        # ERROR
                        elif event_type == "error":

                            print("TTS ERROR:", data)

                            break

                    continue

                # Ignore non-string objects
                if not isinstance(text, str):
                    continue

                text_str = text.strip()

                if not text_str:
                    continue

                # Better punctuation handling
                if text_str in [".", ",", "!", "?"]:
                    buffer = buffer.rstrip() + text_str + " "
                else:
                    buffer += text_str + " "
                
            