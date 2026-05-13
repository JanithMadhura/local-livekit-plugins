from __future__ import annotations

import io
import wave
import numpy as np
#import httpx
import websockets
import json

from livekit.agents import stt, APIConnectOptions, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr


class GrokSTT(stt.STT):

    def __init__(
        self,
        api_key: str,
        model: str = "grok-1",
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self.api_key = api_key
        self._model = model
        self.language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:

        if isinstance(buffer, list):

            all_data = []

            for frame in buffer:
                frame_data = np.frombuffer(
                    frame.data,
                    dtype=np.int16
                )

                all_data.append(frame_data)

            audio_data = np.concatenate(all_data)

            sample_rate = buffer[0].sample_rate

        else:

            audio_data = np.frombuffer(
                buffer.data,
                dtype=np.int16
            )

            sample_rate = buffer.sample_rate

        audio_bytes = audio_data.tobytes()

        # Convert to WAV format in memory for HTTP REST API only
        #wav_io = io.BytesIO()

        #with wave.open(wav_io, "wb") as wav_file:

        #    wav_file.setnchannels(1)

        #    wav_file.setsampwidth(2)

        #    wav_file.setframerate(sample_rate)

        #    wav_file.writeframes(audio_data.tobytes())

        #wav_io.seek(0)

        #==============================================================================================================================================================
        # for HTTP rest API
        '''
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        files = {
            "file": ("audio.wav", wav_io, "audio/wav"),
        }

        data = {
            "format": "true",
            "language": language if language is not NOT_GIVEN else self.language,
        }

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.post(
                "https://api.x.ai/v1/stt",
                headers=headers,
                files=files,
                data=data,
            )

            print("STATUS:", response.status_code)

            result = response.json()

            print("RESPONSE:", result)

            # Handle API errors safely
            if response.status_code != 200:

                error_msg = result.get("error", "Unknown STT error")

                print(f"Grok STT Error: {error_msg}")

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text="",
                            start_time=0,
                            end_time=0,
                            language=self.language,
                        )
                    ],
                )

            text = result.get("text", "").strip()

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        start_time=0,
                        end_time=0,
                        language=self.language,
                    )
                ],
            )

        '''
        #==============================================================================================================================================================
        # for WebSocket API
        lang = (
            language
            if language is not NOT_GIVEN
            else self.language
        )

        ws_url = (
            f"wss://api.x.ai/v1/stt"
            f"?language={lang}"
            f"&format=true"
        )

        async with websockets.connect(
            ws_url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}"
            }
        ) as ws:
        
            await ws.send(audio_bytes)

            await ws.send(json.dumps({
                "type": "audio.done"
            }))

            text = ""

            while True:

                response = await ws.recv()

                result = json.loads(response)

                print("WebSocket Response:", result)

                event_type = result.get("type", "")

                # transcript event
                if event_type == "transcript.partial":

                    print("PARTIAL:", result)

                    # xAI marks final transcript here
                    if result.get("is_final", False):

                        text = result.get("text", "").strip()

                        break

            text = result.get("text", "").strip()

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        start_time=0,
                        end_time=0,
                        language=self.language,
                    )
                ],
            )

