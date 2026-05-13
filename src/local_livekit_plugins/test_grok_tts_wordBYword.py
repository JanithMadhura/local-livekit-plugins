import asyncio
import base64
import json
import os
import wave
import io

import websockets

XAI_API_KEY = os.getenv("GROK_API_KEY_TTS")

async def test_word_by_word():
    ws_url = (
        "wss://api.x.ai/v1/tts"
        "?voice=eve"
        "&language=en"
        "&codec=pcm"
        "&sample_rate=24000"
        "&optimize_streaming_latency=1"
    )

    sentence = "Quantum entanglement challenges classical physics."
    words = sentence.split()

    print(f"Testing word-by-word synthesis for: '{sentence}'")
    print(f"Words: {words}")
    print("-" * 50)

    all_audio = bytearray()

    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {XAI_API_KEY}"},
    ) as ws:

        for i, word in enumerate(words):
            import time
            start = time.perf_counter()

            # Send single word
            await ws.send(json.dumps({"type": "text.delta", "delta": word}))
            await ws.send(json.dumps({"type": "text.done"}))

            # Collect audio for this word
            word_audio = bytearray()
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                event_type = data.get("type", "")

                if event_type == "audio.delta":
                    word_audio.extend(base64.b64decode(data["delta"]))

                elif event_type == "audio.done":
                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"Word {i+1}: '{word}' → {len(word_audio):,} bytes in {elapsed:.0f}ms")
                    all_audio.extend(word_audio)
                    break

                elif event_type == "error":
                    print(f"ERROR on word '{word}': {data}")
                    break

    # Save combined audio to WAV so you can listen to it
    output_path = "/home/mexxar-asus-1/local-livekit-plugins/word_by_word_test.wav"
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(24000)
        wf.writeframes(bytes(all_audio))

    print("-" * 50)
    print(f"Total audio: {len(all_audio):,} bytes")
    print(f"Saved to: {output_path}")
    print("Play with: aplay /tmp/word_by_word_test.wav")


async def test_sentence_at_once():
    """Compare against full sentence synthesis."""
    ws_url = (
        "wss://api.x.ai/v1/tts"
        "?voice=eve"
        "&language=en"
        "&codec=pcm"
        "&sample_rate=24000"
        "&optimize_streaming_latency=1"
    )

    sentence = "Quantum entanglement challenges classical physics."
    print(f"\nTesting full sentence synthesis for: '{sentence}'")
    print("-" * 50)

    import time
    start = time.perf_counter()
    all_audio = bytearray()

    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {XAI_API_KEY}"},
    ) as ws:
        await ws.send(json.dumps({"type": "text.delta", "delta": sentence}))
        await ws.send(json.dumps({"type": "text.done"}))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            event_type = data.get("type", "")

            if event_type == "audio.delta":
                all_audio.extend(base64.b64decode(data["delta"]))

            elif event_type == "audio.done":
                elapsed = (time.perf_counter() - start) * 1000
                print(f"Full sentence → {len(all_audio):,} bytes in {elapsed:.0f}ms")
                break

    output_path = "/home/mexxar-asus-1/local-livekit-plugins/sentence_test.wav"
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(bytes(all_audio))

    print(f"Saved to: {output_path}")
    print("Play with: aplay /tmp/sentence_test.wav")


async def main():
    await test_word_by_word()
    await test_sentence_at_once()


asyncio.run(main())