#!/usr/bin/env python3

import asyncio
import logging
import sys
import os

import numpy as np
import librosa

# add src path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

from livekit.rtc import AudioFrame
from local_livekit_plugins import FasterWhisperSTT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("test-stt")


async def main():

    # ==========================================
    # STT CONFIG
    # ==========================================

    stt = FasterWhisperSTT(
        model_size="small.en",
        device="cpu",
        compute_type="int8",
        streaming=False,
        vad_filter=False,
        max_audio_seconds=0.0,  # Process all audio without truncation
    )

    # ==========================================
    # AUDIO FILE
    # ==========================================

    wav_path = os.path.join(
        os.path.dirname(__file__),
        "test.wav"
    )

    logger.info(f"Loading: {wav_path}")

    # ==========================================
    # LOAD AUDIO
    # ==========================================

    audio_np, original_sr = librosa.load(
        wav_path,
        sr=None,
        mono=True
    )

    logger.info(
        f"Original audio: "
        f"{original_sr}Hz, "
        f"{len(audio_np)} samples"
    )

    # ==========================================
    # RESAMPLE TO 16kHz
    # ==========================================

    target_sr = 16000

    if original_sr != target_sr:

        logger.info(
            f"Resampling {original_sr}Hz -> {target_sr}Hz"
        )

        audio_np = librosa.resample(
            audio_np,
            orig_sr=original_sr,
            target_sr=target_sr
        )

    # ==========================================
    # CONVERT TO PCM INT16
    # ==========================================

    audio_int16 = (audio_np * 32767).astype(np.int16)

    audio_data = audio_int16.tobytes()

    sample_rate = target_sr
    num_channels = 1
    num_frames = len(audio_int16)

    logger.info(
        f"Final audio: "
        f"{sample_rate}Hz, "
        f"{num_channels} channel, "
        f"{num_frames} frames"
    )

    # ==========================================
    # CREATE AUDIO FRAME
    # ==========================================

    frame = AudioFrame(
        data=audio_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=num_frames,
    )

    # ==========================================
    # RUN STT
    # ==========================================

    logger.info("Running STT...")

    result = await stt.recognize(frame)

    # ==========================================
    # PRINT RESULT
    # ==========================================

    print("\n" + "=" * 60)
    print("FINAL TRANSCRIPT")
    print("=" * 60)

    if result.alternatives:
        print(result.alternatives[0].text)
    else:
        print("No transcript detected")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())