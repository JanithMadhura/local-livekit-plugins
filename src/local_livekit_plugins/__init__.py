"""
LiveKit Local Plugins
=====================

Custom STT and TTS plugins for running LiveKit Agents with fully local
speech processing - no cloud APIs required.

Plugins:
    FasterWhisperSTT: GPU-accelerated speech-to-text using faster-whisper
    PiperTTS: Fast local text-to-speech using Piper

Example:
    >>> from local_livekit_plugins import FasterWhisperSTT, PiperTTS
    >>>
    >>> stt = FasterWhisperSTT(model_size="medium", device="cuda")
    >>> tts = PiperTTS(model_path="/path/to/voice.onnx")

Repository: https://github.com/CoreWorxLab/local-livekit-plugins
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Corey MacPherson"

from .faster_whisper_stt import FasterWhisperSTT
from .piper_tts import PiperTTS

__all__ = ["FasterWhisperSTT", "PiperTTS", "__version__"]
