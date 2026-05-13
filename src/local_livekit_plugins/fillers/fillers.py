from piper.voice import PiperVoice
from piper.config import SynthesisConfig

import wave

voice = PiperVoice.load(
    "/home/mexxar-asus-1/local-livekit-plugins/models/piper/en_US-ryan-high.onnx",
    use_cuda=False
)

fillers = [
    "...Okay, give me a moment to think.",
]

for text in fillers:

    filename = text.lower() + ".wav"

    syn_config = SynthesisConfig(
        length_scale=1.25,
        noise_scale=0.667,
        noise_w_scale=0.8,
    )

    with wave.open(filename, "wb") as wav_file:

        voice.synthesize_wav(
            text,
            wav_file,
            syn_config=syn_config,
            set_wav_format=True,
        )

    print("Generated:", filename)