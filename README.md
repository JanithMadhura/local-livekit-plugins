# Local LiveKit Plugins

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents-purple.svg)](https://docs.livekit.io/agents/)

**Run LiveKit voice agents with fully local STT and TTS - no cloud APIs required.**

Custom plugins for [LiveKit Agents](https://docs.livekit.io/agents/) that enable completely local speech processing using [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) for STT and [Piper](https://github.com/rhasspy/piper) for TTS.

## Tested On

- **OS:** Linux (Arch 6.17.7)
- **GPU:** NVIDIA RTX 3060 (12GB VRAM)
- **CUDA:** 12.x
- **Python:** 3.10+

**Windows/Mac users:** Not yet tested. Community contributions welcome! Please report issues on GitHub.

## Why Local?

| | Cloud | Local |
|---|---|---|
| **Quality** | Better | Good |
| **Latency** | ~2.1s total | ~2.8s total |
| **Cost** | ~$150/year* | ~$20/year |
| **Privacy** | Data sent externally | Stays on your network |
| **Control** | Vendor dependent | Full ownership |

*Based on 100 hours/year: Deepgram Nova-2 ($0.35/hr) + Cartesia Sonic ($50/1M chars).

## Features

- **FasterWhisperSTT** - GPU-accelerated speech-to-text
  - Multiple model sizes (tiny → large-v3)
  - ~230-460ms processing time on RTX 3060
  - Configurable beam search and VAD

- **PiperTTS** - Fast local text-to-speech
  - Multiple voice models available
  - ~9ms per character (~300ms for short responses)
  - Configurable speed, volume, pitch

## Quick Start

### Prerequisites

**Required:**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Python 3.10+
- Docker (for LiveKit server)
- [Ollama](https://ollama.ai/) (for local LLM) - Must be running: `ollama serve`
- ffmpeg or libavcodec (for audio processing)

**For GPU Acceleration (recommended):**
- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended for larger Whisper models)
- NVIDIA drivers with CUDA 11.8+ support
- Note: PyTorch (~2GB download) includes bundled CUDA libraries

### 1. Clone and Install

```bash
git clone https://github.com/CoreWorxLab/local-livekit-plugins.git
cd local-livekit-plugins

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[examples]"
```

### 2. Download a Piper Voice Model

```bash
mkdir -p models/piper && cd models/piper

# Download Ryan (male US English, high quality)
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json
```

More voices: [Piper Voices on HuggingFace](https://huggingface.co/rhasspy/piper-voices/tree/main)

### 3. Start LiveKit Server

```bash
docker compose up -d

# Verify it's running
curl http://localhost:7880
```

### 4. Configure and Run

```bash
# Setup environment file
cp examples/.env.local.example examples/.env.local
# Edit examples/.env.local with your model paths

# Run with local pipeline (console mode for testing)
uv run examples/voice_agent.py console

# Or run with cloud pipeline (requires API keys)
USE_LOCAL=false uv run examples/voice_agent.py console

# For dev mode (connects to LiveKit playground)
uv run examples/voice_agent.py dev
```

## 🐳 Docker - Run on Any Computer

Get the entire stack (LiveKit, LLM, Voice Agent) running with one command:

```bash
# Pull all images and start services
docker compose -f docker-compose.prod.yaml up -d

# Check service status
docker compose -f docker-compose.prod.yaml ps

# View logs
docker compose -f docker-compose.prod.yaml logs -f voice-agent

# Stop all services
docker compose -f docker-compose.prod.yaml down
```

### Docker Setup Details

**What's included:**
- **livekit-server**: WebRTC media server (localhost:7880)
- **ollama**: Local LLM inference (localhost:11434)
- **voice-agent**: Voice AI with streaming STT + TTS

**First run:**
The image builds automatically. First pull takes 5-10 minutes (Python, dependencies). On first run, Ollama will download the LLM model (~7GB for neural-chat, ~4GB for tinyllama).

```bash
# Pre-download model to avoid delays:
docker compose -f docker-compose.prod.yaml exec ollama ollama pull neural-chat

# Then start the agent
docker compose -f docker-compose.prod.yaml restart voice-agent
```

**GPU Support:**
Uncomment the `CUDA_VISIBLE_DEVICES` line in `docker-compose.prod.yaml` to enable GPU:
```yaml
environment:
  CUDA_VISIBLE_DEVICES: "0"  # GPU device ID
```

**Custom Configuration:**
Edit environment variables in `docker-compose.prod.yaml`:
```yaml
voice-agent:
  environment:
    WHISPER_MODEL: "base"          # Model size: tiny, base, small, medium
    WHISPER_DEVICE: "cuda"         # cuda or cpu
    PIPER_SPEED: "1.5"             # Speech rate multiplier
    OLLAMA_MODEL: "neural-chat"    # LLM model
```

---
## ⚡ Low-Latency Voice Agents

Getting **3x faster responses** with streaming TTS? Read [LATENCY_OPTIMIZATION.md](LATENCY_OPTIMIZATION.md)

If you're seeing 6-7 second round-trip latency, use the optimized agent:

```bash
# Standard latency: 6-7 seconds
uv run examples/voice_agent.py console

# Optimized latency: 1.5-3 seconds (60% improvement!)
uv run examples/voice_agent_optimized.py console
```

**Optimizations included:**
- **Streaming TTS**: Audio plays while still generating (200-400ms first audio)
- **Pipelined LLM**: Start TTS while LLM is outputting
- **Faster models**: Smaller Whisper model, faster LLM configuration
- **Speed boost**: 1.5x speech rate (sounds responsive, still clear)

See [LATENCY_OPTIMIZATION.md](LATENCY_OPTIMIZATION.md) for detailed configuration and benchmarking.

---

## Using the Plugins in Your Own Project

### Install from GitHub

```bash
uv add git+https://github.com/CoreWorxLab/local-livekit-plugins.git
```

### Use in Your Agent

```python
from local_livekit_plugins import FasterWhisperSTT, PiperTTS, PiperTTSStreaming
from livekit.agents import AgentSession
from livekit.plugins import silero, openai as lk_openai

# Create local STT
stt = FasterWhisperSTT(
    model_size="medium",      # tiny, base, small, medium, large-v3
    device="cuda",            # cuda or cpu
    compute_type="float16",   # float16, int8
)

# Create local TTS - Standard (buffered)
tts = PiperTTS(
    model_path="/path/to/en_US-ryan-high.onnx",
    use_cuda=False,           # CPU recommended for compatibility
    speed=1.0,
)

# OR - Create streaming TTS for low-latency (RECOMMENDED)
tts = PiperTTSStreaming(
    model_path="/path/to/en_US-ryan-high.onnx",
    use_cuda=False,
    speed=1.5,                # 50% faster speech (recommended)
    streaming=True,           # Enable streaming
)

# Create session with local LLM (Ollama)
session = AgentSession(
    stt=stt,
    llm=lk_openai.LLM.with_ollama(model="llama3.1:8b"),
    tts=tts,
    vad=silero.VAD.load(),
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                       │
├─────────────────────────────────────────────────────────────┤
│                    LiveKit Agents SDK                       │
├───────────────────┬──────────────────┬──────────────────────┤
│  FasterWhisperSTT │      LLM        │      PiperTTS         │
│  (this package)   │   (Ollama)      │   (this package)      │
├───────────────────┼─────────────────┼───────────────────────┤
│  faster-whisper   │   ollama        │    piper-tts          │
│     + CUDA        │                 │   + onnxruntime       │
└───────────────────┴─────────────────┴───────────────────────┘
```

## Configuration Reference

### FasterWhisperSTT

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | "medium" | Model size: tiny, base, small, medium, large-v3 |
| `device` | str | "cuda" | Processing device: cuda, cpu, auto |
| `compute_type` | str | "float16" | Quantization: float16, int8, int8_float16 |
| `language` | str | "en" | Language hint for recognition |
| `beam_size` | int | 5 | Beam search width (1-10) |
| `vad_filter` | bool | True | Enable voice activity detection |

### PiperTTS

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to .onnx voice model |
| `use_cuda` | bool | False | Enable GPU (has CUDA version constraints) |
| `speed` | float | 1.0 | Speech rate (0.5-2.0) |
| `volume` | float | 1.0 | Volume level |
| `noise_scale` | float | 0.667 | Phoneme variation |
| `noise_w` | float | 0.8 | Phoneme width variation |

### PiperTTSStreaming (Low-Latency)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to .onnx voice model |
| `use_cuda` | bool | False | Enable GPU (has CUDA version constraints) |
| `speed` | float | 1.0 | Speech rate (1.5 recommended for low-latency) |
| `streaming` | bool | True | Enable streaming mode (for real-time synthesis) |
| `volume` | float | 1.0 | Volume level |
| `noise_scale` | float | 0.667 | Phoneme variation |
| `noise_w` | float | 0.8 | Phoneme width variation |

## Development

```bash
# Clone the repo
git clone https://github.com/CoreWorxLab/local-livekit-plugins.git
cd local-livekit-plugins

# Install with dev dependencies
uv sync --all-extras

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest
```

**Note:** End-to-end latency includes VAD buffering. Local STT uses batch processing (waits for speech to end), while cloud STT streams in real-time.

## Known Issues

### Installation & Platform

1. **Large Download Size**: PyTorch with CUDA support is ~2GB. First install may take a while depending on your connection.

2. **Windows/Mac Untested**: This has only been tested on Linux. You may encounter:
   - Path handling issues (especially Windows)
   - Platform-specific audio library requirements
   - Different PyTorch wheel availability
   - **Help wanted!** If you get it working, please share your setup in a GitHub issue or PR.

3. **Whisper Model Downloads**: Models download automatically on first run (1-5GB depending on size). This happens silently - check logs if first startup seems slow.

### Technical Limitations

4. **Piper GPU Support**: Piper has CUDA version constraints. If you're on CUDA 12+, you may need to:
   - Use CPU mode (`use_cuda=False`) - often faster anyway for short utterances
   - Containerize with a compatible CUDA version
   - See: [Piper CUDA 12 Discussion](https://github.com/rhasspy/piper/discussions/544)


**Built with [Claude Code](https://claude.ai/claude-code)**
