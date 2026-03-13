# Parakeet Whisper-Compatible API

A simple FastAPI server that provides an OpenAI Whisper API-compatible endpoint backed by [NVIDIA's Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for speech recognition + [Pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Features

- Complete drop-in replacement for OpenAI's Whisper API
- Uses [NVIDIA's Parakeet-TDT 0.6B V2 model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for high-quality transcription
- Supports all Whisper API response formats (json, text, srt, vtt, verbose_json)
- Supports word-level and segment-level timestamps
- Optional speaker diarization using [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- FastAPI-based server with automatic OpenAPI documentation

## Requirements

- NVIDIA GPU with CUDA support (recommended)
- HuggingFace account and access token (required for speaker diarization)

---

## Quick Start (Docker) — Recommended

Docker is the easiest way to get up and running. Everything is pre-configured — no Python, CUDA, or ffmpeg installation required on your host machine.

### Prerequisites

#### 1. Install Docker Desktop

Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/). During setup:
- Enable the **WSL 2 backend** when prompted
- After installation, open Docker Desktop and confirm it's running (whale icon in system tray)

#### 2. Install NVIDIA Container Toolkit

This lets Docker access your GPU. Open a **WSL terminal** (or Ubuntu from the Start menu) and run:

```bash
# Add the NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Then **restart Docker Desktop**.

#### 3. Verify GPU access in Docker

Run this to confirm Docker can see your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

You should see your GPU listed (e.g. "NVIDIA GeForce RTX 4090"). If this fails, check that:
- Docker Desktop is running with the WSL 2 backend
- Your NVIDIA drivers are up to date
- You restarted Docker Desktop after installing the container toolkit

#### 4. Set up HuggingFace (required for speaker diarization)

1. Create a free account at [huggingface.co](https://huggingface.co/)
2. Generate an access token at [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
3. **Accept the model licenses** — you must visit **both** links below and click "Agree":
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> **Important:** Without accepting both licenses, diarization will fail on startup even with a valid token. You'll see a `401` or `Access to model is restricted` error in the logs.

### Setup

#### 5. Clone the repository

```bash
git clone https://github.com/jfgonsalves/parakeet-diarized
cd parakeet-diarized
```

#### 6. Create your `.env` file

Create a file called `.env` in the project root with the following contents. Replace `hf_YourTokenHere` with your actual HuggingFace token:

```env
# Required
HUGGINGFACE_ACCESS_TOKEN=hf_YourTokenHere

# GPU settings (defaults work well for RTX 3090/4090)
BATCH_SIZE=32
CHUNK_DURATION=30
MAX_CONCURRENT_REQUESTS=1
MAX_CONCURRENT_DIARIZE=1

# Diarization tuning (good defaults, no need to change)
DIARIZE_SEGMENTATION_BATCH_SIZE=8
DIARIZE_EMBEDDING_BATCH_SIZE=8
DIARIZE_SEGMENTATION_STEP=0.3

# Leave these as-is
TORCH_COMPILE=false
ENABLE_DIARIZATION=true
INCLUDE_DIARIZATION_IN_TEXT=true
REQUEST_TIMEOUT=300
```

#### 7. Build and start

```bash
docker compose up --build
```

**What to expect on first run:**

| Stage | Time | What's happening |
|-------|------|-----------------|
| Docker build | 15-30 min | Downloading CUDA base image, compiling Python packages |
| Model download | 2-3 min | Downloading Parakeet (~1.5GB) and pyannote (~500MB) models from HuggingFace |
| Model loading | 1-2 min | Loading models into GPU memory |

You'll see `API server ready.` followed by `Starting Gradio frontend on port 8001...` when everything is loaded.

**Subsequent starts** are much faster — the Docker image is cached and models are stored in Docker volumes.

#### 8. Use it

| Service | URL |
|---------|-----|
| **Gradio UI** (upload & transcribe) | http://localhost:8001 |
| **API server** | http://localhost:8000 |
| **Health check** | http://localhost:8000/health |
| **API docs** (Swagger) | http://localhost:8000/docs |

**Test with curl:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@/path/to/your/audio.m4a \
  -F model=whisper-1 \
  -F timestamps=true \
  -F diarize=true
```

Or just open http://localhost:8001 in your browser for the Gradio UI.

### Docker Commands Reference

```bash
# Start (foreground — see logs in terminal)
docker compose up

# Start (background — runs silently)
docker compose up -d

# View logs when running in background
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build

# Full reset (removes cached models — will re-download on next start)
docker compose down -v
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found in container | Install NVIDIA Container Toolkit (step 2) and restart Docker Desktop |
| `401 Unauthorized` during model download | Check your `HUGGINGFACE_ACCESS_TOKEN` in `.env` |
| `Access to model is restricted` | Accept **both** pyannote model licenses on HuggingFace (step 4) |
| Build fails on `pip install torch` | Ensure Docker has internet access; try `docker compose build --no-cache` |
| Out of GPU memory | Reduce `BATCH_SIZE` to 16 or 8 in `.env` |
| Container exits immediately | Run `docker compose logs` to see the error |
| Gradio UI not loading | Wait for `API server ready.` in logs — models take a few minutes to load |

---

## Manual Installation (without Docker)

If you prefer to run without Docker (e.g. directly in WSL):

1. Clone this repository:
   ```bash
   git clone https://github.com/jfgonsalves/parakeet-diarized
   cd parakeet-diarized
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up speaker diarization (optional):
   - Create a free account at [HuggingFace](https://huggingface.co/)
   - Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Accept the user agreement for the [Pyannote speaker diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1)

5. Run the server:

   **With speaker diarization:**
   ```bash
   ./run.sh --hf-token "your_token_here"
   ```

   **Without speaker diarization:**
   ```bash
   ./run.sh
   ```

   **Other options:**
   ```bash
   ./run.sh --help  # See all available options
   ./run.sh --port 8080 --debug --hf-token "your_token_here"
   ```

## Usage

### API Endpoints

The API mimics the OpenAI Whisper API interface:

#### Transcribe Audio

```
POST /v1/audio/transcriptions
```

Parameters:
- `file`: The audio file to transcribe (multipart/form-data)
- `model`: Model to use (defaults to "whisper-1", but will use Parakeet regardless)
- `language`: Language of the audio (optional)
- `response_format`: Format of the response (defaults to "json", options: json, text, srt, vtt, verbose_json)
- `timestamps`: Whether to include timestamps (defaults to false)
- `timestamp_granularities`: Timestamp detail level (accepts "segment")
- `temperature`: Temperature for sampling (defaults to 0.0)
- `vad_filter`: Voice activity detection filter (defaults to false)
- `prompt`: Optional prompt to guide the transcription (ignored but accepted for compatibility)
- `diarize`: Enable speaker diarization (defaults to true, requires HuggingFace token)
- `include_diarization_in_text`: Include speaker labels in transcript text (defaults to true)

Example with curl:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F model=whisper-1 \
  -F timestamps=true \
  -F diarize=true
```

#### Health Check

```
GET /health
```

Returns the health status of the API and the loaded model.

## Compatibility with OpenAI Whisper API

This API is designed to be a drop-in replacement for the OpenAI Whisper API:

1. Supports all Whisper API response formats (json, text, srt, vtt, verbose_json)
2. Accepts all major Whisper API parameters for compatibility
3. Returns responses in the same format as the OpenAI Whisper API
4. Provides a `/v1/models` endpoint for application compatibility

Minor differences:
1. The `model` parameter is accepted but ignored - always uses Parakeet-TDT
2. Some advanced Whisper-specific parameters might have no effect
3. Performance characteristics may differ from OpenAI's implementation

## API Response Formats

The API supports multiple response formats:

### JSON (default)
```json
{
  "text": "Full transcription text goes here"
}
```

### Verbose JSON
```json
{
  "text": "Full transcription text goes here",
  "task": "transcribe",
  "language": "en",
  "duration": 10.5,
  "model": "parakeet-tdt-0.6b-v2",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Segment text",
      "tokens": [50364, 2425, 286, 257],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.1
    },
    {
      "id": 1,
      "start": 2.5,
      "end": 5.0,
      "text": "Another segment",
      "tokens": [50364, 5816, 2121],
      "temperature": 0.0,
      "avg_logprob": -0.6,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.05
    }
  ]
}
```

### Plain Text
```
Full transcription text goes here
```

### SRT
```
1
00:00:00,000 --> 00:00:02,500
Segment text

2
00:00:02,500 --> 00:00:05,000
Another segment
```

### VTT
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Segment text

00:00:02.500 --> 00:00:05.000
Another segment
```

The `segments` field is included when the `timestamps` parameter is set to `true` or when using `verbose_json` format.

## Speaker Diarization

The API includes speaker diarization capabilities using [Pyannote.audio](https://github.com/pyannote/pyannote-audio):

### Setup Requirements

For speaker diarization to work, you need:

1. **HuggingFace Account**: Create a free account at [huggingface.co](https://huggingface.co/)
2. **Access Token**: Generate a token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
3. **Model Agreement**: Accept the user agreement for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. **Environment Variable**: Set `HUGGINGFACE_ACCESS_TOKEN` with your token

### Features

- Automatic speaker detection and labeling
- Integration with transcription segments
- Optional speaker labels in transcript text
- Support for multiple speakers per audio file

### Usage

Enable diarization by setting `diarize=true` in your API request:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F diarize=true \
  -F include_diarization_in_text=true
```

When `include_diarization_in_text=true`, the transcript will include speaker labels:
```
Speaker 1: Hello, how are you today?
Speaker 2: I'm doing well, thank you for asking.
```

### Configuration

Use the `run.sh` script to configure and start the server:

```bash
./run.sh --help
# Options:
#   --debug             Enable debug mode
#   --port PORT         Set server port (default: 8000)
#   --host HOST         Set server host (default: 0.0.0.0)
#   --skip-deps-check   Skip dependency checking
#   --hf-token TOKEN    Set HuggingFace access token for speaker diarization
#   --help              Show help message
```

**Environment Variables** (for settings not available as command line arguments):
- `ENABLE_DIARIZATION`: Enable/disable diarization globally (default: true)
- `INCLUDE_DIARIZATION_IN_TEXT`: Include speaker labels in text by default (default: true)
- `MODEL_ID`: Parakeet model to use (default: nvidia/parakeet-tdt-0.6b-v2)
- `TEMPERATURE`: Sampling temperature (default: 0.0)
- `CHUNK_DURATION`: Audio chunk duration in seconds (default: 500)
- `TEMP_DIR`: Temporary directory for audio processing (default: /tmp/parakeet)

## Performance

The [NVIDIA Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) offers:
- Fast transcription (top model on the HF Open ASR leaderboard)
- Support for punctuation and capitalization
- High accuracy with word error rates as low as 1.69% on LibriSpeech test-clean

[Pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization adds:
- Automatic speaker identification using state-of-the-art models
- Real-time speaker change detection
- Support for unlimited number of speakers

## Acknowledgments

This project builds upon excellent work by:

- **NVIDIA NeMo Team**: For the outstanding [Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) that provides state-of-the-art speech recognition
- **Pyannote Team**: For the powerful [Pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization toolkit

## License

This project is released under MIT License. However, the Parakeet-TDT model is governed by the CC-BY-4.0 license.
