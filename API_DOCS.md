# Parakeet Transcription API — Client Integration Guide

Base URL: `http://<host>:8000`

---

## Two ways to transcribe

| | **Sync** (`/v1/audio/transcriptions`) | **Async** (`/v1/jobs`) |
|---|---|---|
| **How it works** | Upload file, wait, get result in same request | Upload file, get job ID, poll for result |
| **Best for** | Short recordings (<5 min), when you can wait | Long meetings, batch uploads, background processing |
| **Response time** | Seconds to minutes (blocks until done) | Instant (202-style), results ready later |
| **Includes** | Transcription + diarization + optional LLM analysis | Same |

For **live captions during a meeting**, use `POST /transcribe` (raw audio stream, no diarization — see "Real-Time Streaming" section at the bottom).

---

## Option A: Synchronous transcription

Send the recording and wait for the result in one request. Simple, but the connection stays open until processing finishes.

### Request

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | Yes | — | Audio file (mp3, m4a, wav, webm, ogg, flac — anything ffmpeg supports) |
| `diarize` | bool | No | `true` | Identify who said what (speaker labels) |
| `analyze` | bool | No | `false` | Run LLM analysis (summary, action items, decisions) |
| `timestamps` | bool | No | `false` | Include per-segment timestamps in response |
| `include_diarization_in_text` | bool | No | `true` | Prepend "Speaker N:" labels in the text output |
| `response_format` | string | No | `json` | `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `language` | string | No | auto | Language hint (e.g., `en`) |
| `model` | string | No | `whisper-1` | Ignored (compatibility with OpenAI API) |

### Example

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.m4a \
  -F diarize=true \
  -F analyze=true \
  -F timestamps=true
```

### Response (JSON)

```json
{
  "text": "Speaker 1: Let's review the migration timeline. 2: I can have the backups ready by Friday...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.52,
      "text": "Speaker 1: Let's review the migration timeline.",
      "speaker": "speaker_SPEAKER_00"
    },
    {
      "id": 1,
      "start": 3.52,
      "end": 7.84,
      "text": "2: I can have the backups ready by Friday.",
      "speaker": "speaker_SPEAKER_01"
    }
  ],
  "language": null,
  "task": "transcribe",
  "duration": 12.3,
  "model": "parakeet-tdt-0.6b-v2",
  "meeting_intelligence": {
    "summary": "Team reviewed migration timeline. Backups to be completed by Friday.",
    "action_items": [
      {"assignee": "Speaker 2", "task": "Complete backups", "deadline": "Friday"}
    ],
    "decisions": [
      {"decision": "Proceed with migration", "context": "Timeline reviewed and agreed"}
    ],
    "unresolved_questions": [],
    "key_topics": ["migration", "backups"],
    "participants": [
      {"speaker_label": "Speaker 1", "role": null},
      {"speaker_label": "Speaker 2", "role": null}
    ],
    "follow_ups": [],
    "risks_and_blockers": []
  }
}
```

`meeting_intelligence` is only present when `analyze=true`.
`segments` is only present when `timestamps=true` or `response_format=verbose_json`.

---

## Option B: Async job queue (recommended for meetings)

Submit the recording, get a job ID instantly, and poll (or receive a webhook) when it's done. The connection doesn't need to stay open.

**Requires `ENABLE_JOB_QUEUE=true` on the server.**

### Step 1: Submit a job

```
POST /v1/jobs
Content-Type: multipart/form-data
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | Yes | — | Audio file |
| `diarize` | bool | No | `true` | Speaker diarization |
| `analyze` | bool | No | `false` | LLM meeting analysis |
| `include_diarization_in_text` | bool | No | `true` | Speaker labels in text |
| `response_format` | string | No | `json` | Output format |
| `language` | string | No | auto | Language hint |
| `webhook_url` | string | No | — | URL to POST results to when done |

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -F file=@meeting.m4a \
  -F diarize=true \
  -F analyze=true
```

**Response** (immediate):

```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "queued",
  "created_at": 1711382400.123
}
```

### Step 2: Poll for status

```
GET /v1/jobs/{job_id}
```

```bash
curl http://localhost:8000/v1/jobs/a1b2c3d4e5f6
```

**Response:**

```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "processing",
  "created_at": 1711382400.123,
  "updated_at": 1711382405.456,
  "started_at": 1711382405.456,
  "completed_at": null,
  "original_filename": "meeting.m4a",
  "error_message": null,
  "duration_seconds": null
}
```

Status values: `queued` → `processing` → `completed` or `failed`

**Recommended polling interval:** Every 5 seconds. A 23-minute meeting typically takes 15-20 seconds to process.

### Step 3: Get the result

Once `status` is `completed`:

```
GET /v1/jobs/{job_id}/result
```

```bash
curl http://localhost:8000/v1/jobs/a1b2c3d4e5f6/result
```

**Response:**

```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "completed",
  "result": {
    "text": "Speaker 1: Let's review the migration timeline...",
    "segments": [...],
    "meeting_intelligence": {...}
  }
}
```

The `result` object is identical to what `/v1/audio/transcriptions` returns.

Returns `409` if the job isn't completed yet.

### Alternative: Webhook instead of polling

If you provide `webhook_url` when creating the job, the server will POST the result to that URL when processing finishes:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -F file=@meeting.m4a \
  -F analyze=true \
  -F webhook_url=https://your-server.com/transcription-callback
```

**Webhook payload** (POST to your URL):

```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "completed",
  "result": {
    "text": "...",
    "segments": [...],
    "meeting_intelligence": {...}
  }
}
```

The webhook is fire-and-forget (no retries). Use polling as a fallback.

### Other job endpoints

**List all jobs:**
```
GET /v1/jobs?limit=50&offset=0
```

**Delete a job** (and its audio/result files):
```
DELETE /v1/jobs/{job_id}
```
Returns `409` if the job is currently processing.

---

## Chrome extension integration flow

```
┌─────────────────────────────────────────────────────┐
│  During meeting (live captions)                     │
│                                                     │
│  Extension captures audio chunks (2-30s)            │
│       ↓                                             │
│  POST /transcribe  (raw Float32, 16kHz mono)        │
│       ↓                                             │
│  {"text": "..."} displayed as live caption          │
│  (no diarization — speaker detected via DOM)        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  After meeting (full transcript)                    │
│                                                     │
│  Extension has the full recording (webm/m4a)        │
│       ↓                                             │
│  POST /v1/jobs  with file + diarize + analyze       │
│       ↓                                             │
│  Get job_id back immediately                        │
│       ↓                                             │
│  Poll GET /v1/jobs/{id} every 5s                    │
│  (or wait for webhook)                              │
│       ↓                                             │
│  GET /v1/jobs/{id}/result                           │
│       ↓                                             │
│  Display transcript with speaker labels,            │
│  action items, summary, etc.                        │
└─────────────────────────────────────────────────────┘
```

### JavaScript example (extension background script)

```javascript
// After meeting ends, upload the recording
async function submitRecording(audioBlob, filename) {
  const form = new FormData();
  form.append('file', audioBlob, filename);
  form.append('diarize', 'true');
  form.append('analyze', 'true');

  const res = await fetch('http://localhost:8000/v1/jobs', {
    method: 'POST',
    body: form,
  });
  const { job_id } = await res.json();
  return job_id;
}

// Poll until done
async function waitForResult(jobId) {
  while (true) {
    const res = await fetch(`http://localhost:8000/v1/jobs/${jobId}`);
    const status = await res.json();

    if (status.status === 'completed') {
      const result = await fetch(`http://localhost:8000/v1/jobs/${jobId}/result`);
      return await result.json();
    }

    if (status.status === 'failed') {
      throw new Error(status.error_message || 'Transcription failed');
    }

    // Wait 5 seconds before next poll
    await new Promise(r => setTimeout(r, 5000));
  }
}

// Usage
const jobId = await submitRecording(blob, 'meeting.webm');
const { result } = await waitForResult(jobId);
console.log(result.text);
console.log(result.meeting_intelligence.summary);
console.log(result.meeting_intelligence.action_items);
```

---

## Real-time streaming (live captions)

For live transcription during a meeting. The extension captures audio as raw Float32Array at 16kHz mono and sends chunks every few seconds.

```
POST /transcribe
Content-Type: application/octet-stream
Body: raw bytes (Float32Array, 16kHz mono, 2-30 seconds)
```

**Response:**
```json
{"text": "the transcribed text for this chunk"}
```

No diarization, no file conversion. Speaker identification is handled by the extension (via DOM observation in Google Meet).

If the server is overloaded (>20 requests queued), it returns `503`. The extension should skip that chunk and continue.

---

## Health check

```
GET /health
```

Use this to check if the server is ready before sending requests:

```json
{
  "status": "ok",
  "model_loaded": true,
  "diarizer_loaded": true,
  "llm_available": true,
  "transcribe_queue": {
    "depth": 0,
    "max": 20,
    "total_served": 142,
    "total_rejected": 0
  }
}
```

Wait for `model_loaded: true` before sending transcription requests (takes 2-3 minutes after server start).

---

## Error codes

| Code | Meaning |
|------|---------|
| 400 | Bad request (audio too long, invalid format) |
| 404 | Job not found |
| 409 | Job is still processing (can't get result yet / can't delete) |
| 503 | Server not ready (model loading), queue full, or job queue not enabled |
| 504 | Transcription timed out |
| 500 | Internal error |
