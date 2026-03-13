#!/usr/bin/env python3
"""
Benchmark script for Parakeet transcription API.

Usage:
    python benchmark.py --file experiments/meeting.m4a
    python benchmark.py --dir experiments/ --concurrent 3
    python benchmark.py --file experiments/test.mp3 --no-diarize
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}


def _build_multipart(filepath: Path, fields: dict):
    """Build a multipart/form-data request body."""
    boundary = f"----Benchmark{os.urandom(8).hex()}"
    body = b""

    # Add form fields
    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()

    # Add file
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filepath.name}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += filepath.read_bytes()
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def transcribe_file(filepath: str, base_url: str, diarize: bool = True,
                    timestamps: bool = True) -> dict:
    """Send a file to the API and return timing info."""
    filepath = Path(filepath)
    t_start = time.perf_counter()

    fields = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamps": str(timestamps).lower(),
        "diarize": str(diarize).lower(),
    }
    body, content_type = _build_multipart(filepath, fields)

    req = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            resp_data = resp.read().decode()
    except urllib.error.HTTPError as e:
        t_elapsed = time.perf_counter() - t_start
        return {
            "file": filepath.name,
            "error": f"HTTP {e.code}: {e.read().decode()[:200]}",
            "elapsed": t_elapsed,
        }

    t_elapsed = time.perf_counter() - t_start

    if status != 200:
        return {
            "file": filepath.name,
            "error": f"HTTP {status}: {resp_data[:200]}",
            "elapsed": t_elapsed,
        }

    data = json.loads(resp_data)
    segments = data.get("segments", [])
    duration = data.get("duration", 0)
    text_len = len(data.get("text", ""))
    num_segments = len(segments)

    return {
        "file": filepath.name,
        "size_mb": filepath.stat().st_size / 1024 / 1024,
        "duration_est": duration,
        "segments": num_segments,
        "text_chars": text_len,
        "elapsed": t_elapsed,
        "rtf": t_elapsed / duration if duration > 0 else None,
    }


def print_results(results: list):
    """Print a formatted results table."""
    print()
    print(f"{'File':<30} {'Size':>7} {'Segments':>8} {'Elapsed':>8} {'RTF':>8}")
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"{r['file']:<30} {'ERROR':>7} {'':<8} {r['elapsed']:>7.2f}s {r['error']}")
        else:
            rtf_str = f"{r['rtf']:.4f}" if r['rtf'] is not None else "N/A"
            print(
                f"{r['file']:<30} "
                f"{r['size_mb']:>6.1f}M "
                f"{r['segments']:>8} "
                f"{r['elapsed']:>7.2f}s "
                f"{rtf_str:>8}"
            )

    print("-" * 70)
    total_elapsed = sum(r["elapsed"] for r in results)
    print(f"{'Total':<30} {'':>7} {len(results):>8} {total_elapsed:>7.2f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Parakeet transcription API")
    parser.add_argument("--file", type=str, help="Single audio file to transcribe")
    parser.add_argument("--dir", type=str, help="Directory of audio files to transcribe")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--concurrent", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--no-diarize", action="store_true", help="Disable diarization")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each file N times")
    args = parser.parse_args()

    # Collect files
    files = []
    if args.file:
        files.append(args.file)
    elif args.dir:
        dir_path = Path(args.dir)
        files = sorted(
            str(f) for f in dir_path.iterdir()
            if f.suffix.lower() in AUDIO_EXTENSIONS
        )
    else:
        parser.error("Specify --file or --dir")

    if not files:
        print("No audio files found.")
        sys.exit(1)

    # Repeat if requested
    files = files * args.repeat

    # Health check
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        if not health.get("model_loaded"):
            print("WARNING: Model not loaded yet. Wait for startup to complete.")
            sys.exit(1)
        print(f"Server healthy — model: {health.get('model_id')}, GPU: {health.get('gpu_info')}")
        if "gpu_memory_allocated_mb" in health:
            print(f"  GPU memory: {health['gpu_memory_allocated_mb']} MB allocated, "
                  f"{health['gpu_max_memory_mb']} MB peak")
        if health.get("torch_compile_enabled"):
            print("  torch.compile() enabled")
    except (urllib.error.URLError, ConnectionError):
        print(f"Cannot connect to {args.url}. Is the server running?")
        sys.exit(1)

    print(f"\nBenchmarking {len(files)} file(s), concurrency={args.concurrent}, "
          f"diarize={not args.no_diarize}")

    diarize = not args.no_diarize
    results = []
    t_wall = time.perf_counter()

    if args.concurrent <= 1:
        # Sequential
        for f in files:
            print(f"  Transcribing {Path(f).name}...", end="", flush=True)
            r = transcribe_file(f, args.url, diarize=diarize)
            print(f" {r['elapsed']:.2f}s")
            results.append(r)
    else:
        # Concurrent
        with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            futures = {
                pool.submit(transcribe_file, f, args.url, diarize=diarize): f
                for f in files
            }
            for fut in as_completed(futures):
                r = fut.result()
                print(f"  {r['file']}: {r['elapsed']:.2f}s")
                results.append(r)

    wall_time = time.perf_counter() - t_wall
    print(f"\nWall time: {wall_time:.2f}s")

    # Sort by file name for consistent output
    results.sort(key=lambda r: r["file"])
    print_results(results)


if __name__ == "__main__":
    main()
