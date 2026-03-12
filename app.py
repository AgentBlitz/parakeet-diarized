import os
import tempfile
import zipfile
from datetime import datetime

import gradio as gr
import pandas as pd
import requests

API_URL = "http://localhost:8000/v1/audio/transcriptions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def _build_name_map(speaker_df: pd.DataFrame) -> dict:
    name_map = {}
    for _, row in speaker_df.iterrows():
        label = str(row.iloc[0])
        name = str(row.iloc[1]).strip()
        name_map[label] = name if name else label
    return name_map


def _group_segments(segments: list, name_map: dict) -> list:
    """Merge consecutive segments from the same (resolved) speaker."""
    groups = []
    for seg in segments:
        raw_speaker = seg.get("speaker") or "UNKNOWN"
        name = name_map.get(raw_speaker, raw_speaker)
        text = seg.get("text", "").strip()
        if not text:
            continue
        if groups and groups[-1]["name"] == name:
            groups[-1]["end"] = seg.get("end", groups[-1]["end"])
            groups[-1]["text"] += " " + text
        else:
            groups.append({
                "name": name,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            })
    return groups


def build_markdown(segments: list, speaker_df: pd.DataFrame) -> str:
    if not segments:
        return ""

    name_map = _build_name_map(speaker_df)
    groups = _group_segments(segments, name_map)

    duration = segments[-1].get("end", 0.0) if segments else 0.0
    unique_names = list(dict.fromkeys(g["name"] for g in groups))
    n_speakers = len(unique_names)

    date_str = datetime.now().strftime("%Y-%m-%d")
    header = (
        f"# Meeting Transcript\n"
        f"_{date_str} · {_fmt_duration(duration)} · "
        f"{n_speakers} speaker{'s' if n_speakers != 1 else ''}_\n\n---\n\n"
    )

    body = ""
    for g in groups:
        time_range = f"`{_fmt_time(g['start'])} – {_fmt_time(g['end'])}`"
        body += f"**{g['name']}** {time_range}  \n{g['text']}\n\n"

    return header + body


# ---------------------------------------------------------------------------
# Single-file functions
# ---------------------------------------------------------------------------

def transcribe(audio_path: str):
    if audio_path is None:
        return (
            [],
            pd.DataFrame(columns=["Detected Label", "Name"]),
            gr.update(visible=False),
            "Upload an audio file first.",
            "",
        )

    status = "Transcribing… this may take a minute."
    yield [], pd.DataFrame(columns=["Detected Label", "Name"]), gr.update(visible=False), status, ""

    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                API_URL,
                files={"file": f},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",
                    "timestamps": "true",
                    "diarize": "true",
                    "include_diarization_in_text": "false",
                },
                timeout=600,
            )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        yield (
            [],
            pd.DataFrame(columns=["Detected Label", "Name"]),
            gr.update(visible=False),
            "Could not reach the server. Start it with `.\\start.ps1` first.",
            "",
        )
        return
    except Exception as e:
        yield (
            [],
            pd.DataFrame(columns=["Detected Label", "Name"]),
            gr.update(visible=False),
            f"Error: {e}",
            "",
        )
        return

    segments = data.get("segments", [])
    if not segments:
        yield (
            [],
            pd.DataFrame(columns=["Detected Label", "Name"]),
            gr.update(visible=False),
            "No speech detected.",
            "",
        )
        return

    # Build ordered unique speaker list
    seen = []
    for seg in segments:
        sp = seg.get("speaker") or "UNKNOWN"
        if sp not in seen:
            seen.append(sp)

    speaker_df = pd.DataFrame({"Detected Label": seen, "Name": [""] * len(seen)})
    preview = build_markdown(segments, speaker_df)

    yield (
        segments,
        speaker_df,
        gr.update(visible=True),
        f"Done — {len(segments)} segments, {len(seen)} speaker(s) detected.",
        preview,
    )


def update_preview(segments: list, speaker_df: pd.DataFrame) -> str:
    return build_markdown(segments, speaker_df)


def export_markdown(segments: list, speaker_df: pd.DataFrame):
    md = build_markdown(segments, speaker_df)
    if not md:
        return None
    tmp = tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(md)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Batch functions
# ---------------------------------------------------------------------------

def _call_api(file_path: str) -> dict:
    """Call the transcription API for a single file. Returns parsed JSON."""
    with open(file_path, "rb") as f:
        resp = requests.post(
            API_URL,
            files={"file": (os.path.basename(file_path), f)},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamps": "true",
                "diarize": "true",
                "include_diarization_in_text": "false",
            },
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()


def transcribe_batch(file_objs):
    """
    Process multiple files sequentially, streaming progress after each file.
    Yields: (status_rows, results_or_NO_UPDATE, progress_text)

    Intermediate yields use gr.update() (no-op) for the State so Gradio does NOT
    reset the state or fire .change events until the final yield delivers results.
    """
    if not file_objs:
        yield [], [], ""
        return

    # Gradio may pass a list of dicts with 'name'/'path' key or plain strings
    paths = []
    for f in file_objs:
        if isinstance(f, dict):
            paths.append(f.get("name") or f.get("path", ""))
        else:
            paths.append(str(f))

    total = len(paths)
    results = []
    status_rows = []

    for idx, path in enumerate(paths):
        name = os.path.basename(path)
        pct = int(idx / total * 100)
        progress_text = f"**Processing file {idx + 1} of {total}** ({pct}%)"

        # gr.update() with no args = leave the State unchanged (don't trigger .change)
        yield status_rows + [[name, "Processing…", "–", "–"]], gr.update(), progress_text

        try:
            data = _call_api(path)
            segments = data.get("segments", [])
            speakers = len({s.get("speaker") for s in segments if s.get("speaker")})
            results.append({"name": name, "path": path, "segments": segments})
            status_rows.append([name, "Done ✓", str(speakers), str(len(segments))])
        except requests.exceptions.ConnectionError:
            status_rows.append([name, "Error: server not running", "–", "–"])
        except Exception as e:
            status_rows.append([name, f"Error: {e}", "–", "–"])

        yield status_rows, gr.update(), progress_text

    done_count = sum(1 for r in status_rows if r[1].startswith("Done"))
    yield status_rows, results, f"**Done — {done_count} of {total} files completed.**"


def _get_batch_choices(results: list):
    """Populate dropdown with names of successfully transcribed files."""
    choices = [r["name"] for r in results] if results else []
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _view_batch_transcript(results: list, selected_name: str) -> str:
    """Render markdown for the selected file."""
    if not results or not selected_name:
        return ""
    empty_df = pd.DataFrame({"Detected Label": [], "Name": []})
    for r in results:
        if r["name"] == selected_name:
            return build_markdown(r["segments"], empty_df)
    return ""


def _export_single_batch(results: list, selected_name: str):
    """Write selected file's transcript to a temp .md and return its path."""
    if not results or not selected_name:
        return None
    empty_df = pd.DataFrame({"Detected Label": [], "Name": []})
    for r in results:
        if r["name"] == selected_name:
            md = build_markdown(r["segments"], empty_df)
            stem = os.path.splitext(selected_name)[0]
            tmp = tempfile.NamedTemporaryFile(
                suffix=".md", prefix=f"{stem}_", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(md)
            tmp.close()
            return tmp.name
    return None


def export_batch_zip(results: list):
    """Create a ZIP of per-file .md transcripts."""
    if not results:
        return None

    empty_df = pd.DataFrame({"Detected Label": [], "Name": []})
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            md = build_markdown(r["segments"], empty_df)
            stem = os.path.splitext(r["name"])[0]
            zf.writestr(f"{stem}.md", md)

    return tmp.name


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Parakeet Transcriber") as demo:

    with gr.Tabs():

        # ── Single File tab ───────────────────────────────────────────────
        with gr.Tab("Single File"):
            segments_state = gr.State([])

            gr.Markdown("Upload or record audio, transcribe it, rename speakers, then export.")

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Audio (record or upload)",
                    )
                    transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
                    status_md = gr.Markdown("")

                with gr.Column(scale=2):
                    with gr.Group(visible=False) as results_group:
                        speaker_table = gr.Dataframe(
                            headers=["Detected Label", "Name"],
                            datatype=["str", "str"],
                            label="Rename Speakers  (edit the Name column)",
                            interactive=True,
                            wrap=True,
                            row_count=(1, "dynamic"),
                            column_count=(2, "fixed"),
                        )
                        preview_md = gr.Markdown(label="Transcript Preview")
                        export_btn = gr.DownloadButton(
                            "Export Markdown",
                            variant="secondary",
                            visible=True,
                        )

            transcribe_btn.click(
                fn=transcribe,
                inputs=[audio_input],
                outputs=[segments_state, speaker_table, results_group, status_md, preview_md],
            )
            speaker_table.change(
                fn=update_preview,
                inputs=[segments_state, speaker_table],
                outputs=[preview_md],
            )
            export_btn.click(
                fn=export_markdown,
                inputs=[segments_state, speaker_table],
                outputs=[export_btn],
            )

        # ── Batch tab ─────────────────────────────────────────────────────
        with gr.Tab("Batch"):
            batch_results_state = gr.State([])

            gr.Markdown(
                "Drop multiple audio files. Each is sent to the server in turn "
                "(GPU work is queued automatically)."
            )

            with gr.Row():
                # Left: controls + progress
                with gr.Column(scale=1, min_width=280):
                    batch_input = gr.File(
                        file_count="multiple",
                        file_types=["audio", ".m4a", ".mp3", ".wav", ".ogg", ".flac", ".webm"],
                        label="Audio files",
                    )
                    batch_btn = gr.Button("Process All", variant="primary", size="lg")
                    batch_status_md = gr.Markdown("")

                # Right: progress table
                with gr.Column(scale=2):
                    batch_table = gr.Dataframe(
                        headers=["File", "Status", "Speakers", "Segments"],
                        datatype=["str", "str", "str", "str"],
                        label="Progress",
                        interactive=False,
                        wrap=True,
                    )

            # Results panel — hidden until batch completes
            with gr.Group(visible=False) as batch_results_group:
                gr.Markdown("---")
                batch_file_select = gr.Dropdown(
                    label="View transcript",
                    choices=[],
                    interactive=True,
                )
                batch_preview_md = gr.Markdown()
                with gr.Row():
                    batch_download_one_btn = gr.DownloadButton(
                        "Download Selected .md",
                        variant="secondary",
                    )
                    batch_export_btn = gr.DownloadButton(
                        "Export All (ZIP)",
                        variant="secondary",
                    )

            # --- Wiring ---

            batch_btn.click(
                fn=transcribe_batch,
                inputs=[batch_input],
                outputs=[batch_table, batch_results_state, batch_status_md],
            )
            batch_results_state.change(
                fn=_get_batch_choices,
                inputs=[batch_results_state],
                outputs=[batch_file_select],
            )
            batch_results_state.change(
                fn=lambda r: gr.update(visible=bool(r)),
                inputs=[batch_results_state],
                outputs=[batch_results_group],
            )
            batch_file_select.change(
                fn=_view_batch_transcript,
                inputs=[batch_results_state, batch_file_select],
                outputs=[batch_preview_md],
            )
            batch_download_one_btn.click(
                fn=_export_single_batch,
                inputs=[batch_results_state, batch_file_select],
                outputs=[batch_download_one_btn],
            )
            batch_export_btn.click(
                fn=export_batch_zip,
                inputs=[batch_results_state],
                outputs=[batch_export_btn],
            )


if __name__ == "__main__":
    demo.queue()  # required for generator streaming (live progress updates)
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, theme=gr.themes.Soft())
