import tempfile
import os
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
# Core functions wired to Gradio events
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
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Parakeet Transcriber", theme=gr.themes.Soft()) as demo:
    segments_state = gr.State([])

    gr.Markdown("# Parakeet Transcriber\nUpload an audio file, transcribe it, rename speakers, then export.")

    with gr.Row():
        # --- Left panel ---
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio (record or upload)",
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
            status_md = gr.Markdown("")

        # --- Right panel (hidden until transcription complete) ---
        with gr.Column(scale=2):
            with gr.Group(visible=False) as results_group:
                speaker_table = gr.Dataframe(
                    headers=["Detected Label", "Name"],
                    datatype=["str", "str"],
                    label="Rename Speakers  (edit the Name column)",
                    interactive=True,
                    wrap=True,
                    row_count=(1, "dynamic"),
                    col_count=(2, "fixed"),
                )
                preview_md = gr.Markdown(label="Transcript Preview")
                export_btn = gr.DownloadButton(
                    "Export Markdown",
                    variant="secondary",
                    visible=True,
                )

    # --- Events ---
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
