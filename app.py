import os
import re
import math
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from imageio_ffmpeg import get_ffmpeg_exe


st.set_page_config(page_title="Video → GIF", layout="centered")

FFMPEG_PATH = get_ffmpeg_exe()


def human_size(num_bytes: int) -> str:
    if not num_bytes:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    return f"{num_bytes / (1024 ** i):.2f} {units[i]}"


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    log = (p.stderr or "") + ("\n" + p.stdout if p.stdout else "")
    return p.returncode, log


def probe_video_with_ffmpeg(video_path: str) -> dict:
    """
    Uses `ffmpeg -i` output to parse duration, fps, width, height.
    This avoids MoviePy (which can trigger PIL.Image.ANTIALIAS issues with Pillow>=10).
    """
    cmd = [FFMPEG_PATH, "-hide_banner", "-i", video_path]
    rc, log = run(cmd)
    # Note: ffmpeg -i typically returns non-zero; we parse the log anyway.

    # Duration
    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", log)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = hh * 3600 + mm * 60 + ss

    # Find first video stream line
    video_line = None
    for line in log.splitlines():
        if "Stream #" in line and "Video:" in line:
            video_line = line
            break

    width = height = 0
    fps = 0.0

    if video_line:
        # Resolution: look for "####x####" near the Video line
        m2 = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
        if m2:
            width, height = int(m2.group(1)), int(m2.group(2))

        # FPS: prefer "... 29.97 fps", else fall back to "... 30 tbr"
        m3 = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line)
        if m3:
            fps = float(m3.group(1))
        else:
            m4 = re.search(r"(\d+(?:\.\d+)?)\s*tbr", video_line)
            if m4:
                fps = float(m4.group(1))

    return {"duration": duration, "fps": fps, "width": width, "height": height, "raw_log": log}


def video_to_gif(
    input_path: str,
    output_path: str,
    start_s: float,
    end_s: float,
    out_fps: float,
    scale_pct: int,
    max_colors: int,
    dither: str,
    loop_forever: bool,
) -> None:
    clip_dur = float(end_s) - float(start_s)
    if clip_dur <= 0:
        raise ValueError("End time must be greater than start time.")

    scale_factor = max(1, int(scale_pct)) / 100.0

    # High-quality GIF approach: palettegen + paletteuse
    # - Keeps video layout/aspect ratio (no crop)
    # - Uses lanczos scaler for best down/up-scaling quality
    vf = (
        f"fps={out_fps},"
        f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos,"
        f"split[s0][s1];"
        f"[s0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither={dither}"
    )

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(clip_dur),
        "-i",
        input_path,
        "-an",
        "-vf",
        vf,
        "-loop",
        "0" if loop_forever else "-1",
        output_path,
    ]

    rc, log = run(cmd)
    if rc != 0 or (not os.path.exists(output_path)) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg failed.\n\nCommand:\n{' '.join(cmd)}\n\nLog:\n{log}")


st.title("Video → GIF (quality-focused)")

st.markdown(
    """
**Your requirements covered:**
1. **Best possible GIF quality** using `palettegen/paletteuse` (GIF still has a hard 256-color limit).
2. **Match the video FPS by default**, with a regulator to increase/decrease output FPS.
3. **Same layout as the video** (no cropping; aspect ratio preserved; default is full resolution).
"""
)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])

if "gif_bytes" not in st.session_state:
    st.session_state.gif_bytes = None
if "gif_name" not in st.session_state:
    st.session_state.gif_name = None

if not uploaded:
    st.stop()

# Write uploaded file to temp path
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
    tmp.write(uploaded.read())
    video_path = tmp.name

info = probe_video_with_ffmpeg(video_path)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Video preview")
    st.video(uploaded)

with col2:
    st.subheader("Video info")
    if info["width"] and info["height"]:
        st.write(f"Resolution: **{info['width']}×{info['height']}**")
    else:
        st.write("Resolution: **Unknown**")

    st.write(f"FPS (detected): **{info['fps']:.3f}**" if info["fps"] else "FPS (detected): **Unknown**")
    st.write(f"Duration: **{info['duration']:.2f} s**" if info["duration"] else "Duration: **Unknown**")

st.divider()
st.subheader("GIF settings")

detected_fps = float(info["fps"] or 15.0)
duration = float(info["duration"] or 0.0)

if duration > 0:
    default_end = min(duration, 5.0)
    start_s, end_s = st.slider(
        "Trim (seconds)",
        min_value=0.0,
        max_value=duration,
        value=(0.0, float(default_end)),
        step=0.1,
        help="Select the time range to convert.",
    )
else:
    st.warning("Could not detect duration; using manual start/end inputs.")
    start_s = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=0.1)
    end_s = st.number_input("End time (s)", min_value=0.1, value=5.0, step=0.1)

use_original_fps = st.checkbox("Use original video FPS (recommended)", value=True)

if use_original_fps:
    out_fps = detected_fps
    st.caption(f"Output FPS set to detected FPS: {out_fps:.3f}")
else:
    out_fps = st.slider(
        "Output GIF FPS (regulator)",
        min_value=1,
        max_value=60,
        value=min(60, max(1, int(round(detected_fps)))),
        step=1,
        help="Lower FPS reduces size; higher FPS increases smoothness (may duplicate frames if higher than source).",
    )

scale_pct = st.select_slider(
    "Output size (keeps same layout/aspect ratio)",
    options=[25, 50, 75, 100],
    value=100,
    help="100% keeps the same resolution as the video. Lower values reduce file size.",
)

with st.expander("Advanced quality controls"):
    max_colors = st.slider(
        "Max colors (GIF limit is 256)",
        min_value=32,
        max_value=256,
        value=256,
        step=16,
    )
    dither = st.selectbox(
        "Dithering",
        options=["sierra2_4a", "bayer:bayer_scale=3", "none"],
        index=0,
    )

loop_forever = st.checkbox("Loop GIF forever", value=True)

st.session_state.gif_name = Path(uploaded.name).stem + ".gif"

if st.button("Generate GIF", type="primary"):
    st.session_state.gif_bytes = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
            gif_path = out_tmp.name

        with st.spinner("Converting (high-quality palette)…"):
            video_to_gif(
                input_path=video_path,
                output_path=gif_path,
                start_s=float(start_s),
                end_s=float(end_s),
                out_fps=float(out_fps),
                scale_pct=int(scale_pct),
                max_colors=int(max_colors),
                dither=str(dither),
                loop_forever=bool(loop_forever),
            )

        with open(gif_path, "rb") as f:
            st.session_state.gif_bytes = f.read()

        st.success(f"GIF created: {human_size(len(st.session_state.gif_bytes))}")

    except Exception as e:
        st.error(f"Error during conversion: {e}")

    finally:
        try:
            if "gif_path" in locals() and os.path.exists(gif_path):
                os.remove(gif_path)
        except Exception:
            pass

st.divider()
st.subheader("GIF preview (before download)")

if st.session_state.gif_bytes:
    st.image(st.session_state.gif_bytes, caption="Preview", use_container_width=True)

    st.download_button(
        "Download GIF",
        data=st.session_state.gif_bytes,
        file_name=st.session_state.gif_name or "output.gif",
        mime="image/gif",
    )
else:
    st.info("Generate a GIF to preview it here.")

# Cleanup temp uploaded video
try:
    os.remove(video_path)
except Exception:
    pass
