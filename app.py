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
    Avoids MoviePy/Pillow compatibility issues.
    """
    cmd = [FFMPEG_PATH, "-hide_banner", "-i", video_path]
    rc, log = run(cmd)

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
        # Resolution
        m2 = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
        if m2:
            width, height = int(m2.group(1)), int(m2.group(2))

        # FPS: prefer "... 29.97 fps", else fallback to "... 30 tbr"
        m3 = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line)
        if m3:
            fps = float(m3.group(1))
        else:
            m4 = re.search(r"(\d+(?:\.\d+)?)\s*tbr", video_line)
            if m4:
                fps = float(m4.group(1))

    return {"duration": duration, "fps": fps, "width": width, "height": height, "raw_log": log}


def gif_safe_fps(requested_fps: float, min_delay_cs: int = 2) -> tuple[float, str, int]:
    """
    GIF frame delays are stored in centiseconds (1/100 s). Many players also clamp very small delays.
    If you ask for e.g. 60 fps (16.67 ms), it often gets rounded/clamped to 20 ms -> plays slower.

    We quantize to an *exactly representable* GIF frame delay:
      delay_cs = round(100 / fps), but at least min_delay_cs (default 2 -> 20 ms -> max 50 fps).
      effective_fps = 100 / delay_cs (exact)
    Returns: (effective_fps_float, ffmpeg_fps_expr, delay_cs)
    """
    if not requested_fps or requested_fps <= 0:
        requested_fps = 15.0

    delay_cs = int(round(100.0 / float(requested_fps)))
    delay_cs = max(min_delay_cs, min(100, delay_cs))  # clamp to [min_delay_cs..100]
    fps_expr = f"100/{delay_cs}"  # exact rational in ffmpeg
    effective_fps = 100.0 / delay_cs
    return effective_fps, fps_expr, delay_cs


def video_to_gif(
    input_path: str,
    output_path: str,
    start_s: float,
    end_s: float,
    fps_expr: str,
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
    vf = (
        f"fps={fps_expr},"
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
**Notes about speed (important):**
- GIF stores frame delays in **1/100 second** steps (centiseconds).
- If you request FPS that doesn't map cleanly to that (common: **60 fps**), many GIFs end up playing **slower** due to rounding/clamping.
- This app automatically adjusts your requested FPS to a **GIF-safe FPS** so playback speed matches the original timing much better.
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

detected_fps = float(info["fps"] or 30.0)
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
    requested_fps = detected_fps
else:
    requested_fps = float(
        st.slider(
            "Output GIF FPS (regulator)",
            min_value=1,
            max_value=60,
            value=min(60, max(1, int(round(detected_fps)))),
            step=1,
            help="Lower FPS reduces size; higher FPS increases smoothness. Non-GIF-safe FPS can play slower due to GIF timing limits, so we auto-adjust.",
        )
    )

# Make FPS GIF-safe to avoid slow playback due to centisecond rounding/clamping
effective_fps, fps_expr, delay_cs = gif_safe_fps(requested_fps, min_delay_cs=2)

st.caption(
    f"Requested FPS: {requested_fps:.3f} → **GIF-safe FPS: {effective_fps:.6f}** "
    f"(frame delay = {delay_cs} cs = {delay_cs*10} ms). "
    f"This prevents the common 'GIF plays slower than video' issue."
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

    gif_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
            gif_path = out_tmp.name

        with st.spinner("Converting (high-quality palette)…"):
            video_to_gif(
                input_path=video_path,
                output_path=gif_path,
                start_s=float(start_s),
                end_s=float(end_s),
                fps_expr=fps_expr,  # IMPORTANT: GIF-safe FPS expression
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
            if gif_path and os.path.exists(gif_path):
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
