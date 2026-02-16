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
    # We use explicit encoding/errors to prevent crashes on weird log characters
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    )
    log = (p.stderr or "") + ("\n" + p.stdout if p.stdout else "")
    return p.returncode, log


def probe_video_with_ffmpeg(video_path: str) -> dict:
    """
    Uses `ffmpeg -i` to parse duration, fps, resolution.
    We trust FFmpeg's display output here.
    """
    cmd = [FFMPEG_PATH, "-hide_banner", "-i", video_path]
    rc, log = run(cmd)

    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", log)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = hh * 3600 + mm * 60 + ss

    # Regex to find video stream info
    # We look for "Video: ..., 1920x1080, ..." pattern
    video_line = None
    for line in log.splitlines():
        if "Stream #" in line and "Video:" in line:
            video_line = line
            break

    width = height = 0
    fps = 0.0

    if video_line:
        # 1. Resolution
        m2 = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
        if m2:
            width, height = int(m2.group(1)), int(m2.group(2))

        # 2. FPS
        m3 = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line)
        if m3:
            fps = float(m3.group(1))
        else:
            m4 = re.search(r"(\d+(?:\.\d+)?)\s*tbr", video_line)
            if m4:
                fps = float(m4.group(1))
    
    # Check for rotation purely for display purposes in the UI
    rotation = 0
    m_rot = re.search(r"rotate\s*:\s*(\d+)", log)
    if m_rot:
        rotation = int(m_rot.group(1))
    elif re.search(r"rotation of\s*90", log):
        rotation = 90
    elif re.search(r"rotation of\s*-90", log):
        rotation = 270 # -90 usually means 270 in metadata

    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "rotation": rotation,
        "raw_log": log,
    }


def gif_safe_fps(requested_fps: float, min_delay_cs: int = 2) -> tuple[float, str, int]:
    """
    Aligns FPS to GIF centisecond delays (1/100s) to prevent playback speed drift.
    """
    if not requested_fps or requested_fps <= 0:
        requested_fps = 15.0

    # Calculate delay in centiseconds
    delay_cs = int(round(100.0 / float(requested_fps)))
    # Clamp delay (min 2 = 50fps max)
    delay_cs = max(min_delay_cs, min(100, delay_cs))
    
    fps_expr = f"100/{delay_cs}"
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

    # SCALE FILTER EXPLANATION:
    # 1. We allow FFmpeg to auto-rotate the input (default behavior). 
    #    So 'iw' and 'ih' are the correct Display Width/Height.
    # 2. We use 'iw*factor' and '-2' for height. 
    #    The '-2' tells FFmpeg to calculate height automatically but keep it divisible by 2 
    #    (required by some codecs/formats, safer for GIF).
    # 3. setsar=1 forces Square Pixels. This fixes "squished" looking GIFs on some players.
    
    vf = (
        f"fps={fps_expr},"
        f"scale=iw*{scale_factor}:-2:flags=lanczos,"
        f"setsar=1,"
        f"split[s0][s1];"
        f"[s0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither={dither}"
    )

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss", str(start_s),
        "-t", str(clip_dur),
        # Removed "-noautorotate" -> This fixes the portrait issue. 
        # FFmpeg will now apply the rotation metadata automatically before the filter graph.
        "-i", input_path,
        "-an",
        "-vf", vf,
        "-loop", "0" if loop_forever else "-1",
        output_path,
    ]

    rc, log = run(cmd)
    
    # Check if file exists and has size
    if rc != 0 or (not os.path.exists(output_path)) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"FFmpeg failed.\n\nLog:\n{log}")


# --- STREAMLIT UI ---

st.title("Video → GIF (Auto-Portrait Fix)")
st.markdown("Creates high-quality GIFs. Automatically detects phone orientation.")

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

# Probe video
info = probe_video_with_ffmpeg(video_path)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Preview")
    st.video(uploaded)

with col2:
    st.subheader("Details")
    st.write(f"Resolution: **{info['width']}×{info['height']}**")
    if info['rotation']:
        st.write(f"Orientation: **Rotated {info['rotation']}°** (Will be fixed automatically)")
    st.write(f"Duration: **{info['duration']:.2f} s**" if info["duration"] else "Duration: **Unknown**")
    st.write(f"FPS: **{info['fps']:.2f}**" if info["fps"] else "FPS: **Unknown**")

st.divider()

# --- CONTROLS ---

detected_fps = float(info["fps"] or 30.0)
duration = float(info["duration"] or 0.0)

# Time Trimming
if duration > 0:
    default_end = min(duration, 5.0)
    start_s, end_s = st.slider(
        "Trim (seconds)",
        min_value=0.0,
        max_value=duration,
        value=(0.0, float(default_end)),
        step=0.1,
    )
else:
    start_s = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=0.1)
    end_s = st.number_input("End time (s)", min_value=0.1, value=5.0, step=0.1)

# FPS Controls
use_original_fps = st.checkbox("Keep original FPS", value=True)
if use_original_fps:
    requested_fps = detected_fps
else:
    requested_fps = st.slider("Output FPS", 1, 60, 15)

effective_fps, fps_expr, delay_cs = gif_safe_fps(requested_fps)

# Dimensions
scale_pct = st.select_slider(
    "Resolution Scale (%)",
    options=[25, 50, 75, 100],
    value=100,
    help="100% keeps the display resolution of the video."
)

# Advanced Settings
with st.expander("Advanced Settings"):
    c1, c2 = st.columns(2)
    with c1:
        max_colors = st.slider("Max Colors", 32, 256, 256, step=16)
    with c2:
        dither = st.selectbox("Dithering Method", ["sierra2_4a", "bayer:bayer_scale=3", "none"], index=0)
    loop_forever = st.checkbox("Loop Forever", value=True)

st.session_state.gif_name = Path(uploaded.name).stem + ".gif"

# --- GENERATION ---

if st.button("Generate GIF", type="primary"):
    st.session_state.gif_bytes = None
    gif_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
            gif_path = out_tmp.name

        with st.spinner("Processing..."):
            video_to_gif(
                input_path=video_path,
                output_path=gif_path,
                start_s=float(start_s),
                end_s=float(end_s),
                fps_expr=fps_expr,
                scale_pct=int(scale_pct),
                max_colors=int(max_colors),
                dither=str(dither),
                loop_forever=bool(loop_forever),
                # Note: We no longer pass rotation explicitly. FFmpeg handles it.
            )

        with open(gif_path, "rb") as f:
            st.session_state.gif_bytes = f.read()

        st.success(f"Done! Size: {human_size(len(st.session_state.gif_bytes))}")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        # Cleanup generated file from disk (it's in memory now)
        if gif_path and os.path.exists(gif_path):
            os.remove(gif_path)

# --- DISPLAY RESULT ---

if st.session_state.gif_bytes:
    st.divider()
    st.subheader("Result")
    
    # We use use_container_width=True to ensure it fits the layout,
    # but since the aspect ratio is now correct, it won't look squished.
    st.image(st.session_state.gif_bytes, caption="Generated GIF", use_container_width=True)
    
    st.download_button(
        label="Download GIF",
        data=st.session_state.gif_bytes,
        file_name=st.session_state.gif_name,
        mime="image/gif",
    )

# Cleanup source
try:
    if os.path.exists(video_path):
        os.remove(video_path)
except:
    pass
