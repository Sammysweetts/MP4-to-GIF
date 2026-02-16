import os
import math
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from imageio_ffmpeg import get_ffmpeg_exe

from moviepy.config import change_settings
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos


st.set_page_config(page_title="Video → GIF", layout="centered")

FFMPEG_PATH = get_ffmpeg_exe()
# Tell MoviePy to use the same ffmpeg binary (important on Streamlit Cloud)
change_settings({"FFMPEG_BINARY": FFMPEG_PATH})


def human_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    val = num_bytes / (1024**i)
    return f"{val:.2f} {units[i]}"


def get_video_info(video_path: str) -> dict:
    info = ffmpeg_parse_infos(video_path, print_infos=False)
    fps = float(info.get("video_fps") or 0.0)
    w, h = info.get("video_size") or (0, 0)
    duration = float(info.get("duration") or 0.0)
    return {"fps": fps, "width": w, "height": h, "duration": duration}


def run_ffmpeg(cmd: list[str]) -> tuple[int, str]:
    # Capture stderr (ffmpeg logs there)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, (p.stderr or "") + (p.stdout or "")


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
    clip_dur = max(0.0, end_s - start_s)
    if clip_dur <= 0:
        raise ValueError("End time must be greater than start time.")

    scale_factor = max(1, scale_pct) / 100.0

    # Keep layout = keep aspect ratio and scale both dimensions equally (no crop).
    # For best GIF quality: palettegen + paletteuse.
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
        # -ss before -i = fast seek
        "-ss",
        str(start_s),
        "-t",
        str(clip_dur),
        "-i",
        input_path,
        "-an",
        "-vf",
        vf,
    ]

    # Loop control for GIF
    # 0 = loop forever, -1 = no loop (implementation-dependent in viewers; 1 pass)
    cmd += ["-loop", "0" if loop_forever else "-1", output_path]

    rc, log = run_ffmpeg(cmd)
    if rc != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg failed.\n\nCommand:\n{' '.join(cmd)}\n\nLog:\n{log}")


st.title("Video → GIF (high-quality, same layout, adjustable FPS)")

st.markdown(
    """
**Meets your requirements**
1. **Best possible GIF quality** (palettegen/paletteuse). Note: GIF is limited to 256 colors, so it cannot *perfectly* match every video, but this is the standard best-quality approach.
2. **Match video FPS by default**, with a regulator to increase/decrease output FPS.
3. **Same layout as the video** (no cropping; same aspect ratio; default is full resolution).
"""
)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])

if "gif_bytes" not in st.session_state:
    st.session_state.gif_bytes = None
if "gif_name" not in st.session_state:
    st.session_state.gif_name = None

if uploaded is None:
    st.stop()

# Save upload to a temp file (Streamlit uploader is in-memory)
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
    tmp.write(uploaded.read())
    video_path = tmp.name

try:
    info = get_video_info(video_path)
except Exception as e:
    st.error(f"Could not read video metadata: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Video preview")
    st.video(uploaded)

with col2:
    st.subheader("Video info")
    st.write(f"Resolution: **{info['width']}×{info['height']}**")
    st.write(f"FPS (detected): **{info['fps']:.3f}**" if info["fps"] else "FPS (detected): **Unknown**")
    st.write(f"Duration: **{info['duration']:.2f} s**")

st.divider()
st.subheader("GIF settings")

duration = max(0.0, info["duration"])
default_end = min(duration, 5.0) if duration else 0.0

start_s, end_s = st.slider(
    "Trim (seconds)",
    min_value=0.0,
    max_value=float(duration) if duration else 0.0,
    value=(0.0, float(default_end)),
    step=0.1,
    help="Select the time range to convert.",
)

use_original_fps = st.checkbox(
    "Use original video FPS (recommended for matching)",
    value=True,
)

detected_fps = float(info["fps"] or 15.0)
detected_fps_rounded = max(1, int(round(detected_fps)))

out_fps = detected_fps
if not use_original_fps:
    out_fps = st.slider(
        "Output GIF FPS (regulator)",
        min_value=1,
        max_value=60,
        value=min(60, detected_fps_rounded),
        step=1,
        help="Lower FPS reduces file size; higher FPS increases smoothness (may duplicate frames if higher than source).",
    )
else:
    st.caption(f"Output FPS will match detected video FPS: {out_fps:.3f}")

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
        help="256 is best quality; lower can reduce size but may band more.",
    )
    dither = st.selectbox(
        "Dithering",
        options=["sierra2_4a", "bayer:bayer_scale=3", "none"],
        index=0,
        help="Dithering improves gradients but can add grain. 'sierra2_4a' is a good default.",
    )

loop_forever = st.checkbox("Loop GIF forever", value=True)

gif_name = Path(uploaded.name).stem + ".gif"
st.session_state.gif_name = gif_name

convert = st.button("Create GIF", type="primary")

if convert:
    st.session_state.gif_bytes = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
            gif_path = out_tmp.name

        with st.spinner("Converting with ffmpeg (high-quality palette)…"):
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
        st.error(str(e))
    finally:
        # best-effort cleanup
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
        label="Download GIF",
        data=st.session_state.gif_bytes,
        file_name=st.session_state.gif_name or "output.gif",
        mime="image/gif",
    )
else:
    st.info("Create a GIF to see the preview here.")

# Cleanup uploaded temp video
try:
    os.remove(video_path)
except Exception:
    pass
