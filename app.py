import os
import re
import math
import base64
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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
    Parse metadata from `ffmpeg -i` output (no MoviePy, no Pillow dependency issues).
    """
    cmd = [FFMPEG_PATH, "-hide_banner", "-i", video_path]
    rc, log = run(cmd)  # ffmpeg often returns non-zero for -i; parse log anyway

    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", log)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = hh * 3600 + mm * 60 + ss

    video_line = None
    for line in log.splitlines():
        if "Stream #" in line and "Video:" in line:
            video_line = line
            break

    width = height = 0
    fps = 0.0

    if video_line:
        m2 = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
        if m2:
            width, height = int(m2.group(1)), int(m2.group(2))

        m3 = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line)
        if m3:
            fps = float(m3.group(1))
        else:
            m4 = re.search(r"(\d+(?:\.\d+)?)\s*tbr", video_line)
            if m4:
                fps = float(m4.group(1))

    return {"duration": duration, "fps": fps, "width": width, "height": height}


def is_gif_bytes(b: bytes) -> bool:
    return isinstance(b, (bytes, bytearray)) and len(b) >= 6 and (b[:6] in (b"GIF87a", b"GIF89a"))


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

    # Robust FPS fallback
    if out_fps is None or float(out_fps) <= 0:
        out_fps = 15.0

    scale_factor = max(1, int(scale_pct)) / 100.0

    # Ensure scale expressions evaluate to integers (avoid float dimension issues).
    if int(scale_pct) == 100:
        scale_filter = "scale=iw:ih:flags=lanczos"
    else:
        scale_filter = f"scale=trunc(iw*{scale_factor}):trunc(ih*{scale_factor}):flags=lanczos"

    # High-quality GIF: palettegen + paletteuse
    # format=rgb24 improves palette generation accuracy.
    vf = (
        f"fps={out_fps},"
        f"{scale_filter},"
        f"format=rgb24,"
        f"split[s0][s1];"
        f"[s0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither={dither}"
    )

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss", str(start_s),
        "-t", str(clip_dur),
        "-i", input_path,
        "-an",
        "-vf", vf,
        "-f", "gif",  # force GIF muxer
        "-loop", "0" if loop_forever else "-1",
        output_path,
    ]

    rc, log = run(cmd)
    if rc != 0 or (not os.path.exists(output_path)) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg failed.\n\nCommand:\n{' '.join(cmd)}\n\nLog:\n{log}")


def render_gif_preview(gif_bytes: bytes, max_width: str = "100%") -> None:
    """
    Use HTML <img> with data URI to ensure:
    - animated preview works reliably
    - browser recognizes it as image/gif (not png)
    """
    b64 = base64.b64encode(gif_bytes).decode("utf-8")
    html = f"""
    <div style="display:flex;justify-content:center;">
      <img src="data:image/gif;base64,{b64}" style="max-width:{max_width};height:auto;" />
    </div>
    """
    components.html(html, height=500, scrolling=False)


# ---------------- UI ----------------

st.title("Video → GIF")

st.markdown(
    """
**Requirements covered**
- **Layout = video layout**: no crop; aspect ratio preserved.
- **FPS**: matches detected FPS by default + regulator to change output FPS.
- **Quality**: best-practice `palettegen/paletteuse` conversion (GIF is limited to 256 colors).
- **Preview before download**: animated GIF preview section included.
"""
)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])

if "gif_bytes" not in st.session_state:
    st.session_state.gif_bytes = None
if "gif_name" not in st.session_state:
    st.session_state.gif_name = None

if not uploaded:
    st.stop()

# Save upload to a temp file
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
    st.write(f"Resolution: **{info['width']}×{info['height']}**" if info["width"] else "Resolution: **Unknown**")
    st.write(f"FPS (detected): **{info['fps']:.3f}**" if info["fps"] else "FPS (detected): **Unknown**")
    st.write(f"Duration: **{info['duration']:.2f} s**" if info["duration"] else "Duration: **Unknown**")

st.divider()
st.subheader("GIF settings")

detected_fps = float(info["fps"] or 0.0)
fallback_fps = 15.0
safe_detected_fps = detected_fps if detected_fps > 0 else fallback_fps

duration = float(info["duration"] or 0.0)
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

use_original_fps = st.checkbox("Use original video FPS (recommended)", value=True)

if use_original_fps:
    out_fps = safe_detected_fps
    if detected_fps <= 0:
        st.caption(f"Could not detect FPS reliably; using fallback FPS: {out_fps:.2f}")
    else:
        st.caption(f"Output FPS = detected FPS: {out_fps:.3f}")
else:
    out_fps = st.slider(
        "Output GIF FPS (regulator)",
        min_value=1,
        max_value=60,
        value=min(60, max(1, int(round(safe_detected_fps)))),
        step=1,
    )

scale_pct = st.select_slider(
    "Output size (keeps same layout/aspect ratio)",
    options=[25, 50, 75, 100],
    value=100,
)

with st.expander("Advanced quality controls"):
    max_colors = st.slider("Max colors (GIF limit 256)", 32, 256, 256, 16)
    dither = st.selectbox("Dithering", ["sierra2_4a", "bayer:bayer_scale=3", "none"], index=0)

loop_forever = st.checkbox("Loop forever", value=True)

st.session_state.gif_name = Path(uploaded.name).stem + ".gif"

if st.button("Generate GIF", type="primary"):
    st.session_state.gif_bytes = None

    gif_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
            gif_path = out_tmp.name

        with st.spinner("Converting…"):
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
            gif_bytes = f.read()

        if not is_gif_bytes(gif_bytes):
            raise RuntimeError("Output file is not a valid GIF (header check failed).")

        st.session_state.gif_bytes = gif_bytes
        st.success(f"GIF created: {human_size(len(gif_bytes))}")

    except Exception as e:
        st.error(f"Error during conversion: {e}")

    finally:
        if gif_path and os.path.exists(gif_path):
            try:
                os.remove(gif_path)
            except Exception:
                pass

st.divider()
st.subheader("GIF preview (animated)")

if st.session_state.gif_bytes:
    render_gif_preview(st.session_state.gif_bytes)

    st.download_button(
        label="Download GIF",
        data=st.session_state.gif_bytes,
        file_name=st.session_state.gif_name or "output.gif",
        mime="image/gif",
    )

    st.caption(
        "Tip: Don’t use right-click → “Save image as…” on the preview (browsers may save a PNG). "
        "Use the **Download GIF** button to get the actual .gif file."
    )
else:
    st.info("Generate a GIF to preview it here.")

# Cleanup temp uploaded video
try:
    if os.path.exists(video_path):
        os.remove(video_path)
except Exception:
    pass
