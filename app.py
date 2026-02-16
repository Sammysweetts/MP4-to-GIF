import streamlit as st
import tempfile
import os
import time
import io
import gc
import subprocess
import shutil
from pathlib import Path
from datetime import timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import imageio
from moviepy.editor import VideoFileClip, vfx
import cv2

# ─── Page Config ───
st.set_page_config(
    page_title="🎬 Video to GIF Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 50%, #16213e 100%);
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #FF4B4B, #FF8E53, #FFC837);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding-top: 1rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #8899AA;
        text-align: center;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255, 75, 75, 0.15);
    }

    .stat-box {
        background: linear-gradient(135deg, rgba(255,75,75,0.1), rgba(255,142,83,0.1));
        border: 1px solid rgba(255,75,75,0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #8899AA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .stButton > button {
        background: linear-gradient(90deg, #FF4B4B, #FF8E53);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(255, 75, 75, 0.4);
    }

    .stDownloadButton > button {
        background: linear-gradient(90deg, #00C851, #007E33);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }

    .step-indicator {
        display: inline-block;
        background: linear-gradient(135deg, #FF4B4B, #FF8E53);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-right: 8px;
    }

    .preview-container {
        border: 2px dashed rgba(255,75,75,0.3);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }

    .processing-info {
        background: rgba(255,200,55,0.1);
        border: 1px solid rgba(255,200,55,0.2);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }

    .success-box {
        background: rgba(0,200,81,0.1);
        border: 1px solid rgba(0,200,81,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stSlider > div > div > div > div {
        background-color: #FF4B4B;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FF8E53;
        border-bottom: 2px solid rgba(255,142,83,0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ───

def format_time(seconds):
    """Format seconds to MM:SS.ms"""
    td = timedelta(seconds=seconds)
    minutes = int(td.total_seconds() // 60)
    secs = td.total_seconds() % 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_video_info(video_path):
    """Extract video metadata."""
    clip = VideoFileClip(video_path)
    info = {
        "duration": clip.duration,
        "fps": clip.fps,
        "size": clip.size,
        "width": clip.size[0],
        "height": clip.size[1],
        "n_frames": int(clip.duration * clip.fps),
        "rotation": getattr(clip, 'rotation', 0),
    }
    clip.close()
    return info


def apply_text_overlay(frame, text, position, font_size, font_color, bg_color, opacity):
    """Apply text overlay to a frame using PIL."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    w, h = img.size
    padding = 10

    positions = {
        "Top-Left": (padding, padding),
        "Top-Center": ((w - text_w) // 2, padding),
        "Top-Right": (w - text_w - padding, padding),
        "Center": ((w - text_w) // 2, (h - text_h) // 2),
        "Bottom-Left": (padding, h - text_h - padding),
        "Bottom-Center": ((w - text_w) // 2, h - text_h - padding),
        "Bottom-Right": (w - text_w - padding, h - text_h - padding),
    }
    pos = positions.get(position, (padding, padding))

    if bg_color:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        bg_r, bg_g, bg_b = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        overlay_draw.rectangle(
            [pos[0] - padding, pos[1] - padding, pos[0] + text_w + padding, pos[1] + text_h + padding],
            fill=(bg_r, bg_g, bg_b, int(255 * opacity))
        )
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

    fc_r, fc_g, fc_b = tuple(int(font_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    draw.text(pos, text, fill=(fc_r, fc_g, fc_b), font=font)

    return np.array(img)


def apply_filters(frame, filters_config):
    """Apply image filters to a frame."""
    img = Image.fromarray(frame)

    # Brightness
    if filters_config.get("brightness", 1.0) != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(filters_config["brightness"])

    # Contrast
    if filters_config.get("contrast", 1.0) != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(filters_config["contrast"])

    # Saturation
    if filters_config.get("saturation", 1.0) != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(filters_config["saturation"])

    # Sharpness
    if filters_config.get("sharpness", 1.0) != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(filters_config["sharpness"])

    frame = np.array(img)

    # Special Effects
    effect = filters_config.get("effect", "None")

    if effect == "Grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif effect == "Sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        frame = cv2.transform(frame, kernel)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    elif effect == "Negative/Invert":
        frame = 255 - frame

    elif effect == "Blur":
        blur_amount = filters_config.get("blur_amount", 5)
        k = max(1, blur_amount) * 2 + 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)

    elif effect == "Edge Detection":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif effect == "Emboss":
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        frame = cv2.filter2D(frame, -1, kernel)

    elif effect == "Posterize":
        levels = filters_config.get("posterize_levels", 4)
        frame = (frame // (256 // levels)) * (256 // levels)
        frame = frame.astype(np.uint8)

    elif effect == "Pixelate":
        pixel_size = filters_config.get("pixel_size", 10)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // pixel_size), max(1, h // pixel_size)),
                           interpolation=cv2.INTER_LINEAR)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    elif effect == "Vignette":
        rows, cols = frame.shape[:2]
        X = cv2.getGaussianKernel(cols, cols / 2)
        Y = cv2.getGaussianKernel(rows, rows / 2)
        mask = Y * X.T
        mask = mask / mask.max()
        for i in range(3):
            frame[:, :, i] = frame[:, :, i] * mask

    elif effect == "Vintage":
        # Warm tone + slight blur + vignette
        frame = frame.astype(np.float32)
        frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.1, 0, 255)  # R
        frame[:, :, 1] = np.clip(frame[:, :, 1] * 0.95, 0, 255)  # G
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.8, 0, 255)  # B
        frame = frame.astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    elif effect == "Comic/Cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        frame = cv2.bitwise_and(color, color, mask=edges)

    return frame


def apply_crop(frame, crop_config, original_w, original_h):
    """Apply cropping to a frame."""
    crop_type = crop_config.get("type", "None")

    if crop_type == "None":
        return frame

    h, w = frame.shape[:2]

    if crop_type == "Custom":
        x1 = int(w * crop_config.get("left", 0) / 100)
        y1 = int(h * crop_config.get("top", 0) / 100)
        x2 = int(w * crop_config.get("right", 100) / 100)
        y2 = int(h * crop_config.get("bottom", 100) / 100)
        return frame[y1:y2, x1:x2]

    # Aspect ratio crops
    ratios = {
        "1:1 (Square)": 1.0,
        "16:9 (Widescreen)": 16 / 9,
        "9:16 (Portrait)": 9 / 16,
        "4:3 (Standard)": 4 / 3,
        "3:4 (Portrait Standard)": 3 / 4,
    }

    target_ratio = ratios.get(crop_type, 1.0)
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        return frame[:, offset:offset + new_w]
    else:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        return frame[offset:offset + new_h, :]


def create_gif(video_path, output_path, config, progress_callback=None):
    """Main GIF creation function."""
    clip = VideoFileClip(video_path)

    # Trim
    start = config.get("start_time", 0)
    end = config.get("end_time", clip.duration)
    clip = clip.subclip(start, min(end, clip.duration))

    # Speed
    speed = config.get("speed", 1.0)
    if speed != 1.0:
        clip = clip.fx(vfx.speedx, speed)

    # Reverse
    if config.get("reverse", False):
        clip = clip.fx(vfx.time_mirror)

    # Resize
    target_width = config.get("width", clip.size[0])
    target_height = config.get("height", clip.size[1])

    if config.get("maintain_aspect", True):
        ratio = min(target_width / clip.size[0], target_height / clip.size[1])
        new_size = (int(clip.size[0] * ratio), int(clip.size[1] * ratio))
    else:
        new_size = (target_width, target_height)

    clip = clip.resize(new_size)

    # Get FPS
    gif_fps = config.get("fps", 15)

    # Calculate total frames
    total_frames = int(clip.duration * gif_fps)

    if progress_callback:
        progress_callback(0, "Extracting frames...")

    frames = []
    frame_times = np.linspace(0, clip.duration - 1 / gif_fps, total_frames)

    for i, t in enumerate(frame_times):
        try:
            frame = clip.get_frame(t)
        except Exception:
            continue

        # Apply crop
        if config.get("crop"):
            frame = apply_crop(frame, config["crop"], new_size[0], new_size[1])

        # Apply filters
        if config.get("filters"):
            frame = apply_filters(frame, config["filters"])

        # Apply rotation
        rotation = config.get("rotation", 0)
        if rotation != 0:
            img = Image.fromarray(frame)
            img = img.rotate(-rotation, expand=True, fillcolor=(0, 0, 0))
            frame = np.array(img)

        # Apply flip
        if config.get("flip_horizontal", False):
            frame = np.fliplr(frame)
        if config.get("flip_vertical", False):
            frame = np.flipud(frame)

        # Apply text overlay
        if config.get("text_overlay") and config["text_overlay"].get("text"):
            tc = config["text_overlay"]
            frame = apply_text_overlay(
                frame, tc["text"], tc.get("position", "Bottom-Center"),
                tc.get("font_size", 24), tc.get("font_color", "#FFFFFF"),
                tc.get("bg_color", "#000000"), tc.get("opacity", 0.7)
            )

        frames.append(frame)

        if progress_callback:
            progress_callback((i + 1) / total_frames * 0.7,
                              f"Processing frame {i + 1}/{total_frames}")

    clip.close()

    if not frames:
        raise ValueError("No frames extracted from video!")

    # Handle ping-pong / boomerang
    if config.get("boomerang", False):
        frames = frames + frames[-2:0:-1]

    if progress_callback:
        progress_callback(0.75, "Creating GIF...")

    # Color quantization
    color_count = config.get("colors", 256)
    quality = config.get("quality", "Medium")

    # Dithering
    dither_map = {
        "None": Image.Dither.NONE,
        "Floyd-Steinberg": Image.Dither.FLOYDSTEINBERG,
    }
    dither = dither_map.get(config.get("dither", "Floyd-Steinberg"), Image.Dither.FLOYDSTEINBERG)

    # Convert to PIL images and quantize
    pil_frames = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.convert("RGB")

        if config.get("transparency", False):
            img = img.convert("RGBA")

        img_q = img.quantize(colors=color_count, dither=dither)
        pil_frames.append(img_q)

        if progress_callback:
            progress_callback(0.75 + (i + 1) / len(frames) * 0.15,
                              f"Quantizing frame {i + 1}/{len(frames)}")

    if progress_callback:
        progress_callback(0.9, "Saving GIF...")

    # Frame duration in ms
    duration_ms = int(1000 / gif_fps)

    # Loop count
    loop = 0 if config.get("loop", True) else 1

    # Save GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )

    # Optimize with gifsicle if available
    if config.get("optimize_gifsicle", True):
        try:
            gifsicle_path = shutil.which("gifsicle")
            if gifsicle_path:
                opt_level = config.get("optimization_level", 2)
                lossy_val = config.get("lossy", 0)

                cmd = [
                    gifsicle_path,
                    f"-O{opt_level}",
                    "--no-warnings",
                ]

                if lossy_val > 0:
                    cmd.append(f"--lossy={lossy_val}")

                cmd += ["-o", output_path, output_path]
                subprocess.run(cmd, capture_output=True, timeout=120)
        except Exception:
            pass

    if progress_callback:
        progress_callback(1.0, "Complete!")

    # Clean up
    del frames
    del pil_frames
    gc.collect()


def get_preview_frame(video_path, time_point, config):
    """Get a single processed preview frame."""
    clip = VideoFileClip(video_path)
    t = min(time_point, clip.duration - 0.01)
    frame = clip.get_frame(t)

    target_width = config.get("width", clip.size[0])
    target_height = config.get("height", clip.size[1])
    if config.get("maintain_aspect", True):
        ratio = min(target_width / clip.size[0], target_height / clip.size[1])
        new_size = (int(clip.size[0] * ratio), int(clip.size[1] * ratio))
    else:
        new_size = (target_width, target_height)

    img = Image.fromarray(frame)
    img = img.resize(new_size, Image.LANCZOS)
    frame = np.array(img)

    if config.get("crop"):
        frame = apply_crop(frame, config["crop"], new_size[0], new_size[1])

    if config.get("filters"):
        frame = apply_filters(frame, config["filters"])

    rotation = config.get("rotation", 0)
    if rotation != 0:
        img = Image.fromarray(frame)
        img = img.rotate(-rotation, expand=True, fillcolor=(0, 0, 0))
        frame = np.array(img)

    if config.get("flip_horizontal", False):
        frame = np.fliplr(frame)
    if config.get("flip_vertical", False):
        frame = np.flipud(frame)

    if config.get("text_overlay") and config["text_overlay"].get("text"):
        tc = config["text_overlay"]
        frame = apply_text_overlay(
            frame, tc["text"], tc.get("position", "Bottom-Center"),
            tc.get("font_size", 24), tc.get("font_color", "#FFFFFF"),
            tc.get("bg_color", "#000000"), tc.get("opacity", 0.7)
        )

    clip.close()
    return frame


# ─── Session State Init ───
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "gif_data" not in st.session_state:
    st.session_state.gif_data = None
if "gif_path" not in st.session_state:
    st.session_state.gif_path = None


# ─── Header ───
st.markdown('<h1 class="hero-title">🎬 Video to GIF Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Transform your videos into stunning GIFs with advanced controls</p>',
            unsafe_allow_html=True)

# Feature highlights
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🎨 15+ Effects</h4>
        <p style="color: #8899AA; font-size: 0.85rem;">Sepia, Cartoon, Vintage, Edge Detection & more</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>✂️ Smart Crop</h4>
        <p style="color: #8899AA; font-size: 0.85rem;">Aspect ratio presets & custom crop regions</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🔤 Text Overlay</h4>
        <p style="color: #8899AA; font-size: 0.85rem;">Add captions with custom fonts, colors & positioning</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="feature-card">
        <h4>⚡ Optimized Output</h4>
        <p style="color: #8899AA; font-size: 0.85rem;">Gifsicle compression for smallest file sizes</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Step 1: Upload ───
st.markdown('<p><span class="step-indicator">1</span><strong style="font-size:1.2rem;">Upload Your Video</strong></p>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v", "3gp"],
    help="Max file size: 200MB. Supported: MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V, 3GP"
)

if uploaded_file is not None:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.video_path = tmp.name

    try:
        st.session_state.video_info = get_video_info(st.session_state.video_path)
    except Exception as e:
        st.error(f"❌ Error reading video: {str(e)}")
        st.stop()

    info = st.session_state.video_info

    # Video info display
    st.markdown('<p><span class="step-indicator">ℹ</span><strong style="font-size:1.2rem;">Video Information</strong></p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{format_time(info['duration'])}</div>
            <div class="stat-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{info['fps']:.1f}</div>
            <div class="stat-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{info['width']}×{info['height']}</div>
            <div class="stat-label">Resolution</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{info['n_frames']:,}</div>
            <div class="stat-label">Frames</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        file_size_mb = os.path.getsize(st.session_state.video_path) / (1024 * 1024)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{file_size_mb:.1f} MB</div>
            <div class="stat-label">File Size</div>
        </div>
        """, unsafe_allow_html=True)

    # Preview original video
    with st.expander("📹 Preview Original Video", expanded=False):
        st.video(st.session_state.video_path)

    st.markdown("---")

    # ─── Step 2: Configuration ───
    st.markdown('<p><span class="step-indicator">2</span><strong style="font-size:1.2rem;">Configure Your GIF</strong></p>',
                unsafe_allow_html=True)

    # ─── Sidebar Config ───
    with st.sidebar:
        st.markdown("## ⚙️ GIF Settings")
        st.markdown("---")

        # ── Timing ──
        st.markdown('<p class="section-header">⏱ Timing</p>', unsafe_allow_html=True)

        time_range = st.slider(
            "Trim Video (seconds)",
            min_value=0.0,
            max_value=float(info["duration"]),
            value=(0.0, min(float(info["duration"]), 10.0)),
            step=0.1,
            format="%.1f",
            help="Select the portion of video to convert"
        )
        start_time, end_time = time_range
        clip_duration = end_time - start_time
        st.caption(f"Selected: {format_time(start_time)} → {format_time(end_time)} ({clip_duration:.1f}s)")

        speed = st.slider("Playback Speed", 0.25, 4.0, 1.0, 0.25,
                           help="1.0 = normal, 2.0 = 2x fast, 0.5 = half speed")

        col_a, col_b = st.columns(2)
        with col_a:
            reverse = st.checkbox("🔄 Reverse", False)
        with col_b:
            boomerang = st.checkbox("🪃 Boomerang", False,
                                     help="Play forward then backward")

        st.markdown("---")

        # ── Dimensions ──
        st.markdown('<p class="section-header">📐 Dimensions</p>', unsafe_allow_html=True)

        maintain_aspect = st.checkbox("Maintain Aspect Ratio", True)

        preset_sizes = {
            "Original": (info["width"], info["height"]),
            "Small (320px)": (320, 240),
            "Medium (480px)": (480, 360),
            "Large (640px)": (640, 480),
            "HD (720px)": (720, 480),
            "Social (500px)": (500, 500),
            "Custom": (0, 0),
        }

        size_preset = st.selectbox("Size Preset", list(preset_sizes.keys()), index=2)

        if size_preset == "Custom":
            gif_width = st.number_input("Width (px)", 50, 1920, info["width"], 10)
            gif_height = st.number_input("Height (px)", 50, 1080, info["height"], 10)
        else:
            gif_width, gif_height = preset_sizes[size_preset]

        st.markdown("---")

        # ── Quality ──
        st.markdown('<p class="section-header">🎯 Quality & Performance</p>', unsafe_allow_html=True)

        gif_fps = st.slider("GIF Frame Rate (FPS)", 5, 30, 15, 1,
                              help="Higher = smoother but larger file")

        color_count = st.select_slider(
            "Color Palette",
            options=[16, 32, 64, 128, 256],
            value=256,
            help="Fewer colors = smaller file"
        )

        dither_option = st.selectbox(
            "Dithering",
            ["Floyd-Steinberg", "None"],
            help="Dithering simulates more colors"
        )

        loop_gif = st.checkbox("Loop GIF", True)

        st.markdown("---")

        # ── Optimization ──
        st.markdown('<p class="section-header">🗜 Optimization</p>', unsafe_allow_html=True)

        optimize_gifsicle = st.checkbox("Use Gifsicle Optimization", True,
                                         help="Additional compression pass")

        if optimize_gifsicle:
            optimization_level = st.select_slider(
                "Optimization Level",
                options=[1, 2, 3],
                value=2,
                help="3 = best compression, slowest"
            )
            lossy_val = st.slider("Lossy Compression", 0, 200, 30, 10,
                                   help="0 = lossless, higher = smaller file")
        else:
            optimization_level = 2
            lossy_val = 0

    # ─── Main Area Tabs ───
    tab_crop, tab_filters, tab_text, tab_transform, tab_preview = st.tabs([
        "✂️ Crop", "🎨 Filters & Effects", "🔤 Text Overlay", "🔄 Transform", "👁 Preview"
    ])

    with tab_crop:
        st.markdown('<p class="section-header">Crop Settings</p>', unsafe_allow_html=True)

        crop_type = st.selectbox(
            "Crop Mode",
            ["None", "1:1 (Square)", "16:9 (Widescreen)", "9:16 (Portrait)",
             "4:3 (Standard)", "3:4 (Portrait Standard)", "Custom"]
        )

        crop_config = {"type": crop_type}

        if crop_type == "Custom":
            cc1, cc2 = st.columns(2)
            with cc1:
                crop_left = st.slider("Left %", 0, 49, 0)
                crop_top = st.slider("Top %", 0, 49, 0)
            with cc2:
                crop_right = st.slider("Right %", 51, 100, 100)
                crop_bottom = st.slider("Bottom %", 51, 100, 100)
            crop_config.update({
                "left": crop_left, "top": crop_top,
                "right": crop_right, "bottom": crop_bottom
            })

    with tab_filters:
        st.markdown('<p class="section-header">Image Adjustments</p>', unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            brightness = st.slider("Brightness", 0.0, 3.0, 1.0, 0.05)
            contrast = st.slider("Contrast", 0.0, 3.0, 1.0, 0.05)
        with fc2:
            saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.05)
            sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.05)

        st.markdown('<p class="section-header">Special Effects</p>', unsafe_allow_html=True)

        effect = st.selectbox(
            "Apply Effect",
            ["None", "Grayscale", "Sepia", "Negative/Invert", "Blur",
             "Edge Detection", "Emboss", "Posterize", "Pixelate",
             "Vignette", "Vintage", "Comic/Cartoon"]
        )

        filters_config = {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "sharpness": sharpness,
            "effect": effect,
        }

        if effect == "Blur":
            filters_config["blur_amount"] = st.slider("Blur Amount", 1, 20, 5)
        elif effect == "Posterize":
            filters_config["posterize_levels"] = st.slider("Posterize Levels", 2, 8, 4)
        elif effect == "Pixelate":
            filters_config["pixel_size"] = st.slider("Pixel Size", 2, 30, 10)

    with tab_text:
        st.markdown('<p class="section-header">Text Overlay</p>', unsafe_allow_html=True)

        overlay_text = st.text_input("Text", "", placeholder="Enter text for overlay...")

        text_config = {"text": overlay_text}

        if overlay_text:
            tc1, tc2 = st.columns(2)
            with tc1:
                text_position = st.selectbox(
                    "Position",
                    ["Top-Left", "Top-Center", "Top-Right", "Center",
                     "Bottom-Left", "Bottom-Center", "Bottom-Right"],
                    index=5
                )
                font_size = st.slider("Font Size", 10, 80, 24)
            with tc2:
                font_color = st.color_picker("Text Color", "#FFFFFF")
                bg_color = st.color_picker("Background Color", "#000000")
                bg_opacity = st.slider("BG Opacity", 0.0, 1.0, 0.7, 0.05)

            text_config.update({
                "position": text_position,
                "font_size": font_size,
                "font_color": font_color,
                "bg_color": bg_color,
                "opacity": bg_opacity,
            })

    with tab_transform:
        st.markdown('<p class="section-header">Transformations</p>', unsafe_allow_html=True)

        rotation = st.select_slider(
            "Rotation",
            options=[0, 90, 180, 270],
            value=0,
        )

        tf1, tf2 = st.columns(2)
        with tf1:
            flip_h = st.checkbox("Flip Horizontal", False)
        with tf2:
            flip_v = st.checkbox("Flip Vertical", False)

    with tab_preview:
        st.markdown('<p class="section-header">Live Preview</p>', unsafe_allow_html=True)

        preview_time = st.slider(
            "Preview at time (seconds)",
            min_value=float(start_time),
            max_value=float(end_time),
            value=float(start_time),
            step=0.1
        )

        if st.button("🔄 Generate Preview Frame"):
            with st.spinner("Generating preview..."):
                try:
                    preview_config = {
                        "width": gif_width,
                        "height": gif_height,
                        "maintain_aspect": maintain_aspect,
                        "crop": crop_config,
                        "filters": filters_config,
                        "rotation": rotation,
                        "flip_horizontal": flip_h,
                        "flip_vertical": flip_v,
                        "text_overlay": text_config,
                    }
                    preview_frame = get_preview_frame(
                        st.session_state.video_path,
                        preview_time,
                        preview_config
                    )
                    st.image(preview_frame, caption=f"Preview at {format_time(preview_time)}",
                             use_container_width=True)
                except Exception as e:
                    st.error(f"Preview error: {str(e)}")

    st.markdown("---")

    # ─── Step 3: Convert ───
    st.markdown('<p><span class="step-indicator">3</span><strong style="font-size:1.2rem;">Convert to GIF</strong></p>',
                unsafe_allow_html=True)

    # Estimated info
    effective_duration = clip_duration / speed
    if boomerang:
        effective_duration *= 2
    est_frames = int(effective_duration * gif_fps)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.info(f"📊 Estimated frames: **{est_frames}**")
    with ec2:
        st.info(f"⏱ Effective duration: **{effective_duration:.1f}s**")
    with ec3:
        st.info(f"📐 Output size: **{gif_width}×{gif_height}px**")

    if clip_duration > 30:
        st.warning("⚠️ Long clips (>30s) may produce very large GIF files. Consider trimming.")

    if st.button("🚀 Convert to GIF", key="convert_btn"):
        config = {
            "start_time": start_time,
            "end_time": end_time,
            "speed": speed,
            "reverse": reverse,
            "boomerang": boomerang,
            "width": gif_width,
            "height": gif_height,
            "maintain_aspect": maintain_aspect,
            "fps": gif_fps,
            "colors": color_count,
            "dither": dither_option,
            "loop": loop_gif,
            "optimize_gifsicle": optimize_gifsicle,
            "optimization_level": optimization_level,
            "lossy": lossy_val,
            "crop": crop_config,
            "filters": filters_config,
            "rotation": rotation,
            "flip_horizontal": flip_h,
            "flip_vertical": flip_v,
            "text_overlay": text_config,
        }

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(min(pct, 1.0))
            status_text.markdown(f'<div class="processing-info">⏳ {msg}</div>',
                                  unsafe_allow_html=True)

        start_proc_time = time.time()

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp_gif:
                output_gif_path = tmp_gif.name

            create_gif(st.session_state.video_path, output_gif_path, config, update_progress)

            proc_time = time.time() - start_proc_time
            gif_size = os.path.getsize(output_gif_path) / (1024 * 1024)

            with open(output_gif_path, "rb") as f:
                st.session_state.gif_data = f.read()
            st.session_state.gif_path = output_gif_path

            progress_bar.progress(1.0)
            status_text.empty()

            st.markdown(f"""
            <div class="success-box">
                <h3>✅ GIF Created Successfully!</h3>
                <p>Processed in <strong>{proc_time:.1f}s</strong> | 
                   Size: <strong>{gif_size:.2f} MB</strong> | 
                   Frames: <strong>{est_frames}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Conversion failed: {str(e)}")
            st.exception(e)

    # ─── Step 4: Download ───
    if st.session_state.gif_data:
        st.markdown("---")
        st.markdown('<p><span class="step-indicator">4</span><strong style="font-size:1.2rem;">Download & Preview</strong></p>',
                    unsafe_allow_html=True)

        dl1, dl2 = st.columns([1, 1])

        with dl1:
            gif_filename = Path(uploaded_file.name).stem + ".gif"
            st.download_button(
                label="⬇️ Download GIF",
                data=st.session_state.gif_data,
                file_name=gif_filename,
                mime="image/gif",
                key="download_gif",
            )

            gif_size_mb = len(st.session_state.gif_data) / (1024 * 1024)
            st.caption(f"File size: {gif_size_mb:.2f} MB")

        with dl2:
            if st.button("🗑 Clear & Start Over"):
                for key in ["video_path", "video_info", "gif_data", "gif_path"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Preview GIF
        st.markdown("### 🖼 GIF Preview")
        st.image(st.session_state.gif_data, use_container_width=True)

else:
    # Landing page when no video uploaded
    st.markdown("""
    <div class="preview-container">
        <h3 style="color: #FF8E53;">📤 Upload a video to get started</h3>
        <p style="color: #8899AA;">
            Drag and drop or click to upload your video file<br>
            Supports MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V, 3GP<br>
            Max file size: 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🌟 Features")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        **🎯 Precise Control**
        - Frame-accurate trimming
        - Variable speed (0.25x - 4x)
        - Reverse & Boomerang modes
        - Custom FPS control

        **📐 Smart Sizing**
        - Preset sizes for social media
        - Custom dimensions
        - Aspect ratio preservation
        - Multiple crop presets
        """)
    with f2:
        st.markdown("""
        **🎨 Effects & Filters**
        - Brightness, Contrast, Saturation
        - Sepia, Grayscale, Vintage
        - Comic/Cartoon, Pixelate
        - Edge Detection, Emboss
        - Posterize, Vignette, Blur

        **🔤 Text Overlay**
        - Custom text with positioning
        - Font size & color control
        - Background with opacity
        """)
    with f3:
        st.markdown("""
        **🔄 Transformations**
        - 90°/180°/270° rotation
        - Horizontal & vertical flip
        - Custom crop regions

        **⚡ Optimization**
        - Gifsicle compression
        - Adjustable color palette
        - Lossy compression control
        - Floyd-Steinberg dithering
        """)

# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; padding: 1rem;">
    <p>🎬 <strong>Video to GIF Pro</strong> | Built with ❤️ using Streamlit</p>
    <p style="font-size: 0.8rem;">Tip: For best results, keep clips under 10 seconds and use 480px width</p>
</div>
""", unsafe_allow_html=True)
