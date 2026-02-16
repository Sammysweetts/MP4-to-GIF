import streamlit as st
import tempfile
import os
from moviepy.editor import VideoFileClip

# Page Configuration
st.set_page_config(
    page_title="High-Quality Video to GIF",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Video to GIF Converter")
st.markdown("""
    Convert your videos to GIF maintaining layout and frame rate.
""")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload a Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # --- 2. Save Uploaded File to Temp ---
    # MoviePy requires a file path, so we save the uploaded bytes to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    try:
        # Load the video
        clip = VideoFileClip(video_path)
        
        # Display Video Info
        st.info(f"**Original Resolution:** {clip.w}x{clip.h} | **Duration:** {clip.duration}s | **Original FPS:** {clip.fps}")

        # --- Settings Sidebar ---
        st.sidebar.header("GIF Settings")

        # REQUIREMENT 2: FPS Regulator (Default to video FPS)
        # We cap the slider at 60 to prevent browser crashes, but default is native
        default_fps = min(clip.fps, 60.0)
        fps_value = st.sidebar.slider(
            "Frame Rate (FPS)", 
            min_value=1.0, 
            max_value=60.0, 
            value=float(default_fps),
            step=1.0,
            help="Higher FPS = Smoother GIF but larger file size."
        )

        # REQUIREMENT 3: Layout (Dimensions)
        # Added a resize option for performance, but default is 'Original'
        resize_factor = st.sidebar.select_slider(
            "Dimensions (Scale)",
            options=[0.3, 0.5, 0.75, 1.0],
            value=1.0,
            format_func=lambda x: f"{int(x*100)}% (Original)" if x == 1.0 else f"{int(x*100)}%",
            help="Reduce size to prevent memory errors on large videos."
        )

        # REQUIREMENT 1: Quality Logic
        # We use FFMPEG optimization flags in the backend
        speed_up = st.sidebar.checkbox("Speed Up Generation (Lower Quality)", value=False)
        program_opt = 'ffmpeg' if not speed_up else 'imageio'

        # --- Preview & Conversion Section ---
        st.subheader("Preview Video")
        st.video(uploaded_file)

        if st.button("Generate GIF"):
            output_gif_path = video_path.replace(".mp4", ".gif")
            
            with st.spinner("Processing... This may take a moment depending on file size..."):
                try:
                    # Apply Resizing if selected
                    final_clip = clip.resize(resize_factor)
                    
                    # Write GIF
                    # opt='OptimizePlus' generally gives better compression/quality balance
                    # colors=256 ensures max GIF color depth
                    final_clip.write_gif(
                        output_gif_path, 
                        fps=fps_value,
                        program=program_opt,
                        colors=256, 
                        logger=None # Suppress terminal output
                    )
                    
                    st.success("GIF Generated Successfully!")
                    
                    # REQUIREMENT 4: GIF Preview
                    st.subheader("GIF Preview")
                    st.image(output_gif_path)

                    # Download Button
                    with open(output_gif_path, "rb") as file:
                        btn = st.download_button(
                            label="Download GIF",
                            data=file,
                            file_name="converted_video.gif",
                            mime="image/gif"
                        )
                        
                except Exception as e:
                    st.error(f"Error during conversion: {e}")
                    st.warning("Tip: If you ran out of memory, try reducing the Dimensions or FPS.")
                
                finally:
                    # Cleanup generated GIF
                    if os.path.exists(output_gif_path):
                        os.remove(output_gif_path)

        # Cleanup original video clip object
        clip.close()

    except Exception as e:
        st.error(f"Error loading video: {e}")
    
    finally:
        # Cleanup temp video file
        tfile.close()
        os.unlink(video_path)

else:
    st.info("Please upload a video file to begin.")
