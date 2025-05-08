import streamlit as st
import os
import subprocess
from PIL import Image, ImageDraw
import torch
import tempfile
import cv2
import matplotlib.pyplot as plt
from realesrgan import RealESRGAN

# ----- UI Components -----
import torch

# GPU capability check
gpu_available = torch.cuda.is_available()
gpu_status = "✅ CUDA is available. GPU acceleration enabled." if gpu_available else "⚠️ CUDA not available. Running on CPU."
st.info(gpu_status)

import platform

user_os = platform.system()
if user_os == "Windows":
    st.markdown("🔧 Download the [Windows Launcher](Run_3rockglobal_AI_Upscaler.bat)")
elif user_os == "Darwin":
    st.markdown("🍎 Download the [Mac Launcher](Run_3rockglobal_AI_Upscaler.command)")
else:
    st.markdown("💡 Please use the terminal to run: `streamlit run app.py`")
st.set_page_config(page_title="3rockglobal AI Upscaler", page_icon="logo.png")

# GitHub-style badges
st.markdown("""
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![View on GitHub](https://img.shields.io/badge/GitHub-3rockglobal--ai--upscaler-blue?logo=github)](https://github.com/your-username/your-repo-name)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Enabled-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
""")
st.image("logo.png", width=150)
st.title("AI Video Upscaler with Real-ESRGAN")
st.markdown("Upscale and enhance your video using Real-ESRGAN. Configure downscaling, aspect padding, model choice, and preview side-by-side comparison.")

# Upload video
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if video_file:
    import time
    import shutil
    from pathlib import Path

    # Save temp input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    st.markdown(f"**Original Resolution:** {original_width}×{original_height}")
    is_landscape = original_width >= original_height

    # Suggest ranges
    min_scale, max_scale = 0.25, 4.0
    min_res = (int(original_width * min_scale), int(original_height * min_scale))
    max_res = (int(original_width * max_scale), int(original_height * max_scale))
    st.markdown(f"Suggested min: {min_res[0]}×{min_res[1]} | max: {max_res[0]}×{max_res[1]}")

    # Inputs
    selected_model = st.selectbox("Choose model", ["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B", "RealESRGANv2-anime"])
    scale_factor = st.selectbox("Upscale factor", [2.0, 4.0])
    downscale_factor = st.slider("Downscale before upscaling", 0.1, 1.0, 1.0, 0.05)
    target_width = st.number_input("Target width", value=1920)
    target_height = st.number_input("Target height", value=1080)
    pad_color = tuple(map(int, st.text_input("Padding color (R,G,B)", "0,0,0").split(",")))
    enable_pip = st.checkbox("Enable picture-in-picture", value=True)
    enable_text_overlay = st.checkbox("Enable text overlay", value=True)
    output_video_name = st.text_input("Output video filename (no ext)", value="enhanced_output")

    if st.button("Start Processing"):
        log = []
        st.info("Processing started...")
        progress = st.progress(0)

        input_path = os.path.join("temp_input", video_file.name)
        os.makedirs("temp_input", exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(open(tmp_path, "rb").read())

        # Frame extraction + downscaling
        frame_dir = Path("temp_frames"); frame_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(input_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if downscale_factor < 1.0:
                new_size = (int(frame.shape[1]*downscale_factor), int(frame.shape[0]*downscale_factor))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(frame_dir / f"frame_{idx:04d}.png"), frame)
            idx += 1
            progress.progress(min(idx/total_frames, 1.0))
        cap.release()
        log.append(f"Extracted {idx} frames.")

        # Real-ESRGAN enhancement
        enhanced_dir = Path("enhanced_frames"); enhanced_dir.mkdir(exist_ok=True)
        model_path = f"weights/{selected_model}.pth"
        model = RealESRGAN(torch.device("cuda" if gpu_available else "cpu"), scale=int(scale_factor))
        model.load_weights(model_path)
        enhance_idx, start_time = 0, time.time()
        for frame_path in sorted(frame_dir.glob("*.png")):
            img = Image.open(frame_path).convert("RGB")
            sr_img = model.predict(img)
            sr_img.save(enhanced_dir / frame_path.name)
            enhance_idx += 1
            elapsed = time.time() - start_time
            avg = elapsed / enhance_idx
            remaining = int(avg * (idx - enhance_idx))
            progress.progress(min(enhance_idx / idx, 1.0))
            st.info(f"Enhancing {enhance_idx}/{idx} - ETA: {remaining}s")

        # Padding
        padded_dir = Path("padded_frames"); padded_dir.mkdir(exist_ok=True)
        for frame_path in sorted(enhanced_dir.glob("*.png")):
            img = Image.open(frame_path).convert("RGB")
            canvas = Image.new("RGB", (target_width, target_height), pad_color)
            inset_x = (target_width - img.width) // 2
            inset_y = (target_height - img.height) // 2
            canvas.paste(img, (inset_x, inset_y))
            canvas.save(padded_dir / frame_path.name)

        # Comparison
        comp_dir = Path("comparison_frames"); comp_dir.mkdir(exist_ok=True)
        for frame_path in sorted(padded_dir.glob("*.png")):
            name = frame_path.name
            orig = Image.open(frame_dir / name).convert("RGB")
            enh = Image.open(frame_path).convert("RGB")
            if enable_pip:
                pip = orig.resize((int(enh.width * 0.3), int(enh.height * 0.3)), Image.LANCZOS)
                enh.paste(pip, (20, 20))
                if enable_text_overlay:
                    draw = ImageDraw.Draw(enh)
                    draw.text((25, 25 + pip.height), "Before", fill=(255, 255, 255))
                    draw.text((25, 10), "After", fill=(255, 255, 255))
                enh.save(comp_dir / name)
            else:
                orig_resized = orig.resize((enh.width, enh.height), Image.LANCZOS)
                comp = Image.new("RGB", (enh.width*2, enh.height))
                comp.paste(orig_resized, (0,0)); comp.paste(enh, (enh.width,0))
                if enable_text_overlay:
                    draw = ImageDraw.Draw(comp)
                    draw.text((20,20), "Before", fill=(255,255,255))
                    draw.text((enh.width+20,20), "After", fill=(255,255,255))
                comp.save(comp_dir / name)

        # Reassemble
        output_dir = Path("output"); output_dir.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-framerate", str(int(fps)), "-i", str(padded_dir / "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", str(output_dir / f"{output_video_name}.mp4")])
        subprocess.run(["ffmpeg", "-framerate", str(int(fps)), "-i", str(comp_dir / "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", str(output_dir / f"{output_video_name}_comparison.mp4")])

        # Cleanup
        shutil.rmtree(frame_dir)
        shutil.rmtree(enhanced_dir)
        shutil.rmtree(padded_dir)
        shutil.rmtree(comp_dir)
        os.remove(input_path)
        os.remove(tmp_path)

        st.success("All processing complete!")
        st.video(str(output_dir / f"{output_video_name}.mp4"))
        st.video(str(output_dir / f"{output_video_name}_comparison.mp4"))
