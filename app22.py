import streamlit as st
import os
import base64
import glob
import subprocess
import shutil
import stat
import torch
from ObjectDetection.pipeline.training_pipeline import TrainPipeline
from ObjectDetection.utils.main_utils import decodeImage

# --------------------------
# Setup
# --------------------------


st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")
st.title("🔍 YOLOv5 Object Detection Web App")

# Paths
filename = os.path.join(os.getcwd(), "inputImage.jpg")
video_filename = os.path.join(os.getcwd(), "inputVideo.mp4")
weights_path = os.path.join(os.getcwd(), "yolov5", "yolov5s.pt")
detect_folder = os.path.join("yolov5", "runs", "detect")

# --------------------------
# Helper Functions
# --------------------------

# Robust folder deletion handler
def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# If needed, redefine decodeImage
def decodeImage(imgstring, fileName):
    with open(fileName, "wb") as f:
        f.write(base64.b64decode(imgstring))

# --------------------------
# Sidebar Navigation
# --------------------------

st.sidebar.title("Options")
page = st.sidebar.radio("Choose an Action", ("Home", "Train Model", "Image Prediction", "Video Prediction"))

# --------------------------
# Home
# --------------------------

if page == "Home":
    st.markdown("👋 Upload images or videos to detect objects using YOLOv5.")

# --------------------------
# Train Model
# --------------------------

elif page == "Train Model":
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress..."):
            try:
                pipeline = TrainPipeline()
                pipeline.run_pipeline()
                st.success("✅ Training Completed")
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

# --------------------------
# Image Prediction
# --------------------------

elif page == "Image Prediction":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image_data = uploaded_file.read()
        encoded = base64.b64encode(image_data).decode()
        decodeImage(encoded, filename)
        st.image(image_data, caption="📥 Uploaded Image", use_container_width=True)

        if st.button("🧠 Run Detection"):
            with st.spinner("Detecting..."):
                try:
                    if os.path.exists(detect_folder):
                        shutil.rmtree(detect_folder, onerror=handle_remove_readonly)

                    command = [
                        "python", "detect.py",
                        "--weights", weights_path,
                        "--img", "416",
                        "--conf", "0.5",
                        "--source", filename
                    ]
                    result = subprocess.run(command, cwd="yolov5", capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(result.stderr)
                    else:
                        result_dirs = sorted(
                            glob.glob(os.path.join(detect_folder, "exp*")),
                            key=os.path.getmtime,
                            reverse=True
                        )
                        if result_dirs:
                            result_images = glob.glob(os.path.join(result_dirs[0], "*.jpg"))
                            if result_images:
                                st.image(result_images[0], caption="🎯 Detection Result", use_container_width=True)
                                with open(result_images[0], "rb") as f:
                                    st.download_button("📥 Download Result", f, file_name="detected.jpg")
                            else:
                                st.error("No result image found.")
                        else:
                            st.error("No output directory found.")
                except Exception as e:
                    st.error(str(e))

# --------------------------
# Video Prediction
# --------------------------

elif page == "Video Prediction":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_video:
        with open(video_filename, "wb") as f:
            f.write(uploaded_video.read())
        st.video(video_filename)

        if st.button("🧠 Run Detection on Video"):
            with st.spinner("Detecting objects..."):
                try:
                    if os.path.exists(detect_folder):
                        shutil.rmtree(detect_folder, onerror=handle_remove_readonly)

                    command = [
                        "python", "detect.py",
                        "--weights", weights_path,
                        "--img", "416",
                        "--conf", "0.5",
                        "--source", video_filename
                    ]
                    result = subprocess.run(command, cwd="yolov5", capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(result.stderr)
                    else:
                        result_dirs = sorted(
                            glob.glob(os.path.join(detect_folder, "exp*")),
                            key=os.path.getmtime,
                            reverse=True
                        )
                        if result_dirs:
                            result_videos = glob.glob(os.path.join(result_dirs[0], "*.mp4"))
                            if result_videos:
                                st.video(result_videos[0])
                                with open(result_videos[0], "rb") as f:
                                    st.download_button("📥 Download Result", f, file_name="detected_video.mp4")
                            else:
                                st.error("No result video found.")
                        else:
                            st.error("No output directory found.")
                except Exception as e:
                    st.error(str(e))
if st.button("🗑️ Delete Uploaded Video"):
    if os.path.exists("inputVideo.mp4"):
        os.remove("inputVideo.mp4")
        st.success("Video deleted.")
    else:
        st.warning("No video found to delete.")