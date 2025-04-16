import streamlit as st
import os
import base64
import glob
import subprocess
import shutil

from ObjectDetection.pipeline.training_pipeline import TrainPipeline
from ObjectDetection.utils.main_utils import decodeImage

# Page Config
st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")
st.title("🔍 YOLOv5 Object Detection Web App")

# Paths
filename = "inputImage.jpg"
video_filename = "inputVideo.mp4"
weights_path = "yolov5s.pt"

# Sidebar Navigation
st.sidebar.title("Options")
page = st.sidebar.radio("Choose an Action", ("Home", "Train Model", "Image Prediction", "Video Prediction"))

# Home
if page == "Home":
    st.markdown("👋 Upload images or videos to detect objects using YOLOv5.")

# Training
elif page == "Train Model":
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress..."):
            try:
                pipeline = TrainPipeline()
                pipeline.run_pipeline()
                st.success("✅ Training Completed")
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

# Image Detection
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
                    if os.path.exists("yolov5/runs/detect"):
                        shutil.rmtree("yolov5/runs/detect")

                    command = [
                        "python", "detect.py",
                        "--weights", weights_path,
                        "--img", "416",
                        "--conf", "0.5",
                        "--source", os.path.abspath(filename)
                    ]
                    result = subprocess.run(command, cwd="yolov5", capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(result.stderr)
                    else:
                        result_dirs = sorted(glob.glob("yolov5/runs/detect/exp*"), key=os.path.getmtime, reverse=True)
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

# Video Detection
elif page == "Video Prediction":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_video:
        with open(video_filename, "wb") as f:
            f.write(uploaded_video.read())
        st.video(video_filename)

        if st.button("🧠 Run Detection on Video"):
            with st.spinner("Detecting objects..."):
                try:
                    if os.path.exists("yolov5/runs/detect"):
                        shutil.rmtree("yolov5/runs/detect")

                    command = [
                        "python", "detect.py",
                        "--weights", weights_path,
                        "--img", "416",
                        "--conf", "0.5",
                        "--source", os.path.abspath(video_filename)
                    ]
                    result = subprocess.run(command, cwd="yolov5", capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(result.stderr)
                    else:
                        result_dirs = sorted(glob.glob("yolov5/runs/detect/exp*"), key=os.path.getmtime, reverse=True)
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
