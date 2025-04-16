import streamlit as st
import os
import base64
import glob
import platform
from ObjectDetection.pipeline.training_pipeline import TrainPipeline
from ObjectDetection.utils.main_utils import decodeImage, encodeImageIntoBase64

# Page Config
st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")
filename = "inputImage.jpg"
abs_input_path = os.path.abspath(filename)

# Title
st.title("🔍 YOLOv5 Object Detection Web App")
st.sidebar.title("Options")
page = st.sidebar.radio("Choose an Action", ("Home", "Train Model", "Image Prediction", "Live Camera"))

# Home
if page == "Home":
    st.markdown("👋 Welcome to the YOLOv5 Streamlit App. Use the sidebar to navigate!")

# Train Model
elif page == "Train Model":
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress..."):
            try:
                pipeline = TrainPipeline()
                pipeline.run_pipeline()
                st.success("✅ Training Completed Successfully")
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

# Image Prediction
elif page == "Image Prediction":
    uploaded_file = st.file_uploader("Upload an image for prediction", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        encoded = base64.b64encode(image_data).decode()
        decodeImage(encoded, filename)

        st.image(image_data, caption="📥 Uploaded Image", use_container_width=True)

        if st.button("🧠 Run Prediction"):
            with st.spinner("Detecting objects..."):
                try:
                    command = f"cd yolov5 && python detect.py --weights yolov5s.pt --img 416 --conf 0.5 --source ../{filename} --save-txt --save-conf"
                    os.system(command)

                    result_dirs = sorted(glob.glob("yolov5/runs/detect/exp*"), key=os.path.getmtime, reverse=True)
                    if result_dirs:
                        result_files = glob.glob(os.path.join(result_dirs[0], "*.jpg"))
                        if result_files:
                            st.image(result_files[0], caption="🎯 Detection Result", use_container_width=True)
                        else:
                            st.error("❌ Detection result not found.")
                    else:
                        st.error("❌ No output directory found.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

# Live Camera
elif page == "Live Camera":
    if platform.system() == "Linux" and "DISPLAY" not in os.environ:
        st.error("❌ Live camera not supported on this remote server (no display detected).")
    else:
        st.warning("⚠️ This will open your webcam window (works only locally).")
        if st.button("📷 Start Live Camera"):
            os.system("cd yolov5 && python detect.py --weights yolov5s.pt --img 416 --conf 0.5 --source 0")
