import streamlit as st
import os
from PIL import Image
from ObjectDetection.pipeline.training_pipeline import TrainPipeline
from ObjectDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
import base64

# --- Config ---
st.set_page_config(page_title="YOLOv5 Object Detection", layout="centered")
st.title("📦 YOLOv5 Object Detection Web App")

# --- File Handler ---
INPUT_IMAGE = "data/inputImage.jpg"
OUTPUT_IMAGE = "yolov5/runs/detect/exp/inputImage.jpg"
WEIGHTS = "yolov5s.pt"  # or use 'best.pt' if you have

# --- Helper Functions ---
def run_detection(image_path):
    command = f"cd yolov5 && python detect.py --weights {WEIGHTS} --img 416 --conf 0.5 --source ../{image_path}"
    os.system(command)

def remove_output():
    os.system("rm -rf yolov5/runs")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📸 Live Detection", "🛠 Train Model"])

# --- 🔍 PREDICTION TAB ---
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        with open(INPUT_IMAGE, "wb") as f:
            f.write(uploaded_file.read())
        st.image(INPUT_IMAGE, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Running YOLOv5 Detection..."):
                run_detection(INPUT_IMAGE)
                output_path = OUTPUT_IMAGE
                if os.path.exists(output_path):
                    st.success("Detection Completed!")
                    st.image(output_path, caption="Output Image", use_column_width=True)
                    remove_output()
                else:
                    st.error("Detection failed. Please check your setup.")

# --- 📸 LIVE DETECTION TAB ---
with tab2:
    if st.button("Start Live Camera Detection"):
        st.warning("This will open a system camera window via CLI.")
        os.system(f"cd yolov5 && python detect.py --weights {WEIGHTS} --img 416 --conf 0.5 --source 0")
        remove_output()

# --- 🛠 TRAINING TAB ---
with tab3:
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            pipeline = TrainPipeline()
            pipeline.run_pipeline()
            st.success("Model Training Successful ✅")
