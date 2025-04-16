import streamlit as st
import os
import sys
import base64
from ObjectDetection.pipeline.training_pipeline import TrainPipeline
from ObjectDetection.utils.main_utils import decodeImage, encodeImageIntoBase64

# Page Config
st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")

# Initialize client app state
filename = "inputImage.jpg"

# Title
st.title("🔍 YOLOv5 Object Detection Web App")

# --- Sidebar Options ---
st.sidebar.title("Options")
page = st.sidebar.radio("Choose an Action", ("Home", "Train Model", "Image Prediction", "Live Camera"))

# --- Home ---
if page == "Home":
    st.markdown("Welcome to the YOLOv5 Streamlit App. Use the sidebar to navigate!")

# --- Train ---
elif page == "Train Model":
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress..."):
            try:
                pipeline = TrainPipeline()
                pipeline.run_pipeline()
                st.success("✅ Training Completed Successfully")
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

# --- Predict on Image ---
elif page == "Image Prediction":
    uploaded_file = st.file_uploader("Upload an image for prediction", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        encoded = base64.b64encode(image_data).decode()

        # Decode and save input image
        decodeImage(encoded, filename)

        st.image(image_data, caption="📥 Uploaded Image", use_container_width=True)

        if st.button("🧠 Run Prediction"):
            with st.spinner("Detecting objects..."):
                try:
                    os.system("cd yolov5 && python detect.py --weights yolov5s.pt --img 416 --conf 0.5 --source ../inputImage.jpg")

                    # Load and display the result
                    result_path = "yolov5/runs/detect/exp/inputImage.jpg"
                    if os.path.exists(result_path):
                        st.image(result_path, caption="🎯 Detection Result", use_container_width=True)
                        # Cleanup
                        os.system("rm -rf yolov5/runs")
                    else:
                        st.error("Output image not found.")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# --- Live Camera ---
elif page == "Live Camera":
    st.warning("This will launch a camera window on the server (only works if GUI is available).")
    if st.button("📷 Start Live Camera"):
        os.system("cd yolov5 && python detect.py --weights yolov5s.pt --img 416 --conf 0.5 --source 0")
