# Use an official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app


# Copy all files into the container
COPY . /app

# Install required Python packages
RUN pip install --upgrade pip
RUN pip install streamlit torch torchvision opencv-python Pillow matplotlib numpy PyYAML

# Expose Streamlit's default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
