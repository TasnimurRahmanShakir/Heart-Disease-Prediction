# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for PyTorch and image processing
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt from Backend folder to install dependencies
COPY Backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire Backend folder contents into /app inside container
COPY Backend/ .

# Set environment variable for Django settings
ENV DJANGO_SETTINGS_MODULE=Backend.settings

# Expose port 8000 (for uvicorn)
EXPOSE 8000

# Run your ASGI app with uvicorn
CMD ["uvicorn", "Backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
