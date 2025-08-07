FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for PyTorch and image processing
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from Backend folder and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only Backend folder contents to /app/Backend inside container
COPY Backend /app/Backend

# Set environment variable for Django settings
ENV DJANGO_SETTINGS_MODULE=Backend.settings

# Expose port 8000
EXPOSE 8000

# Run the ASGI app with uvicorn
CMD ["uvicorn", "Backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
