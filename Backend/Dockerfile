FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for PyTorch and image processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the Backend directory into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Django
ENV DJANGO_SETTINGS_MODULE=Backend.settings

# Expose port
EXPOSE 8000

# Run using uvicorn (FastAPI via ASGI)
CMD ["uvicorn", "Backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
