# 🩺 Cardiomegaly Detection App

A web application for detecting **Cardiomegaly (enlarged heart)** from chest X-ray images using **Deep Learning**.  
The system combines a **TypeScript + React frontend** with a **Django + FastAPI backend** and a **PyTorch ResNet-18 model** for accurate classification.

---

## 🚀 Features
- **Single Image Analysis**  
  Upload one chest X-ray → get prediction (`Cardiomegaly` / `Normal`).  

- **Batch Analysis**  
  Upload a ZIP/folder of chest X-rays → get detailed evaluation:  
  - Accuracy score  
  - Classification report (precision, recall, F1-score)  
  - Confusion matrix  
  - ROC curve  

---

## 🛠️ Tech Stack
- **Frontend:** React (TypeScript), CSS  
- **Backend:** Django, FastAPI  
- **Deep Learning:** PyTorch, Torchvision, scikit-learn  
- **Visualization:** Matplotlib  

---

### 1. Backend Setup (Django + FastAPI)

1. Create and activate a virtual environment:

    ```bash
    cd backend
    python -m venv venv
    
    # Activate virtual environment
    
    # Linux / Mac
    source venv/bin/activate
    
    # Windows
    venv\Scripts\activate
    ```
2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
  
3. Start Django server:
   ```bash
   uvicorn Backend.asgi:application --reload

### 2. Frontend Setup
1. Navigate to frontend folder:
   ```bash
   cd ../frontend

2. Install frontend dependencies:
   ```bash
   npm install

3. Start React development server:
   ```bash
   npm start



