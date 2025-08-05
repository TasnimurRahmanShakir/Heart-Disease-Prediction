from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from medScan.model_loader import get_model  # Assuming this function loads your model
from PIL import Image
import io
import torch
import zipfile
import tempfile
import os
import aiofiles 
from fastapi.responses import FileResponse
from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import numpy as np


router = APIRouter()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


@router.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    print("Content type:", file.content_type)

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply same preprocessing as during training
    tensor_image = transform(image)
    print("Transformed image type:", type(tensor_image))
    assert isinstance(tensor_image, torch.Tensor), "Transform did not return a tensor"
    image = tensor_image.unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0
    
    # Return both prediction and confidence
    return {
        "prediction": "Cardiomegaly" if prediction == 1 else "No Finding",
        "confidence": round(prob, 4)
    }




def predict_image(model, device, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor_image = transform(image)
    print("Transformed image type:", type(tensor_image))
    assert isinstance(tensor_image, torch.Tensor), "Transform did not return a tensor"
    image = tensor_image.unsqueeze(0).to(device) 
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        return 1 if prob >= 0.5 else 0 
    
@router.post("/predict/folder")
async def predict_folder(file: UploadFile = File(...)):
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Received file:", file.filename)
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")
    zip_path = os.path.join(temp_dir, "images.zip")

    try:
        async with aiofiles.open(zip_path, "wb") as f:
            data = await file.read()
            print(f"Read {len(data)} bytes from uploaded file")
            await f.write(data)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = zip_ref.namelist()
            print(f"Extracted files: {extracted_files}")

        image_paths = []
        labels = []
        y_true = []
        y_pred = []
        y_prob = []
        class_map = {'Cardiomegaly': 1, 'No Finding': 0}
        for class_name in os.listdir(temp_dir):
            if class_name not in class_map:
                continue
            class_dir = os.path.join(temp_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(class_map[class_name])

        if not image_paths:
            print("No images found after extraction.")
            return {"status": "error", "detail": "No images found in the uploaded zip file."}

        model.eval()
        with torch.no_grad():
            for img_path, label in zip(image_paths, labels):
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as img_e:
                    print(f"Failed to open image {img_path}: {img_e}")
                    continue
                tensor = transform(img)
                if not isinstance(tensor, torch.Tensor):
                    print(f"Transform did not return a tensor for {img_path}")
                    continue
                img = tensor.unsqueeze(0).to(device)
                output = model(img)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob > 0.5 else 0
                y_true.append(label)
                y_pred.append(pred)
                y_prob.append(prob)

        if not y_true or not y_pred:
            print("No valid predictions made.")
            return {"status": "error", "detail": "No valid images for prediction."}

        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print(f'Accuracy: {acc:.4f}')

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cardiomegaly','No Finding'], yticklabels=['Cardiomegaly','No Finding'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        # ROC Curve & AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig('roc_curve.png')
            plt.close()
        except Exception as roc_e:
            print(f"ROC/AUC calculation failed: {roc_e}")
            roc_auc = None

        # Classification Report
        try:
            report = classification_report(y_true, y_pred, target_names=['Cardiomegaly','No Finding'])
        except Exception as rep_e:
            print(f"Classification report failed: {rep_e}")
            report = str(rep_e)

        return {
            'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix_img': 'confusion_matrix.png',
            'roc_curve_img': 'roc_curve.png' if roc_auc is not None else None
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "detail": str(e)}
