#views.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import os
from medScan.model_loader import get_model  # Assuming this function loads your model
from PIL import Image
import io
import torch
import zipfile
import tempfile
import os
import aiofiles 
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import numpy as np
from medScan.preprocessing import FilteredBinaryDataset

router = APIRouter()


@router.get("/model/status")
async def model_status():
    return{
        "trained_model_loaded": "Model is loaded and ready for predictions."
    }

@router.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    print("Content type:", file.content_type)

    # Read and convert the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Transform image
    tensor_image = transform(image)
    assert isinstance(tensor_image, torch.Tensor), "Transform did not return a tensor"
    image_tensor = tensor_image.unsqueeze(0)  # Add batch dimension

    # Load model and make prediction
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0

    # Return result
    return {
        "prediction": "Cardiomegaly" if prediction == 1 else "No Finding",
        "confidence": round(prob, 4)
    }




    
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

         # Change to your validation/test folder path
        val_dataset = FilteredBinaryDataset(temp_dir)
        val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)
        
        class_names = ['No Finding', 'Cardiomegaly']

        # --- Run inference on the validation set ---
        all_labels = []
        all_preds = []
        all_probs = []
        all_paths = []

        with torch.no_grad():
            for images, labels, paths in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels.numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_paths.extend(paths)
                
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')   # Save PNG file
        # plt.show()
        plt.close()

        # --- Classification Report ---
        print("Classification Report:")
        report = classification_report(all_labels, all_preds, target_names=class_names)
        # --- ROC Curve & AUC ---
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7,6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')  # Save PNG file
        # plt.show()
        plt.close()

        # --- Visualization of some predictions with images ---
        def imshow(img_tensor, title=None):
            img = img_tensor.cpu().numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean  # unnormalize
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            if title:
                plt.title(title)
            plt.axis('off')

        num_images = 8
        plt.figure(figsize=(16, 6))
        for i in range(num_images):
            image_tensor, true_label, img_path = val_dataset[i]
            pred_label = all_preds[i]
            pred_prob = all_probs[i]
            title = f"T: {class_names[true_label]}\nP: {class_names[pred_label]} ({pred_prob:.2f})"

            plt.subplot(2, 4, i+1)
            imshow(image_tensor, title=title)
        plt.tight_layout()
        plt.savefig('sample_predictions.png')  # Save PNG file
        # plt.show()
        plt.close()

        return {
            'total_images': len(all_labels),
            'positive_cases': int(np.sum(np.array(all_preds))),
            'negative_cases': int(len(all_preds) - np.sum(np.array(all_preds))),
            'confusion_matrix_path': 'confusion_matrix.png' if os.path.exists('confusion_matrix.png') else None,
            'roc_curve_path': 'roc_curve.png' if os.path.exists('roc_curve.png') else None,
            'sample_predictions_path': 'sample_predictions.png' if os.path.exists('sample_predictions.png') else None,
            'classification_report': report,
            'roc_auc': round(roc_auc, 4)
        }


        
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "detail": str(e)}


@router.get("/download/{file_type}")
def download_file(file_type: str):
    file_map = {
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "sample_predictions": "sample_predictions.png"
    }
    file_path = file_map.get(file_type)

    if file_path and os.path.exists(file_path):
        return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='image/png')
    return {"error": "File not found"}
