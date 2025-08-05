import os
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

Model = None  # Global model object

def load_model_once():
    global Model  # <- This must come before you use or assign to Model

    if Model is None:
        print("Model is being loaded... This may take a while.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")


        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)

        model.fc = nn.Linear(model.fc.in_features, 1)

       
        model_path = os.path.join("static", "model", "best_resnet152_binary.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode

        Model = model
        print("✅ Model loaded from .pth file and ready.")
        return Model 
    else:
        print("Model is already loaded, skipping load.")
        
def get_model():
    
    if Model is None:
        raise RuntimeError("Model has not been loaded yet. Call load_model_once() first.")
    return Model
