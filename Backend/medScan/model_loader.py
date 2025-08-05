import os
import torch

Model = None  # Global model object

def load_model_once():
    global Model  # <- This must come before you use or assign to Model

    if Model is None:
        print("Model is being loaded... This may take a while.")

        # Automatically use GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {device}")

        # model_path = os.path.join('static', 'models', 'your_model.pth')
        # Model = torch.load(model_path, map_location=device)
        # Model.to(device)
        # Model.eval()

        print("✅ Model loaded at startup.")
        return Model
    else:
        print("Model is already loaded, skipping load.")
        
def get_model():
    
    if Model is None:
        raise RuntimeError("Model has not been loaded yet. Call load_model_once() first.")
    return Model
