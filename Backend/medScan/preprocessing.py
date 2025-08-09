#preprocessing.py
import torch
from PIL import Image
from torchvision import transforms
import os

# --- Dataset class ---
class FilteredBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_map={'Cardiomegaly': 1, 'No Finding': 0}, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        for class_name, label in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label, image_path