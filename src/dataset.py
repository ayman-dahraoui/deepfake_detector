import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Chemin vers le dataset
DATA_PATH = r"C:\Users\ASUS\Downloads\4IASDR8\deepfake_detector\data\real_vs_fake\real-vs-fake"

# Transformations pour l'entraînement (avec augmentation)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Transformations pour validation/test (sans augmentation)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class DeepfakeDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # Charger les images fake (label = 1)
        fake_path = os.path.join(DATA_PATH, split, "fake")
        for img_name in os.listdir(fake_path):
            self.images.append(os.path.join(fake_path, img_name))
            self.labels.append(1)

        # Charger les images real (label = 0)
        real_path = os.path.join(DATA_PATH, split, "real")
        for img_name in os.listdir(real_path):
            self.images.append(os.path.join(real_path, img_name))
            self.labels.append(0)

        print(f"Dataset {split} : {len(self.images)} images chargées")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Lire l'image
        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        # Appliquer les transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_dataloaders(batch_size=32):
    train_dataset = DeepfakeDataset("train", transform=train_transform)
    val_dataset   = DeepfakeDataset("valid", transform=val_transform)
    test_dataset  = DeepfakeDataset("test",  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# Test rapide
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # Vérifier un batch
    images, labels = next(iter(train_loader))
    print(f"Shape batch images : {images.shape}")
    print(f"Shape batch labels : {labels.shape}")
    print(f"Exemple labels     : {labels[:8]}")
    print(f"Min/Max pixels     : {images.min():.2f} / {images.max():.2f}")
    print("\nDataset pret pour l'entrainement !")