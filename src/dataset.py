from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_DATA_DIR = PROJECT_ROOT / "full_data"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# 3. LOAD DATASETS
# ============================================================

def load_datasets(seed=42):
    if not FULL_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {FULL_DATA_DIR}"
        )

    full_dataset = datasets.ImageFolder(
        root=FULL_DATA_DIR,
        transform=image_transform,
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    for ds in (train_dataset, val_dataset, test_dataset):
        ds.classes = full_dataset.classes
        ds.class_to_idx = full_dataset.class_to_idx

    return train_dataset, val_dataset, test_dataset


# ============================================================
# 4. INITIALIZE DATASETS AND LOADERS
# ============================================================

train_dataset, val_dataset, test_dataset = load_datasets()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# 5. TEST THE DATASET MODULE
# ============================================================

def main():
    print("Dataset initialization successful.")
    print("Classes:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)


if __name__ == "__main__":
    main()