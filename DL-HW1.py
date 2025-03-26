import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
from PIL import Image
import pandas as pd
import argparse
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.utils.parametrizations as param
import timm
from timm.models.convnext import ConvNeXtBlock

# Use argparse choose mode
parser = argparse.ArgumentParser(description="Train or Infer ResNet50 model")
parser.add_argument("--train", action="store_true",
                    help="Enable training mode")
parser.add_argument("--infer", action="store_true",
                    help="Enable inference mode")
args = parser.parse_args()

# Default is --infer
if not args.train and not args.infer:
    args.train = True

# Env setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/ssd5/ia313553058/data"
MODEL_PATH = "resnet50_best_model.pth"
EPOCHS = 400
BATCH_SIZE = 128

# Data augument
transform_train = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.RandomCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class GCBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.LayerNorm([in_channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        context_mask = self.conv_mask(x).view(b, 1, -1)
        context_mask = self.softmax(context_mask)
        x_flat = x.view(b, c, -1)
        context = torch.bmm(x_flat, context_mask.transpose(1, 2))
        context = context.view(b, c, 1, 1)
        transform = self.transform(context)
        return input_x + transform


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet50V2_GC_SE(nn.Module):
    def __init__(self, num_classes=100, freeze_stem=True):
        super().__init__()
        self.backbone = timm.create_model(
            'resnetv2_50x1_bit.goog_in21k_ft_in1k',
            pretrained=True, num_classes=num_classes)

        self.stem = self.backbone.stem
        self.stage1 = self.backbone.stages[0]
        self.stage2 = self.backbone.stages[1]
        self.stage3 = nn.Sequential(
            self.backbone.stages[2],
            GCBlock(1024),
            SEBlock(1024)
        )
        self.stage4 = nn.Sequential(
            self.backbone.stages[3],
            GCBlock(2048)
        )

        self.norm = self.backbone.norm
        self.dropout = nn.Dropout(p=0.3)
        self.head = self.backbone.head

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False
            for p in self.stage1.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


# Model define
def get_model(freeze_stem=True):
    model = ResNet50V2_GC_SE(num_classes=100, freeze_stem=freeze_stem)
    num = 0
    freeze = 0
    for para in model.parameters():
        if para.requires_grad:
            num += para.numel()
        else:
            freeze += para.numel()

    print(f"Trainable Parameters: {num / 1_000_000:.6f} M")
    print(f"Frozen Parameters: {freeze / 1_000_000:.6f} M")
    return model.to(DEVICE)


# Train mode
if args.train:
    print("Training mode activated!")

    # Load data set
    train_dataset = datasets.ImageFolder(
        root=f"{DATA_PATH}/train",
        transform=transform_train)
    val_dataset = datasets.ImageFolder(
        root=f"{DATA_PATH}/val",
        transform=transform_val)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4)

    # Define model
    model = get_model(freeze_stem=True)

    # count class weights,log for smooth
    class_counts = np.bincount(train_dataset.targets)
    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [np.log(1 + (total_samples
         / (len(class_counts) * (class_counts[i] + 1))))
         for i in range(len(class_counts))],
        dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.SGD(
        model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5)

    MAX_train_accurate_value = 0
    MAX_val_accurate_value = 0
    best_epoch_train = 0
    best_epoch_val = 0
    train_accuracy_list = []
    val_accuracy_list = []

    # Training Loop
    scaler = torch.amp.GradScaler()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_correct_train = 0
        total_samples_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == labels).sum().item()
            total_samples_train += labels.size(0)

        train_accuracy_value = total_correct_train / total_samples_train
        train_accuracy_list.append(train_accuracy_value)
        print(f"Epoch {epoch} trained，Train Accuracy:", end=" ")
        print(f"{train_accuracy_value:.4f}")

        # Verify in each Epoch
        model.eval()
        total_correct_val = 0
        total_samples_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct_val += (predicted == labels).sum().item()
                total_samples_val += labels.size(0)

        val_accuracy_value = total_correct_val / total_samples_val
        val_accuracy_list.append(val_accuracy_value)
        print(f"Validation Accuracy: {val_accuracy_value:.4f}")

        # check if update best, and save model
        if train_accuracy_value > MAX_train_accurate_value:
            MAX_train_accurate_value = train_accuracy_value
            best_epoch_train = epoch
        else:
            print(f"train overfitting！(Epoch {epoch})")

        if val_accuracy_value > MAX_val_accurate_value:
            MAX_val_accurate_value = val_accuracy_value
            best_epoch_val = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model save！ (Epoch {epoch})")
        else:
            print(f"val overfitting！(Epoch {epoch})")

        torch.cuda.empty_cache()

    # Epoch finished, out best accuracy
    print(f"Best Train Accuracy:")
    print(f"{MAX_train_accurate_value:.4f}")
    print(f"(於 Epoch {best_epoch_train})")
    print(f"Best Val Accuracy:")
    print(f"{MAX_val_accurate_value:.4f}")
    print(f"(於 Epoch {best_epoch_val})")

    # plot trend graph
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, EPOCHS + 1), train_accuracy_list,
        label="Train Accuracy", marker='o')
    plt.plot(
        range(1, EPOCHS + 1), val_accuracy_list,
        label="Validation Accuracy", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")
    plt.savefig("accuracy_trend.png")
    plt.show()

# infer mode
if args.infer:
    print("🚀 Inference mode activated!")

    # make sure model exist
    while not os.path.exists(MODEL_PATH):
        print(f"Waiting for model file '{MODEL_PATH}' to be saved...")
        time.sleep(5)  # Wait and check again

    # Load model
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    class CustomImageDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.png"))
            self.transform = transform
            print(
                f"Found {len(self.image_paths)} test images in {image_dir}")

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path

    test_dataset = CustomImageDataset(
        image_dir=f"{DATA_PATH}/test", transform=transform_test)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True)
    predictions = []
    image_names = []

    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().tolist())
            image_names.extend(
                [os.path.splitext(os.path.basename(p))[0] for p in paths])
    df = pd.DataFrame({"image_name": image_names, "pred_label": predictions})
    df.to_csv("prediction.csv", index=False)

    print("Finish, saved the prediction.csv！")
