"""
train.py

Trains an image classification model on a Roboflow Universe dataset
exported in "Folder" format (i.e. a standard ImageFolder layout):

    data/
      train/
        class_a/ img1.jpg ...
        class_b/ img1.jpg ...
      valid/
        class_a/ ...
        class_b/ ...
      test/
        class_a/ ...
        class_b/ ...

Approach: transfer learning with a ResNet18 backbone pretrained on
ImageNet. We freeze the backbone, replace the final fully-connected
layer with one sized to our number of classes, and fine-tune.

Usage:
    python src/train.py --data_dir data --epochs 10 --batch_size 32
    python src/train.py --data_dir data --epochs 15 --unfreeze_backbone
"""

import argparse
import copy
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(data_dir: str, batch_size: int, img_size: int = 224):
    """Build train/valid (and test, if present) dataloaders from an
    ImageFolder-style directory tree."""

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Could not find {train_dir}. Did you run download_dataset.py first? "
            "Expected an ImageFolder layout: data/train/<class>/*.jpg"
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)

    # Some Roboflow exports name the val split 'valid', others 'val'. Handle both,
    # and fall back to splitting off part of train if no val split exists.
    if os.path.isdir(valid_dir):
        val_ds = datasets.ImageFolder(valid_dir, transform=eval_tfms)
    elif os.path.isdir(os.path.join(data_dir, "val")):
        val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tfms)
    else:
        print("No 'valid' split found — splitting 15% off the training set instead.")
        n_val = int(0.15 * len(train_ds))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    class_names = train_ds.classes if hasattr(train_ds, "classes") else train_ds.dataset.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2) if test_ds else None
    )

    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes: int, unfreeze_backbone: bool = False) -> nn.Module:
    """ResNet18 pretrained on ImageNet, with a new classifier head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if not unfreeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # new head is always trainable
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0

    # Scope grad-tracking to this function call only. Using the bare
    # torch.set_grad_enabled(train) here (instead of as a context manager)
    # would leave gradients globally disabled after the first eval epoch,
    # silently breaking every training epoch that follows it.
    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

    return running_loss / total, running_correct / total


def plot_history(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier on a Roboflow dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset root (with train/valid/test)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save model + plots")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--unfreeze_backbone",
        action="store_true",
        help="Fine-tune the whole ResNet18 instead of just the new head (slower, often more accurate).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(len(class_names), unfreeze_backbone=args.unfreeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:>2}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.1f} min. Best val_acc={best_val_acc:.4f}")

    model.load_state_dict(best_state)

    # Save model + metadata needed for inference later
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": args.img_size,
            "architecture": "resnet18",
        },
        model_path,
    )
    print(f"Saved best model to {model_path}")

    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Optional: evaluate on test split if it exists
    if test_loader is not None:
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump({"test_loss": test_loss, "test_acc": test_acc}, f, indent=2)


if __name__ == "__main__":
    main()
