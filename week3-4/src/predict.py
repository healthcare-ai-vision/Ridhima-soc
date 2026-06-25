"""
predict.py

Run the trained classifier on a single image (or a folder of images)
and print/save the predicted class + confidence.

Usage:
    python src/predict.py --model outputs/model.pt --image path/to/leaf_photo.jpg
    python src/predict.py --model outputs/model.pt --image_dir path/to/folder_of_leaf_photos
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, img_size


def preprocess(image_path: str, img_size: int):
    tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return tfms(image).unsqueeze(0)


def predict_one(model, class_names, img_size, image_path, device, topk=3):
    tensor = preprocess(image_path, img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, min(topk, len(class_names)))
    results = [(class_names[i], float(p)) for p, i in zip(top_probs, top_idxs)]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained classifier.")
    parser.add_argument("--model", type=str, default="outputs/model.pt", help="Path to saved model.pt")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to a folder of images")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions to show")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide either --image or --image_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_model(args.model, device)

    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths += [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if f.lower().endswith(valid_ext)
        ]

    for path in image_paths:
        results = predict_one(model, class_names, img_size, path, device, topk=args.topk)
        print(f"\n{path}")
        for label, prob in results:
            print(f"  {label:<30s} {prob*100:5.1f}%")


if __name__ == "__main__":
    main()
