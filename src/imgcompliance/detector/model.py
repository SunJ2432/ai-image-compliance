"""DenseNet121 feature extractor with a trainable binary head.

Labels: real=0, ai_edited/ai_generated/fake=1. A tiny manifest is for
smoke-testing only, not measuring real-world detector accuracy.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MODEL_VERSION = "densenet121-frozen-head-v1"
POSITIVE = {"ai", "ai_edited", "ai_generated", "fake", "synthetic", "1"}
NEGATIVE = {"real", "authentic", "0"}


def _transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


class ManifestDataset(Dataset):
    def __init__(self, manifest: Union[str, Path]):
        self.manifest = Path(manifest)
        self.transform = _transform()
        with self.manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("Manifest is empty")
        self.items = []
        for row in rows:
            label = row["label"].strip().lower()
            if label not in POSITIVE | NEGATIVE:
                raise ValueError(f"Unknown label {label!r}")
            filename = Path(row["filename"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError(f"Unsafe image path: {filename}")
            path = self.manifest.parent / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            self.items.append((path, int(label in POSITIVE)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, torch.tensor(label, dtype=torch.float32)


class _BinaryNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=pretrained, num_classes=0,
            global_pool="avg"
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        # Backbone always in eval mode; no batchnorm statistics updates.
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features).flatten()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class Detector:
    def __init__(self, checkpoint: Union[str, Path, None] = None,
                 device: str | None = None, pretrained: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # No need to download ImageNet weights when loading a complete checkpoint.
        self.model = _BinaryNet(pretrained=pretrained if checkpoint is None else False).to(self.device)
        self.model.eval()
        self.model_version = MODEL_VERSION
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state.get("model_version") != MODEL_VERSION:
                raise ValueError("Incompatible model version")
            self.model.load_state_dict(state["state_dict"])
            self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: Union[str, Path, Image.Image]) -> dict:
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                tensor = _transform()(im.convert("RGB"))
        elif isinstance(image, Image.Image):
            tensor = _transform()(image.convert("RGB"))
        else:
            raise TypeError("image must be a path or PIL Image")
        self.model.eval()
        logit = self.model(tensor.unsqueeze(0).to(self.device))
        prob = float(torch.sigmoid(logit)[0].item())
        return {"prob": prob, "label": "ai" if prob >= 0.5 else "real",
                "model_version": self.model_version}

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_version": self.model_version,
                    "state_dict": self.model.state_dict()}, path)
        return path


def train(manifest: Union[str, Path], *, epochs: int = 3, batch_size: int = 6,
          lr: float = 1e-3, output_path: Union[str, Path] = "/content/detector_head.pt",
          device: str | None = None, pretrained: bool = True) -> Detector:
    """Train head only; save checkpoint to local storage.

    For Colab use pretrained=True. For offline unit tests use pretrained=False.
    Under-one-minute is a target for a warmed-up GPU and tiny dataset, not a guarantee.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    dataset = ManifestDataset(manifest)
    if not {label for _, label in dataset.items} == {0, 1}:
        raise ValueError("Training requires both real and AI labels")
    detector = Detector(device=device, pretrained=pretrained)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(detector.model.head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    detector.model.train()
    for epoch in range(epochs):
        total = 0.0
        for images, targets in loader:
            images = images.to(detector.device)
            targets = targets.to(detector.device)
            optimizer.zero_grad(set_to_none=True)
            logits = detector.model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
        print(f"epoch {epoch + 1}/{epochs} loss={total / len(dataset):.4f}")
    detector.model.eval()
    detector.save(output_path)
    return detector
