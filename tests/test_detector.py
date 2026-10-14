import csv
from pathlib import Path

from PIL import Image

from imgcompliance.detector import Detector, train


def test_train_predict_three_images(tmp_path: Path):
    rows = []
    for i, label in enumerate(("real", "ai_edited", "real")):
        filename = f"image_{i}.jpg"
        Image.new("RGB", (48, 48), color=(i * 70, 20, 120)).save(tmp_path / filename)
        rows.append({"filename": filename, "label": label, "source": "test", "prompt": ""})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = tmp_path / "detector.pt"
    detector = train(manifest, epochs=1, batch_size=3, device="cpu",
                     pretrained=False, output_path=checkpoint)
    assert checkpoint.exists()
    prediction = detector.predict(tmp_path / "image_0.jpg")
    assert 0 <= prediction["prob"] <= 1
    assert prediction["label"] in {"real", "ai"}
    assert prediction["model_version"]
    reloaded = Detector(checkpoint, device="cpu")
    assert abs(reloaded.predict(tmp_path / "image_0.jpg")["prob"] - prediction["prob"]) < 1e-5
