import os
from re import A
import sys

import fiftyone.types as fot
from pathlib import Path  # <-- Add this import

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import fiftyone as fo
import torch
from PIL import Image
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DEFAULT_CLASSES, ModelConfig, train_transforms
from src.models.PhotoTagNet_model import PhotoTagNet

# --- Config ---
LABELS = DEFAULT_CLASSES
TEST_FILES_DIR = "src/data/coco/test"
ASSETS_DIR = Path("assets")  # <-- Convert to Path
MODEL_PATH = ASSETS_DIR / "best_model_PhotoNet_10000.pt"
CHECKPOINT_DIR = Path(CHECKPOINT_DIR)  # <-- Ensure CHECKPOINT_DIR is a Path
MODEL_PATH2 = CHECKPOINT_DIR / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint  # fallback if it's a plain state_dict

model = PhotoTagNet(ModelConfig(), num_classes=len(LABELS))
model.load_state_dict(state_dict)
model.eval()

# --- Transform (same as during training) ---
transform = train_transforms

# Load FiftyOne dataset
print("Loading about 100 images...")
dataset = fo.Dataset.from_dir(
    dataset_dir=TEST_FILES_DIR,
    dataset_type=fot.COCODetectionDataset,
    classes=LABELS,
)

# Limit to 100 samples
if len(dataset) > 100:
    dataset = dataset.take(500)


ACTUAL_COLOR = "#3cb44b"           # green for actual-only
ACTUAL_PREDICTED_COLOR = "#ffa500" # orange for actual+predicted
PREDICTED_COLOR = "#e6194b"        # red for predicted

LABEL_COLORS = {label: ACTUAL_COLOR for label in LABELS}
PREDICTED_LABEL_COLORS = {label: PREDICTED_COLOR for label in LABELS}

dataset.default_classes = LABELS
dataset.classes = {
    "detections": LABELS,
    "predictions(PhotoNet_10000)": LABELS,
}

# --- Predict and Annotate ---
print("predicting labels...")
for sample in tqdm(dataset):
    img = Image.open(sample.filepath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).cpu().numpy()

    predicted_labels = [LABELS[i] for i, flag in enumerate(preds) if flag]

    # Save predicted labels as a list for easier filtering
    sample["predicted_labels"] = predicted_labels

    # Get actual labels from detections
    actual_labels = set()
    if sample.detections is not None:
        for det in sample.detections.detections:
            actual_labels.add(det.label)

    # Save actual labels as a list for easier filtering/viewing
    sample["actual_labels"] = list(actual_labels)

    # Assign colors for actual labels
    actual_classifications = []
    for label in actual_labels:
        color = ACTUAL_PREDICTED_COLOR if label in predicted_labels else ACTUAL_COLOR
        actual_classifications.append(fo.Classification(label=label, color=color))

    # Assign colors for predicted labels (always red)
    predicted_classifications = [
        fo.Classification(label=label, color=PREDICTED_COLOR)
        for label in predicted_labels
    ]

    # Only set the predictions field; do not overwrite detections
    sample["predictions(PhotoNet_10000)"] = fo.Classifications(classifications=predicted_classifications)
    sample.save()

# Build a pretty view: show filepath, detections, predictions, and predicted_labels
test_view = (
    dataset
    .select_fields([
        "filepath",
        "detections",
        "actual_labels",           # <-- add this line
        "predictions(PhotoNet_10000)",
        "predicted_labels"
    ])
    .sort_by("filepath")
)

# --- Launch FiftyOne App ---
print("Launch FifyOne...")
session = fo.launch_app(view=test_view)
session.wait()
