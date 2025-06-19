import os
import sys

import fiftyone.types as fot

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
MODEL_PATH = CHECKPOINT_DIR / "final_model_notebook.pth"
MODEL_PATH2 = CHECKPOINT_DIR / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
model = PhotoTagNet(ModelConfig(), num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Transform (same as during training) ---
transform = train_transforms

# Load FiftyOne dataset
print("Launch FifyOne...")
dataset = fo.Dataset.from_dir(
    dataset_dir=TEST_FILES_DIR,
    dataset_type=fot.COCODetectionDataset,
)

test_view = dataset.view()

# --- Predict and Annotate ---
print("predicting labels...")
for sample in tqdm(test_view):
    img = Image.open(sample.filepath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).cpu().numpy()

    predicted_labels = [LABELS[i] for i, flag in enumerate(preds) if flag]

    sample["predictions"] = fo.Classifications(classifications=[fo.Classification(label=label) for label in predicted_labels])
    sample.save()

# --- Launch FiftyOne App ---
print("Launch FifyOne...")
session = fo.launch_app(view=test_view)
session.wait()
