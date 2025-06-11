import fiftyone as fo
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.config import CHECKPOINT_DIR, DEFAULT_CLASSES, train_transforms 
import src.data.loader as data_loader

# --- Config ---
LABELS = DEFAULT_CLASSES
MODEL_PATH = CHECKPOINT_DIR / "best_epoch2.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# --- Transform (same as during training) ---
transform = train_transforms

# Load FiftyOne dataset
dataset = data_loader.load_fiftyone_dataset(split="test")
test_view = dataset

# --- Predict and Annotate ---
for sample in tqdm(test_view):
    img = Image.open(sample.filepath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = train_transforms(img_tensor).squeeze()
        preds = (probs > 0.5).cpu().numpy()

    predicted_labels = [LABELS[i] for i, flag in enumerate(preds) if flag]

    sample["predictions"] = fo.Classifications(
        classifications=[fo.Classification(label=label) for label in predicted_labels]
    )
    sample.save()

# --- Launch FiftyOne App ---
session = fo.launch_app(view=test_view)
session.wait()
