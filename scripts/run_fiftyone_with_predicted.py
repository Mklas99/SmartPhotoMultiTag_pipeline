import fiftyone as fo
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from src import train_transforms 

# --- Config ---
DATASET_NAME = "your-dataset-name"
LABELS = ["outdoor", "food", "friends", "event", "indoor", "selfie"]
MODEL_PATH = "model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# --- Transform (same as during training) ---
transform = train_transforms

# --- Load Dataset and Test Split ---
dataset = fo.load_dataset(DATASET_NAME)
test_view = dataset.match_tags("test")

# --- Predict and Annotate ---
for sample in tqdm(test_view):
    img = Image.open(sample.filepath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = model(img_tensor).squeeze()
        preds = (probs > 0.5).cpu().numpy()

    predicted_labels = [LABELS[i] for i, flag in enumerate(preds) if flag]

    sample["predictions"] = fo.Classifications(
        classifications=[fo.Classification(label=label) for label in predicted_labels]
    )
    sample.save()

# --- Launch FiftyOne App ---
session = fo.launch_app(view=test_view)
session.wait()
