"""Data acquisition, filtering, splitting, export, and metadata utilties."""
from __future__ import annotations

import json, yaml, logging, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot
from fiftyone import ViewField as F
from torch.utils.data import DataLoader

from src.config import (
    DEFAULT_CLASSES, DATASET_DIR, DATASET_ROOT, META_PATH, IMAGE_CNT,
    BATCH_SIZE, NUM_WORKERS, train_transforms, val_transforms
)
from .cocodataset import CocoDataset, collate_fn


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Download / load dataset -----------------------------
def load_dataset(
    dataset_name: str = "coco-2017",
    splits: Tuple[str, ...] = ("train", "validation", "test"),
    classes: List[str] | None = None,
    max_samples: int = IMAGE_CNT,
    seed: int = 42,
    shuffle: bool = True,
    dataset_instance_name: str | None = None,
) -> fo.Dataset:
    """Download (if needed) and load a subset of COCO‑2017 via FiftyOne.

    Parameters
    ----------
    dataset_name: str
        Name from FiftyOne Zoo (default ``"coco-2017"``).
    splits: tuple[str, ...]
        Which splits to pull; defaults to the canonical 3‑way.
    classes: list[str] | None
        Filter by these label classes *during* load.  If None, keep all.
    max_samples: int
        Maximum number of samples per split ("0" ⇒ no limit).
    seed: int, shuffle: bool
        Reproducibility controls.
    dataset_instance_name: str | None
        Registered name inside FiftyOne; if None, a unique name is generated.

    Returns
    -------
    fo.Dataset
        FiftyOne in‑memory dataset containing the selected samples.
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    if dataset_instance_name is None:
        dataset_instance_name = f"{dataset_name}-{max_samples}-seed{seed}"

    logger.info("Downloading splits %s from %s (max %s samples each)…", splits, dataset_name, max_samples)

    dataset = foz.load_zoo_dataset(
        dataset_name,
        splits=splits,
        label_types=["detections"],  # Keep boxes; segmentation masks optional
        classes=classes,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed
    )

    logger.info("Loaded %d samples in total", len(dataset))
    return dataset

# ----------------------------- Filter labels -----------------------------
def filter_categories(dataset: fo.Dataset, classes: List[str]) -> fo.DatasetView:
    """Return a view containing only labels whose class is in *classes*."""
    logger.info("Filtering dataset to %d target classes", len(classes))
    view = dataset.filter_labels("ground_truth", F("label").is_in(classes))
    logger.info("View retains %d samples after label filtering", len(view))
    return view

# ----------------------------- Deterministic random split -----------------------------
def make_splits(dataset: fo.Dataset, seed: int = 42) -> Dict[str, fo.DatasetView]:
    """Tag samples train/val/test ≈ 70/20/10 and return views."""
    logger.info("Creating train/val/test tags (seed=%d)…", seed)
    
    random.seed(seed)
    
    # Get all sample IDs and shuffle them
    sample_ids = list(dataset.values("id"))
    random.shuffle(sample_ids)
    
    # Split according to proportions
    n = len(sample_ids)
    train_size = int(0.7 * n)
    val_size = int(0.1 * n)
    
    # Create views based on sample IDs
    train_ids = sample_ids[:train_size]
    val_ids = sample_ids[train_size:train_size+val_size]
    test_ids = sample_ids[train_size+val_size:]
    
    # Apply tags
    dataset.select(train_ids).tag_samples("train")
    dataset.select(val_ids).tag_samples("val")
    dataset.select(test_ids).tag_samples("test")
    
    return {tag: dataset.match_tags(tag) for tag in ("train", "val", "test")}

# ----------------------------- COCO‑format export -----------------------------
def export_splits(views: Dict[str, fo.DatasetView], export_root: Path = DATASET_ROOT) -> None:
    export_root.mkdir(parents=True, exist_ok=True)
    for split, view in views.items():
        out_dir = export_root / split
        logger.info("Exporting %s split → %s (COCO) …", split, out_dir)
        view.export(
            export_dir=str(out_dir),
            dataset_type=fot.COCODetectionDataset,
            label_field="ground_truth",
            export_media=True,
            overwrite=True,
        )

# ----------------------------- Metadata serialisation -----------------------------
def write_metadata(dataset: fo.Dataset, views: Dict[str, fo.DatasetView], path: Path = META_PATH) -> None:
    # Ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    distinct_classes = dataset.distinct("ground_truth.detections.label")
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "total_images": len(dataset),
        "split_sizes": {k: len(v) for k, v in views.items()},
        "all_classes": distinct_classes,
        "num_all_classes": len(distinct_classes),
        "classes": DEFAULT_CLASSES,
        "num_classes": len(DEFAULT_CLASSES)
    }
    path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved → %s", path)

# ----------------------------- Convenience one‑shot helper -----------------------------
def prepare_dataset(
    classes: List[str] | None = None,
    max_samples: int = 500,
    seed: int = 42,
) -> Dict[str, fo.DatasetView]:
    """Fully run acquisition → filter → split → export → metadata."""
    dataset = load_dataset(classes=classes, max_samples=max_samples, seed=seed)
    view = filter_categories(dataset, classes or DEFAULT_CLASSES)
    views = make_splits(view._dataset, seed=seed)
    export_splits(views)
    write_metadata(dataset, views)
    return views

def load_data(batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Return train & val loaders using CocoDataset."""
    train_img_dir = DATASET_ROOT / "train" / "data"
    train_ann_file = DATASET_ROOT / "train" / "labels.json"
    val_img_dir = DATASET_ROOT / "val" / "data"
    val_ann_file = DATASET_ROOT / "val" / "labels.json"

    if not (train_img_dir.exists() and train_ann_file.exists() and \
            val_img_dir.exists() and val_ann_file.exists()):
        logger.info("COCO export not found. Running data preparation...")
        prepare_dataset(classes=DEFAULT_CLASSES, max_samples=IMAGE_CNT) # Use IMAGE_CNT for consistency if needed for quick setup
        logger.info("Data preparation complete.")

    # Use provided batch_size and num_workers, or fallback to config defaults
    current_batch_size = batch_size if batch_size is not None else BATCH_SIZE
    current_num_workers = num_workers if num_workers is not None else NUM_WORKERS

    target_classes = DEFAULT_CLASSES
    train_dataset = CocoDataset(
        images_dir=str(train_img_dir),
        annotations_file=str(train_ann_file),
        transform=train_transforms,
        target_category_names=target_classes
    )
    val_dataset = CocoDataset(
        images_dir=str(val_img_dir),
        annotations_file=str(val_ann_file),
        transform=val_transforms,
        target_category_names=target_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=current_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=current_batch_size,
        shuffle=False,
        num_workers=current_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return train_loader, val_loader
