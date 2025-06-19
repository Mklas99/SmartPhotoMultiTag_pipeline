"""Data acquisition, filtering, splitting, export, and metadata utilties."""
from __future__ import annotations

import json, logging, datetime
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
from src.data.cocodataset import CocoDataset, collate_fn


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
    view = dataset.filter_labels("ground_truth", F("label").is_in(classes), only_matches=True)
    logger.info("View retains %d samples after label filtering", len(view))
    return view

# ----------------------------- Deterministic random split -----------------------------
def make_splits(dataset: fo.Dataset, seed: int = 42, label_classes = DEFAULT_CLASSES) -> Dict[str, fo.DatasetView]:
    """Tag samples train/val/test ≈ 70/20/10 and return views."""
    logger.info("Creating train/val/test tags (seed=%d)…", seed)
    random.seed(seed)
    
    # Proportions: Train 70%, Val 10%, Test 20%
    train_prop = 0.7
    val_prop = 0.1
    all_sample_ids = list(dataset.values("id"))

    # Clear existing split tags to ensure a clean split assignment
    for tag_to_clear in ("train", "val", "test"):
        view_with_tag = dataset.match_tags(tag_to_clear)
        if len(view_with_tag) > 0: # only untag if there are samples with this tag
                logger.info(f"Clearing existing tag '{tag_to_clear}' from {len(view_with_tag)} samples.")
                view_with_tag.untag_samples(tag_to_clear)

    train_ids_final = []
    val_ids_final = []
    test_ids_final = []
    assigned_ids = set()

    sorted_classes = sorted(list(set(label_classes))) # Use set to ensure unique classes before sorting
    logger.info(f"Stratifying across {len(sorted_classes)} classes: {sorted_classes}")

    for cls_name in sorted_classes:
        # Filter samples containing the current class
        # The F object comes from `from fiftyone import ViewField as F`
        class_view = dataset.filter_labels("ground_truth", F("label") == cls_name)
        ids_for_current_class = class_view.values("id")

        unassigned_for_cls = [sid for sid in ids_for_current_class if sid not in assigned_ids]
        random.shuffle(unassigned_for_cls)

        n_cls_unassigned = len(unassigned_for_cls)
        if n_cls_unassigned == 0:
            continue

        num_train_cls = int(train_prop * n_cls_unassigned)
        num_val_cls = int(val_prop * n_cls_unassigned)

        current_cls_train_ids = unassigned_for_cls[:num_train_cls]
        current_cls_val_ids = unassigned_for_cls[num_train_cls : num_train_cls + num_val_cls]
        current_cls_test_ids = unassigned_for_cls[num_train_cls + num_val_cls:]

        train_ids_final.extend(current_cls_train_ids)
        val_ids_final.extend(current_cls_val_ids)
        test_ids_final.extend(current_cls_test_ids)
        
        assigned_ids.update(unassigned_for_cls)

    # Handle samples not covered by the class-based stratification (e.g., no labels, or all labels processed)
    remaining_unassigned_ids = [sid for sid in all_sample_ids if sid not in assigned_ids]
    if remaining_unassigned_ids:
        logger.info(f"Distributing {len(remaining_unassigned_ids)} remaining samples randomly...")
        random.shuffle(remaining_unassigned_ids)
        
        n_remaining = len(remaining_unassigned_ids)
        num_train_rem = int(train_prop * n_remaining)
        num_val_rem = int(val_prop * n_remaining)

        train_ids_final.extend(remaining_unassigned_ids[:num_train_rem])
        val_ids_final.extend(remaining_unassigned_ids[num_train_rem : num_train_rem + num_val_rem])
        test_ids_final.extend(remaining_unassigned_ids[num_train_rem + num_val_rem:])
    
    # Using set to count unique IDs, though construction should ensure disjoint sets.
    logger.info(f"Final unique split ID counts - Train: {len(set(train_ids_final))}, Val: {len(set(val_ids_final))}, Test: {len(set(test_ids_final))}")
    
    # Apply tags to the sets of IDs. Set ensures uniqueness.
    dataset.select(list(set(train_ids_final))).tag_samples("train")
    dataset.select(list(set(val_ids_final))).tag_samples("val")
    dataset.select(list(set(test_ids_final))).tag_samples("test")

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
    classes: List[str] | None = DEFAULT_CLASSES,
    max_samples: int = 500,
    seed: int = 42,
) -> Dict[str, fo.DatasetView]:
    """Fully run acquisition → filter → split → export → metadata."""
    dataset = load_dataset(classes=classes, max_samples=max_samples, seed=seed)
    view = filter_categories(dataset, classes)
    views = make_splits(view._dataset, seed, classes)
    export_splits(views)
    write_metadata(dataset, views)
    return views

def load_data(
    classes: list[str] | None = DEFAULT_CLASSES,
    max_samples: int = IMAGE_CNT,
    batch_size: Optional[int] = BATCH_SIZE,
    num_workers: Optional[int] = NUM_WORKERS
) -> Tuple[CocoDataset, CocoDataset, DataLoader, DataLoader]:
    """Return train & val datasets and loaders using CocoDataset."""
    train_img_dir = DATASET_ROOT / "train" / "data"
    train_ann_file = DATASET_ROOT / "train" / "labels.json"
    val_img_dir = DATASET_ROOT / "val" / "data"
    val_ann_file = DATASET_ROOT / "val" / "labels.json"

    if not dataset_already_prepared(train_img_dir, train_ann_file, val_img_dir, val_ann_file, classes, max_samples):
        logger.info("Existing COCO export not sufficient. Running data preparation...")
        prepare_dataset(classes=DEFAULT_CLASSES, max_samples=IMAGE_CNT)
        logger.info("Data preparation complete.")

    train_dataset = CocoDataset(
        images_dir=str(train_img_dir),
        annotations_file=str(train_ann_file),
        transform=train_transforms,
        target_category_names=classes
    )
    val_dataset = CocoDataset(
        images_dir=str(val_img_dir),
        annotations_file=str(val_ann_file),
        transform=val_transforms,
        target_category_names=classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle for training
        num_workers=num_workers,
        collate_fn=collate_fn,
        #pin_memory=True, # pin_memory=True is useful for GPU training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        #pin_memory=True
    )
    return train_dataset, val_dataset, train_loader, val_loader

def dataset_already_prepared(train_img_dir, train_ann_file, val_img_dir, val_ann_file, classes, max_samples):
    # Check if the COCO export exists
    if not (train_img_dir.exists() and train_ann_file.exists() and val_img_dir.exists() and val_ann_file.exists()):
        logger.info("COCO export not found.")
        return False
    
    try:
        with open(train_ann_file, "r") as f:
            train_data = json.load(f)
        with open(val_ann_file, "r") as f:
            val_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading annotation files: {e}")
        return False
    
    # Check if the number of samples is sufficient
    total_samples = len(train_data.get("images", [])) + len(val_data.get("images", []))
    if max_samples is not None and total_samples < max_samples:
        logger.info(f"Dataset has only {total_samples} samples, less than requested {max_samples}.")
        return False

    # Check if the classes match
    if classes is not None:
        # Gather all unique category names from train and val annotation files
        train_categories = {cat["name"] for cat in train_data.get("categories", [])}
        val_categories = {cat["name"] for cat in val_data.get("categories", [])}

        if not set(classes).issubset(train_categories) or not set(classes).issubset(val_categories):
            logger.info(f"Dataset classes do not match requested classes {classes}.")
            return False

    return True