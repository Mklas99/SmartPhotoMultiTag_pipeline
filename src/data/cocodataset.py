import os
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from src import config 
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError


class CocoDataset(Dataset):
    """
    Dataset class for multi-label classification using COCO-format annotations.

    Each sample returns a tuple (image_tensor, multi_hot_label).
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transform: Optional[transforms.Compose] = None,
        target_category_names: Optional[List[str]] = None
    ):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.transform = transform
        self.target_category_names = target_category_names

        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotations_file}")

        with open(self.annotations_file, 'r') as f:
            data = json.load(f)

        self.valid_images = self._validate_images(data['images'])
        if not self.valid_images:
            raise RuntimeError("No valid images found in dataset.")

        (
            self.image_ids,
            self.id_to_filename,
            self.labels,
            self.category_names
        ) = self._parse_annotations(
            image_metadata_list=self.valid_images,
            categories=data['categories'],
            annotations=data.get('annotations', [])
        )

        if self.transform is None:
            if "train" in self.annotations_file.name.lower():
                self.transform = config.train_transforms
            else:
                self.transform = config.val_transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Optional[Tuple[Tensor, Tensor]]:
        img_id = self.image_ids[idx]
        img_path = self.images_dir / self.id_to_filename[img_id]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            warnings.warn(f"Skipping image {img_path}: {e}")
            return None

    def _validate_images(self, candidate_images: List[dict]) -> List[dict]:
        """
        Filters out non-existent or corrupted images from the candidate list.
        """
        valid_images = []
        for img_info in candidate_images:
            img_path = self.images_dir / img_info["file_name"]
            try:
                with Image.open(img_path) as img:
                    img.convert("RGB")
                valid_images.append(img_info)
            except Exception:
                warnings.warn(f"Invalid image skipped: {img_path}")
        return valid_images

    def _parse_annotations(
        self,
        image_metadata_list: List[dict],
        categories: List[dict],
        annotations: List[dict]
    ) -> Tuple[List[int], Dict[int, str], List[Tensor], List[str]]:

        # Category handling
        if self.target_category_names:
            selected_categories = [
                cat for cat in categories if cat['name'] in self.target_category_names
            ]
            selected_categories.sort(key=lambda x: self.target_category_names.index(x['name']))
        else:
            selected_categories = sorted(categories, key=lambda x: x['id'])

        category_names = [cat['name'] for cat in selected_categories]
        category_ids = [cat['id'] for cat in selected_categories]
        catid_to_index = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

        # Image ID to filename map
        id_to_filename = {img['id']: img['file_name'] for img in image_metadata_list}
        image_ids = list(id_to_filename.keys())

        # Build multi-label mappings
        img_to_cats = {img_id: [] for img_id in image_ids}
        for ann in annotations:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if img_id in img_to_cats and cat_id in catid_to_index:
                img_to_cats[img_id].append(cat_id)

        labels = []
        for img_id in image_ids:
            label_vec = torch.zeros(len(category_names), dtype=torch.float32)
            for cat_id in set(img_to_cats[img_id]):
                idx = catid_to_index[cat_id]
                label_vec[idx] = 1.0
            labels.append(label_vec)

        return image_ids, id_to_filename, labels, category_names


def collate_fn(batch: List[Optional[Tuple[Tensor, Tensor]]]) -> Optional[Tuple[Tensor, Tensor]]:
    """
    Collate function that filters out failed samples (None).
    Returns a valid batch or None if all failed.
    """
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        warnings.warn("Empty batch: all samples were invalid.")
        return None
    return torch.utils.data.dataloader.default_collate(batch)
