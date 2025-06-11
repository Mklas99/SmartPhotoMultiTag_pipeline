import pytest
import torch
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import warnings
from ..src.data.cocodataset import CocoDataset

# Relative import for CocoDataset, assuming test_cocodataset.py is in src/data/

# Helper function to create a dummy annotation file
def _create_dummy_annotations_file(filepath: Path, images_data: list, annotations_data: list, categories_data: list):
    content = {
        "images": images_data,
        "annotations": annotations_data,
        "categories": categories_data,
    }
    with open(filepath, 'w') as f:
        json.dump(content, f)

# Helper function to create a dummy image file
def _create_dummy_image(filepath: Path, size: tuple = (10, 10), color: str = 'red'):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.new('RGB', size, color=color)
        img.save(filepath)
    except Exception as e:
        warnings.warn(f"Could not create dummy image {filepath}: {e}")
        # Create a placeholder if image creation fails, to ensure a file exists
        filepath.write_text("dummy content, not a valid image")


@pytest.fixture
def coco_test_setup(tmp_path: Path) -> tuple[str, str]:
    """
    Sets up a temporary directory with dummy COCO annotations and images.
    Returns a tuple of (images_dir_path, annotations_file_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations_file = tmp_path / "annotations.json"

    images_data = [
        {"id": 1, "file_name": "img1.png"},
        {"id": 2, "file_name": "img2.jpg"},
        {"id": 3, "file_name": "img3.png"}, # For multi-label test
    ]
    # Categories sorted by ID: catA (id=10, index 0), catB (id=20, index 1)
    categories_data = [
        {"id": 10, "name": "catA"},
        {"id": 20, "name": "catB"},
    ]
    annotations_data = [
        {"image_id": 1, "category_id": 10},          # img1 -> catA
        {"image_id": 2, "category_id": 20},          # img2 -> catB
        {"image_id": 3, "category_id": 10},          # img3 -> catA
        {"image_id": 3, "category_id": 20},          # img3 -> catB (multi-label)
    ]

    _create_dummy_annotations_file(annotations_file, images_data, annotations_data, categories_data)
    
    _create_dummy_image(images_dir / "img1.png", color='red')
    _create_dummy_image(images_dir / "img2.jpg", color='blue')
    _create_dummy_image(images_dir / "img3.png", color='green')
    
    # Note: The CocoDataset.__init__ contains a line in its validate_images method:
    # img_path = self.images_dir / img_info["file_name"]
    # If self.images_dir is a string (as per __init__ type hint), this will cause a TypeError.
    # For these tests to pass, that line should be Path(self.images_dir) / img_info["file_name"]
    # or self.images_dir should be Path. We proceed assuming __init__ completes successfully.

    return str(images_dir), str(annotations_file)


def test_getitem_valid_index_and_labels(coco_test_setup: tuple[str, str]):
    """
    Tests __getitem__ for valid indices, checking returned image and label tensors.
    """
    images_dir, annotations_file = coco_test_setup
    simple_transform = transforms.Compose([
        transforms.Resize((10, 10)), # Ensure consistent image size
        transforms.ToTensor()
    ])
    
    dataset = CocoDataset(images_dir=images_dir, annotations_file=annotations_file, transform=simple_transform)

    assert len(dataset) == 3, "Dataset should have 3 items based on setup."

    # Expected labels are [catA, catB]
    # img1 (id=1) has catA: [1.0, 0.0]
    # img2 (id=2) has catB: [0.0, 1.0]
    # img3 (id=3) has catA, catB: [1.0, 1.0]
    expected_labels = [
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    ]

    for i in range(len(dataset)):
        img, label = dataset[i]
        assert isinstance(img, torch.Tensor), f"Image at index {i} should be a Tensor."
        assert img.shape == (3, 10, 10), f"Image at index {i} has unexpected shape."
        assert isinstance(label, torch.Tensor), f"Label at index {i} should be a Tensor."
        assert torch.equal(label, expected_labels[i]), f"Label at index {i} is incorrect."


def test_getitem_when_image_transform_returns_none(coco_test_setup: tuple[str, str]):
    """
    Tests __getitem__ when an image transform results in a None image.
    It should return (None, correct_label).
    """
    images_dir, annotations_file = coco_test_setup

    class TransformReturningNoneForFirstImage:
        def __init__(self):
            self.call_count = 0
        
        def __call__(self, pil_img):
            self.call_count += 1
            if self.call_count == 1: # Affects the first image processed by __init__
                return None
            # Standard transform for subsequent images
            return transforms.Compose([transforms.Resize((10,10)), transforms.ToTensor()])(pil_img)

    dataset = CocoDataset(
        images_dir=images_dir, 
        annotations_file=annotations_file, 
        transform=TransformReturningNoneForFirstImage()
    )
    
    assert len(dataset) == 3

    # First image's transform should have returned None
    img1, label1 = dataset[0]
    assert img1 is None, "Image for the first item should be None due to transform."
    assert isinstance(label1, torch.Tensor), "Label for the first item should still be a Tensor."
    expected_label1 = torch.tensor([1.0, 0.0], dtype=torch.float32) # Corresponds to img1 (catA)
    assert torch.equal(label1, expected_label1), "Label for the first item is incorrect."

    # Second image should be processed normally
    img2, label2 = dataset[1]
    assert isinstance(img2, torch.Tensor), "Image for the second item should be a Tensor."
    assert img2.shape == (3, 10, 10), "Image for the second item has unexpected shape."
    expected_label2 = torch.tensor([0.0, 1.0], dtype=torch.float32) # Corresponds to img2 (catB)
    assert torch.equal(label2, expected_label2), "Label for the second item is incorrect."