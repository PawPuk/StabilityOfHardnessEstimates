"""Wrappers for datasets that allowing mapping between"""

import os
from typing import Tuple, Optional, Callable
import glob
import shutil

import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset

from src.utils.download_utils import download_url, extract_archive


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,
            augmentation_applied=True,
            transform=None
    ):
        # To improve speed, we transform the dataset into a TensorDataset (only viable if no augmentation is applied)
        if not isinstance(dataset, TensorDataset) and not augmentation_applied:
            data_list, label_list = [], []
            for i in range(len(dataset)):
                data, label = dataset[i]
                data_list.append(data)
                label_list.append(torch.tensor(label))  # Necessary because some datasets return labels as integers
            data_tensor = torch.stack(data_list)
            label_tensor = torch.tensor(label_list)
            dataset = TensorDataset(data_tensor, label_tensor)

        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class TinyImageNet(Dataset):
    """TinyImageNet dataset - only training split.

    The dataset structure:
    - Downloaded as a zip file from Stanford CS231N page
    - Contains train/val/test folders (we only use train)
    - Classes are organized by WordNet IDs (e.g., n01443537 = "goldfish")

    Args:
        root (str): Root directory for dataset
        transform (callable, optional): Optional transform to apply
        download (bool): If True, download dataset if not already present
    """

    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.dataset_path = os.path.join(self.root, 'tiny-imagenet-200')

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # Load class mapping (WordNet ID -> class index)
        self.classes = sorted(os.listdir(os.path.join(self.dataset_path, 'train')))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load training data
        self.data, self.targets = self._load_data()

    def _check_exists(self):
        if not os.path.exists(self.dataset_path):
            return False

        train_path = os.path.join(self.dataset_path, 'train')
        if not os.path.exists(train_path):
            return False

        class_folders = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        if len(class_folders) != 200:
            return False

        for class_name in class_folders:
            images_dir = os.path.join(train_path, class_name, 'images')
            if not os.path.exists(images_dir):
                return False
            num_images = len(glob.glob(os.path.join(images_dir, '*.JPEG')))
            if num_images != 500:
                return False

        return True

    def _download(self):
        """Download and extract TinyImageNet if not already present."""
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, 'tiny-imagenet-200.zip')

        if not os.path.exists(zip_path):
            download_url(self.URL, zip_path)
        extract_archive(zip_path, self.root)

        val_path = os.path.join(self.dataset_path, 'val')
        test_path = os.path.join(self.dataset_path, 'test')
        for path in [val_path, test_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    def _load_data(self):
        """Load training data from folder structure."""
        data, targets = [], []
        train_dir = os.path.join(self.dataset_path, 'train')

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(train_dir, class_name, 'images')
            image_paths = sorted(glob.glob(os.path.join(class_dir, '*.JPEG')))
            for img_path in image_paths:
                data.append(img_path)
                targets.append(class_idx)

        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data[idx]
        label = self.targets[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
