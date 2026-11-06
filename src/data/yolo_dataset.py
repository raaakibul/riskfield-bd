from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YoloDetectionDataset(Dataset):

    def __init__(self, root: str | Path, split: str = "train", img_size: int = 640):
        self.root = Path(root)
        self.split_dir = self.root / split
        self.image_dir = self.split_dir / "images"
        self.label_dir = self.split_dir / "labels"
        self.img_size = img_size

        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / (image_path.stem + ".txt")

    def load_labels(self, image_path: Path) -> np.ndarray:
        label_path = self._label_path(image_path)
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        rows = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                rows.append([float(x) for x in parts])

        return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(self.load_labels(image_path))

        return image_tensor, labels, str(image_path)