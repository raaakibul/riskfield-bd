from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.geometry import box_iou, box_center, normalized_center_distance, yolo_to_xyxy


class PedestrianRiskGraphDataset(Dataset):
    """Builds one pedestrian-centered graph per sample.

    Node 0 is the pedestrian.
    Remaining nodes are nearby vehicles.
    """

    def __init__(
        self,
        risk_csv: str | Path,
        class_names: List[str],
        pedestrian_class: str = "People",
        vehicle_classes: List[str] | None = None,
        proximity_threshold_norm: float = 0.12,
        image_size: int = 224,
        split: str = "train",
    ):
        self.df = pd.read_csv(risk_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.class_names = class_names
        self.ped_id = class_names.index(pedestrian_class)
        self.vehicle_ids = [class_names.index(c) for c in (vehicle_classes or class_names[1:])]
        self.proximity_threshold_norm = proximity_threshold_norm
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_image(row["image_path"])
        height, width = image.shape[:2]

        ped_box = np.array(row["pedestrian_box"], dtype=np.float32)

        # Crop pedestrian region with context padding.
        x1, y1, x2, y2 = ped_box
        pad_x = 0.25 * (x2 - x1)
        pad_y = 0.25 * (y2 - y1)
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.image_size, self.image_size))
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        # Parse all boxes again to construct the local graph.
        label_path = (
            Path(row["image_path"])
            .parents[1]
            / "labels"
            / (Path(row["image_path"]).stem + ".txt")
        )

        nodes = []
        vehicle_type_ids = []
        edges = []

        ped_center = box_center(ped_box)

        ped_features = np.array(
            [
                1.0,  # is pedestrian
                0.0,  # is vehicle
                0.0,  # normalized distance to focal pedestrian
                0.0,  # IoU with focal pedestrian
                0.0,  # large vehicle flag
                0.0,  # informal vehicle flag
            ],
            dtype=np.float32,
        )

        nodes.append(ped_features)
        vehicle_type_ids.append(0)

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls = int(float(parts[0]))
                    if cls not in self.vehicle_ids:
                        continue

                    box = yolo_to_xyxy(
                        np.array(parts[1:], dtype=np.float32), width, height
                    )

                    dist = normalized_center_distance(ped_box, box, width, height)
                    if dist > self.proximity_threshold_norm:
                        continue

                    iou = box_iou(ped_box, box)
                    is_large = 1.0 if self.class_names[cls] in {"Bus", "Truck", "Mini-Truck"} else 0.0
                    is_informal = 1.0 if self.class_names[cls] in {"Rickshaw", "CNG"} else 0.0

                    node_feat = np.array(
                        [0.0, 1.0, dist, iou, is_large, is_informal],
                        dtype=np.float32,
                    )

                    vehicle_node_id = len(nodes)
                    nodes.append(node_feat)
                    vehicle_type_ids.append(cls)

                    edges.append([0, vehicle_node_id])
                    edges.append([vehicle_node_id, 0])

        x = torch.from_numpy(np.stack(nodes))
        vehicle_type_ids = torch.tensor(vehicle_type_ids, dtype=torch.long)

        if len(edges) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        y = torch.tensor(int(row["risk_label"]), dtype=torch.long)

        return crop_tensor, x, edge_index, vehicle_type_ids, y