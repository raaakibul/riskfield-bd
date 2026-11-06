import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.geometry import box_iou, box_center, normalized_center_distance, yolo_to_xyxy


def build_risk_records(
    dataset_root: str | Path,
    class_names: List[str],
    pedestrian_class: str = "People",
    vehicle_classes: List[str] | None = None,
    large_vehicle_classes: List[str] | None = None,
    informal_vehicle_classes: List[str] | None = None,
    high_risk_distance_norm: float = 0.05,
    medium_risk_distance_norm: float = 0.12,
    overlap_threshold: float = 0.02,
) -> pd.DataFrame:
    """Create one row per pedestrian using weak supervision from YOLO boxes."""
    dataset_root = Path(dataset_root)
    vehicle_classes = vehicle_classes or class_names[1:]
    large_vehicle_classes = large_vehicle_classes or []
    informal_vehicle_classes = informal_vehicle_classes or []

    ped_id = class_names.index(pedestrian_class)
    vehicle_ids = {class_names.index(c) for c in vehicle_classes}
    large_ids = {class_names.index(c) for c in large_vehicle_classes}
    informal_ids = {class_names.index(c) for c in informal_vehicle_classes}

    records = []

    for split in ["train", "valid", "test"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        if not image_dir.exists():
            continue

        image_paths = sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
        )

        for image_path in tqdm(image_paths, desc=f"Processing {split}"):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            label_path = label_dir / (image_path.stem + ".txt")

            if not label_path.exists():
                continue

            boxes = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(float(parts[0]))
                    box_norm = np.array(parts[1:], dtype=np.float32)
                    box_pixel = yolo_to_xyxy(box_norm, width, height)
                    boxes.append((cls, box_pixel))

            pedestrians = [
                (cls, box)
                for cls, box in boxes
                if cls == ped_id
            ]
            vehicles = [
                (cls, box)
                for cls, box in boxes
                if cls in vehicle_ids
            ]

            for ped_cls, ped_box in pedestrians:
                if len(vehicles) == 0:
                    records.append(
                        {
                            "image_path": str(image_path),
                            "split": split,
                            "pedestrian_box": ped_box.tolist(),
                            "risk_label": 0,
                            "min_distance_norm": 1.0,
                            "max_iou": 0.0,
                            "num_vehicles_nearby": 0,
                            "num_large_vehicles_nearby": 0,
                            "num_informal_vehicles_nearby": 0,
                        }
                    )
                    continue

                distances = []
                ious = []
                nearby_large = 0
                nearby_informal = 0

                for veh_cls, veh_box in vehicles:
                    d = normalized_center_distance(ped_box, veh_box, width, height)
                    iou = box_iou(ped_box, veh_box)
                    distances.append(d)
                    ious.append(iou)

                    if d <= medium_risk_distance_norm:
                        if veh_cls in large_ids:
                            nearby_large += 1
                        if veh_cls in informal_ids:
                            nearby_informal += 1

                min_distance = min(distances)
                max_iou = max(ious)
                num_nearby = sum(d <= medium_risk_distance_norm for d in distances)

                if min_distance <= high_risk_distance_norm or max_iou >= overlap_threshold:
                    risk = 2
                elif min_distance <= medium_risk_distance_norm:
                    risk = 1
                else:
                    risk = 0

                records.append(
                    {
                        "image_path": str(image_path),
                        "split": split,
                        "pedestrian_box": ped_box.tolist(),
                        "risk_label": risk,
                        "min_distance_norm": min_distance,
                        "max_iou": max_iou,
                        "num_vehicles_nearby": num_nearby,
                        "num_large_vehicles_nearby": nearby_large,
                        "num_informal_vehicles_nearby": nearby_informal,
                    }
                )

    return pd.DataFrame(records)


def save_risk_records(df: pd.DataFrame, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} pedestrian-risk records to {output_csv}")