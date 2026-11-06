import numpy as np


def yolo_to_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized YOLO box [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_center(box_xyxy: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (box_xyxy[0] + box_xyxy[2]) / 2,
            (box_xyxy[1] + box_xyxy[3]) / 2,
        ],
        dtype=np.float32,
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def normalized_center_distance(
    box_a_xyxy: np.ndarray,
    box_b_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
) -> float:
    """Distance between centers, normalized by image diagonal."""
    ca = box_center(box_a_xyxy)
    cb = box_center(box_b_xyxy)
    dist = np.linalg.norm(ca - cb)
    diag = np.sqrt(image_width**2 + image_height**2)
    return float(dist / diag)