from pathlib import Path

import cv2
import numpy as np


def draw_pedestrian_risk(
    image_path: str,
    pedestrian_boxes: list,
    risk_scores: list,
    output_path: str,
):
    """Draw pedestrian-level risk scores on an image.

    risk_scores must be in [0, 1].
    """
    image = cv2.imread(image_path)
    overlay = image.copy()

    for box, score in zip(pedestrian_boxes, risk_scores):
        x1, y1, x2, y2 = [int(v) for v in box]

        if score >= 0.66:
            color = (0, 0, 255)
        elif score >= 0.33:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    alpha = 0.25
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blended)

    return str(output_path)