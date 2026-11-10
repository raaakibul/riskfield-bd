from pathlib import Path

from ultralytics import YOLO


def train_detector(
    dataset_root: str | Path,
    model_weights: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "0",
    project_dir: str = "outputs/runs",
    run_name: str = "tfpbd_yolov8n",
):
    """Train a YOLOv8 detector on the runtime-downloaded TFP-BD export."""
    data_yaml = Path(dataset_root) / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Expected Roboflow YOLO export metadata at {data_yaml}. "
            "Check that the Roboflow export format is yolov8."
        )

    model = YOLO(model_weights)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=project_dir,
        name=run_name,
        patience=10,
        verbose=True,
    )

    return results