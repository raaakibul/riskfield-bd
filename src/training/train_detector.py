import argparse
from pathlib import Path

import yaml

from src.data.download_tfpbd import download_tfpbd
from src.models.detector import train_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not args.skip_download:
        dataset_root = download_tfpbd(
            workspace=cfg["dataset"]["roboflow_workspace"],
            project=cfg["dataset"]["roboflow_project"],
            version=cfg["dataset"]["roboflow_version"],
            export_format=cfg["dataset"]["export_format"],
            local_cache=cfg["dataset"]["local_cache"],
        )
    else:
        dataset_root = Path(cfg["dataset"]["local_cache"])

    train_detector(
        dataset_root=dataset_root,
        model_weights=cfg["detector"]["model"],
        epochs=cfg["detector"]["epochs"],
        batch_size=cfg["detector"]["batch_size"],
        image_size=cfg["dataset"]["image_size"],
        device=cfg["detector"]["device"],
    )


if __name__ == "__main__":
    main()