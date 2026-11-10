import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.risk_labels import build_risk_records, save_risk_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default="outputs/data/pedestrian_risk_labels.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = args.dataset_root or cfg["dataset"]["local_cache"]

    df = build_risk_records(
        dataset_root=dataset_root,
        class_names=cfg["classes"],
        pedestrian_class=cfg["risk"]["pedestrian_class"],
        vehicle_classes=cfg["risk"]["vehicle_classes"],
        large_vehicle_classes=cfg["risk"]["large_vehicle_classes"],
        informal_vehicle_classes=cfg["risk"]["informal_vehicle_classes"],
        high_risk_distance_norm=cfg["risk"]["high_risk_distance_norm"],
        medium_risk_distance_norm=cfg["risk"]["medium_risk_distance_norm"],
        overlap_threshold=cfg["risk"]["overlap_threshold"],
    )

    save_risk_records(df, args.output)


if __name__ == "__main__":
    main()