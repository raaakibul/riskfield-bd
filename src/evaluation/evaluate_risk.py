import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.risk_graph_dataset import PedestrianRiskGraphDataset
from src.models.risk_gnn import PedestrianRiskGNN
from src.training.train_risk import collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--risk-csv", default="outputs/data/pedestrian_risk_labels.csv")
    parser.add_argument("--weights", default="outputs/runs/risk_gnn/best_risk_gnn.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = PedestrianRiskGraphDataset(
        risk_csv=args.risk_csv,
        class_names=cfg["classes"],
        pedestrian_class=cfg["risk"]["pedestrian_class"],
        vehicle_classes=cfg["risk"]["vehicle_classes"],
        proximity_threshold_norm=cfg["risk"]["proximity_threshold_norm"],
        split="test",
    )

    loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = PedestrianRiskGNN(
        node_feature_dim=6,
        num_vehicle_classes=len(cfg["classes"]),
        hidden_dim=cfg["training"]["hidden_dim"],
        num_layers=cfg["training"]["num_layers"],
        dropout=cfg["training"]["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for crops, x, edge_index, vehicle_type_ids, y, batch_vec in loader:
            x, edge_index = x.to(device), edge_index.to(device)
            vehicle_type_ids = vehicle_type_ids.to(device)
            batch_vec = batch_vec.to(device)

            logits = model(x, edge_index, vehicle_type_ids, batch_vec)
            probs = torch.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)

    metrics = {
        "classification_report": classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }

    try:
        metrics["macro_auc_ovr"] = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["macro_auc_ovr"] = None

    output_path = Path("outputs/metrics/risk_gnn_test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()