import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.risk_graph_dataset import PedestrianRiskGraphDataset
from src.models.risk_gnn import PedestrianRiskGNN


def collate_fn(batch):
    crops, xs, edge_indices, type_ids, ys = zip(*batch)

    batch_x = []
    batch_edge = []
    batch_type = []
    batch_y = []
    batch_vec = []

    node_offset = 0

    for crop, x, edge_index, vehicle_type_ids, y in batch:
        batch_x.append(x)
        batch_type.append(vehicle_type_ids)
        batch_y.append(y)

        batch_edge.append(edge_index + node_offset)
        batch_vec.append(torch.full((x.size(0),), len(batch_vec), dtype=torch.long))

        node_offset += x.size(0)

    return (
        torch.stack(crops),
        torch.cat(batch_x, dim=0),
        torch.cat(batch_edge, dim=1),
        torch.cat(batch_type, dim=0),
        torch.stack(batch_y),
        torch.cat(batch_vec, dim=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--risk-csv", default="outputs/data/pedestrian_risk_labels.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_dataset = PedestrianRiskGraphDataset(
        risk_csv=args.risk_csv,
        class_names=cfg["classes"],
        pedestrian_class=cfg["risk"]["pedestrian_class"],
        vehicle_classes=cfg["risk"]["vehicle_classes"],
        proximity_threshold_norm=cfg["risk"]["proximity_threshold_norm"],
        split="train",
    )

    val_dataset = PedestrianRiskGraphDataset(
        risk_csv=args.risk_csv,
        class_names=cfg["classes"],
        pedestrian_class=cfg["risk"]["pedestrian_class"],
        vehicle_classes=cfg["risk"]["vehicle_classes"],
        proximity_threshold_norm=cfg["risk"]["proximity_threshold_norm"],
        split="valid",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model = PedestrianRiskGNN(
        node_feature_dim=6,
        num_vehicle_classes=len(cfg["classes"]),
        hidden_dim=cfg["training"]["hidden_dim"],
        num_layers=cfg["training"]["num_layers"],
        dropout=cfg["training"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    class_counts = train_dataset.df["risk_label"].value_counts().sort_index()
    weights = class_counts.sum() / (len(class_counts) * class_counts + 1e-6)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights.values, dtype=torch.float32).to(device)
    )

    best_val_f1 = 0.0
    output_dir = Path("outputs/runs/risk_gnn")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        for crops, x, edge_index, vehicle_type_ids, y, batch_vec in tqdm(train_loader):
            x, edge_index = x.to(device), edge_index.to(device)
            vehicle_type_ids = vehicle_type_ids.to(device)
            y = y.to(device)
            batch_vec = batch_vec.to(device)

            optimizer.zero_grad()
            logits = model(x, edge_index, vehicle_type_ids, batch_vec)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for crops, x, edge_index, vehicle_type_ids, y, batch_vec in val_loader:
                x, edge_index = x.to(device), edge_index.to(device)
                vehicle_type_ids = vehicle_type_ids.to(device)
                y = y.to(device)
                batch_vec = batch_vec.to(device)

                logits = model(x, edge_index, vehicle_type_ids, batch_vec)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        report = classification_report(
            all_labels,
            all_preds,
            output_dict=True,
            zero_division=0,
        )

        val_f1 = report["macro avg"]["f1-score"]

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train loss: {total_loss / len(train_loader):.4f} | "
            f"Val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_risk_gnn.pt")

            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "val_macro_f1": val_f1,
                        "classification_report": report,
                        "confusion_matrix": confusion_matrix(
                            all_labels, all_preds
                        ).tolist(),
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()