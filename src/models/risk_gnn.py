from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class PedestrianRiskGNN(nn.Module):
    """Graph neural network for pedestrian-level conflict-risk prediction."""

    def __init__(
        self,
        node_feature_dim: int,
        num_vehicle_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()

        self.vehicle_embedding = nn.Embedding(num_vehicle_classes, 32)

        input_dim = node_feature_dim + 32

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.convs = nn.ModuleList(layers)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        vehicle_type_ids: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        veh_emb = self.vehicle_embedding(vehicle_type_ids)
        x = torch.cat([x, veh_emb], dim=-1)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = global_mean_pool(x, batch)
        logits = self.classifier(pooled)
        return logits