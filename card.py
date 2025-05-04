import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GINEConv
import torch.nn.functional as F
import torch

class HandGNN(nn.Module):
    def __init__(self,
                 rank_embedding_dim=8,
                 suit_embedding_dim=8,
                 hidden_dim=16,
                 edge_attr_dim=2,
                 node_mlp_layers=2,
                 gnn_layers=2,
                 reduction='mean'):
        super().__init__()
        self.reduction = reduction

        self.rank_embedder = nn.Embedding(13, rank_embedding_dim)
        self.suit_embedder = nn.Embedding(4, suit_embedding_dim)
        layers = []
        for i in range(node_mlp_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i < node_mlp_layers-1:
                layers.append(nn.ReLU())
        self.node_mlp_layers = nn.Sequential(*layers)

        self.card_emb_projector = nn.Linear(rank_embedding_dim+suit_embedding_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_layers):
            self.gnn_layers.append(GINEConv(nn=self.node_mlp_layers, edge_dim=edge_attr_dim))

    def forward(self, data):
        rank_emb = self.rank_embedder(data.x[:, 0])
        suit_emb = self.suit_embedder(data.x[:, 1])
        card_emb = torch.cat([rank_emb, suit_emb], dim=1)
        x = self.card_emb_projector(card_emb)
        for conv in self.gnn_layers:
            x = conv(x, data.edge_index, data.edge_attr)
            x = F.relu(x)
        x = tg.utils.scatter(x, data.batch, dim=0, reduce=self.reduction)
        return x

class HandClassifier(nn.Module):
    def __init__(self,
                 rank_embedding_dim=8,
                 suit_embedding_dim=8,
                 hidden_dim=16,
                 edge_attr_dim=2,
                 node_mlp_layers=2,
                 gnn_layers=2,
                 reduction='mean',
                 out_dim=16):
        super().__init__()
        self.hand_encoder = HandGNN(
            rank_embedding_dim=rank_embedding_dim, suit_embedding_dim=suit_embedding_dim, hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim, node_mlp_layers=node_mlp_layers,
            gnn_layers=gnn_layers, reduction=reduction)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.output_layer = nn.Linear(out_dim, 10)

    def forward(self, data):
        x = self.hand_encoder(data)
        x = self.final(x)
        return self.output_layer(x)
