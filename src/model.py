import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear

class HeteroGCN(torch.nn.Module):
    """
    Heterogeneous Graph Convolutional Network.
    Uses SAGEConv layers for message passing across different edge types.
    """
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.metadata = metadata
        
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        })
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in metadata[1]
        })
        
        # Learnable embeddings for nodes if they don't have features
        self.drug_emb = torch.nn.Embedding(100000, hidden_channels) 
        self.disease_emb = torch.nn.Embedding(100000, hidden_channels)
        self.protein_emb = torch.nn.Embedding(100000, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict

class LinkPredictor(torch.nn.Module):
    """
    Link Prediction head using a linear layer on concatenated node embeddings.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.lin = Linear(in_channels * 2, 1)

    def forward(self, x_drug, x_disease, edge_label_index):
        row, col = edge_label_index
        
        drug_feat = x_drug[row]
        disease_feat = x_disease[col]
        
        # Concatenate and predict
        x = torch.cat([drug_feat, disease_feat], dim=-1)
        return self.lin(x).squeeze()

class Model(torch.nn.Module):
    """
    End-to-end model combining HeteroGCN encoder and LinkPredictor decoder.
    """
    def __init__(self, hidden_channels, metadata, num_nodes_dict):
        super().__init__()
        self.encoder = HeteroGCN(hidden_channels, hidden_channels, metadata)
        self.decoder = LinkPredictor(hidden_channels)
        
        # Initialize embeddings
        self.emb_dict = torch.nn.ParameterDict()
        for node_type, num_nodes in num_nodes_dict.items():
            self.emb_dict[node_type] = torch.nn.Parameter(torch.randn(num_nodes, hidden_channels))

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        if not x_dict:
            x_dict = self.emb_dict
            
        z_dict = self.encoder(x_dict, edge_index_dict)
        
        return self.decoder(z_dict['drug'], z_dict['disease'], edge_label_index)
