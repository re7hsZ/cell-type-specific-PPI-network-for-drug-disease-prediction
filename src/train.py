import torch
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
from data_loader import DataLoader
from model import Model
import os
import argparse
import json

def train():
    parser = argparse.ArgumentParser(description='Train Cell-Type Specific PPI Network')
    parser.add_argument('--celltype', type=str, default='acinar_cell_of_salivary_gland', 
                        help='Name of the cell type file (without .txt extension)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_global_ppi', action='store_true', help='Use Global PPI from PrimeKG instead of cell-type specific')
    parser.add_argument('--save_results', type=str, default=None, help='Path to save evaluation results (JSON)')
    args = parser.parse_args()

    # Paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primekg_path = os.path.join(base_path, "data/raw/PrimeKG/kg.csv")
    
    # Construct cell type path
    celltype_filename = f"{args.celltype}.txt"
    celltype_ppi_path = os.path.join(base_path, "data/raw/celltype_ppi/22708126/networks/networks/ppi_edgelists", celltype_filename)
    
    if not args.use_global_ppi and not os.path.exists(celltype_ppi_path):
        print(f"Error: Cell type file not found at {celltype_ppi_path}")
        return

    if args.use_global_ppi:
        print("Training with Global PPI Network")
    else:
        print(f"Training for cell type: {args.celltype}")
    
    # Load Data
    loader = DataLoader(primekg_path, celltype_ppi_path, use_global_ppi=args.use_global_ppi)
    loader.load_data()
    data, label_edge_index = loader.build_graph()
    
    # Add labels to data for splitting
    data['drug', 'indication', 'disease'].edge_index = label_edge_index
    
    # Split
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False, 
        edge_types=[('drug', 'indication', 'disease')],
        rev_edge_types=[('disease', 'indicated_by', 'drug')], 
        add_negative_train_samples=True
    )
    
    train_data, val_data, test_data = transform(data)
    
    # Model
    hidden_channels = 64
    num_nodes_dict = {
        'drug': data['drug'].num_nodes,
        'disease': data['disease'].num_nodes,
        'protein': data['protein'].num_nodes
    }
    model = Model(hidden_channels, data.metadata(), num_nodes_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training Loop
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        edge_label_index = train_data['drug', 'indication', 'disease'].edge_label_index
        edge_label = train_data['drug', 'indication', 'disease'].edge_label
        
        out = model(None, train_data.edge_index_dict, edge_label_index)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        if (epoch + 1) % 5 == 0:
            evaluate(model, val_data, "Validation")

    print("Training finished.")
    metrics = evaluate(model, test_data, "Test")
    
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved to {args.save_results}")

def evaluate(model, data, stage):
    model.eval()
    with torch.no_grad():
        edge_label_index = data['drug', 'indication', 'disease'].edge_label_index
        edge_label = data['drug', 'indication', 'disease'].edge_label
        
        out = model(None, data.edge_index_dict, edge_label_index)
        pred = torch.sigmoid(out).cpu().numpy()
        y_true = edge_label.cpu().numpy()
        
        # Metrics
        auroc = roc_auc_score(y_true, pred)
        auprc = average_precision_score(y_true, pred)
        
        # F1 Score
        pred_binary = (pred > 0.5).astype(int)
        f1 = f1_score(y_true, pred_binary)
        
        # MRR
        pos_scores = pred[y_true == 1]
        neg_scores = pred[y_true == 0]
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            pos_scores_v = torch.tensor(pos_scores).view(-1, 1)
            neg_scores_v = torch.tensor(neg_scores).view(1, -1)
            greater = (neg_scores_v > pos_scores_v).float()
            rank = 1 + greater.sum(dim=1)
            mrr = (1.0 / rank).mean().item()
        else:
            mrr = 0.0
        
        print(f"[{stage}] AuROC: {auroc:.4f}, AuPRC: {auprc:.4f}, F1: {f1:.4f}, MRR: {mrr:.4f}")
        
        return {
            "AuROC": auroc,
            "AuPRC": auprc,
            "F1": f1,
            "MRR": mrr
        }

if __name__ == "__main__":
    train()
