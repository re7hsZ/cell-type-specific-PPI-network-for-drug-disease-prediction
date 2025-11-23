import pandas as pd
import os
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

class DataLoader:
    """
    Handles loading and processing of PrimeKG and cell-type specific PPI networks.
    Constructs a HeteroData object for GNN training.
    """
    def __init__(self, primekg_path, celltype_ppi_path, use_global_ppi=False):
        self.primekg_path = primekg_path
        self.celltype_ppi_path = celltype_ppi_path
        self.use_global_ppi = use_global_ppi
        self.node_map = {}  # Maps original ID to (type, index)
        
        # Node type specific mappings
        self.drug_to_idx = {}
        self.disease_to_idx = {}
        self.protein_to_idx = {}
        
        self.drug_nodes = set()
        self.disease_nodes = set()
        self.protein_nodes = set()
        
        # Edges
        self.drug_protein_edges = []
        self.disease_protein_edges = []
        self.protein_protein_edges = []
        self.drug_disease_edges = [] 
        
    def load_data(self):
        """Loads PrimeKG and Cell-type PPI data."""
        print("Loading PrimeKG...")
        self._load_primekg()
        
        if self.use_global_ppi:
            print("Using Global PPI from PrimeKG (skipping cell-type specific PPI)...")
        else:
            print("Loading Cell-type PPI...")
            self._load_celltype_ppi()
            
        print("Data loaded.")
        self._assign_indices()
        
    def build_graph(self):
        """Constructs the HeteroData graph object."""
        print("Building HeteroData graph...")
        data = HeteroData()
        
        # Add nodes
        data['drug'].num_nodes = len(self.drug_nodes)
        data['disease'].num_nodes = len(self.disease_nodes)
        data['protein'].num_nodes = len(self.protein_nodes)
        
        # Add edges
        def to_tensor(edges, map_a, map_b):
            src = [map_a[u] for u, v in edges]
            dst = [map_b[v] for u, v in edges]
            return torch.tensor([src, dst], dtype=torch.long)
            
        # Drug-Protein
        edge_index = to_tensor(self.drug_protein_edges, self.drug_to_idx, self.protein_to_idx)
        data['drug', 'targets', 'protein'].edge_index = edge_index
        data['protein', 'targeted_by', 'drug'].edge_index = edge_index.flip(0)
        
        # Disease-Protein
        edge_index = to_tensor(self.disease_protein_edges, self.disease_to_idx, self.protein_to_idx)
        data['disease', 'associated_with', 'protein'].edge_index = edge_index
        data['protein', 'associated_with', 'disease'].edge_index = edge_index.flip(0)
        
        # Protein-Protein
        edge_index = to_tensor(self.protein_protein_edges, self.protein_to_idx, self.protein_to_idx)
        data['protein', 'interacts_with', 'protein'].edge_index = edge_index
        
        # Drug-Disease (Labels)
        label_edge_index = to_tensor(self.drug_disease_edges, self.drug_to_idx, self.disease_to_idx)
        
        return data, label_edge_index

    def _load_primekg(self):
        """Parses PrimeKG CSV for relevant relations."""
        usecols = ['x_id', 'x_type', 'x_name', 'y_id', 'y_type', 'y_name', 'display_relation']
        df = pd.read_csv(self.primekg_path, usecols=usecols, low_memory=False)
        
        # Standardize types
        df['x_type'] = df['x_type'].replace('gene/protein', 'protein')
        df['y_type'] = df['y_type'].replace('gene/protein', 'protein')
        
        # 1. Load ALL proteins to ensure complete node map
        # This is crucial because cell-type PPIs might involve proteins that are not direct drug targets
        # but are structural bridges in the network.
        
        # Get all unique proteins from x and y columns
        x_proteins = df[df['x_type'] == 'protein'][['x_id', 'x_name']].drop_duplicates()
        y_proteins = df[df['y_type'] == 'protein'][['y_id', 'y_name']].drop_duplicates()
        
        # Add to node map and set
        for row in x_proteins.itertuples(index=False):
            pid, pname = str(row.x_id), row.x_name
            self.protein_nodes.add(pid)
            self.node_map[pname] = pid
            
        for row in y_proteins.itertuples(index=False):
            pid, pname = str(row.y_id), row.y_name
            self.protein_nodes.add(pid)
            self.node_map[pname] = pid
            
        # Extract Drug-Protein
        dp_mask = ((df['x_type'] == 'drug') & (df['y_type'] == 'protein')) | \
                  ((df['y_type'] == 'drug') & (df['x_type'] == 'protein'))
        dp_edges = df[dp_mask]
        
        # Extract Disease-Protein
        dsp_mask = ((df['x_type'] == 'disease') & (df['y_type'] == 'protein')) | \
                   ((df['y_type'] == 'disease') & (df['x_type'] == 'protein'))
        dsp_edges = df[dsp_mask]
        
        # Extract Drug-Disease (Indication)
        dd_mask = (((df['x_type'] == 'drug') & (df['y_type'] == 'disease')) | \
                   ((df['y_type'] == 'drug') & (df['x_type'] == 'disease'))) & \
                  (df['display_relation'] == 'indication')
        dd_edges = df[dd_mask]
        
        # Extract Global PPI if enabled
        pp_edges = pd.DataFrame()
        if self.use_global_ppi:
            pp_mask = (df['x_type'] == 'protein') & (df['y_type'] == 'protein')
            pp_edges = df[pp_mask]
        
        # Process edges and nodes
        # Drug-Protein
        for row in dp_edges.itertuples(index=False):
            x_id, x_type, x_name = str(row.x_id), row.x_type, row.x_name
            y_id, y_type, y_name = str(row.y_id), row.y_type, row.y_name
            
            self.drug_nodes.add(x_id if x_type == 'drug' else y_id)
            # Proteins already added above
            
            self.drug_protein_edges.append((x_id, y_id) if x_type == 'drug' else (y_id, x_id))
            
        # Disease-Protein
        for row in dsp_edges.itertuples(index=False):
            x_id, x_type, x_name = str(row.x_id), row.x_type, row.x_name
            y_id, y_type, y_name = str(row.y_id), row.y_type, row.y_name
            
            self.disease_nodes.add(x_id if x_type == 'disease' else y_id)
            # Proteins already added above
            
            self.disease_protein_edges.append((x_id, y_id) if x_type == 'disease' else (y_id, x_id))
            
        # Drug-Disease
        for row in dd_edges.itertuples(index=False):
            x_id, x_type = str(row.x_id), row.x_type
            y_id, y_type = str(row.y_id), row.y_type
            
            self.drug_nodes.add(x_id if x_type == 'drug' else y_id)
            self.disease_nodes.add(x_id if x_type == 'disease' else y_id)
            
            self.drug_disease_edges.append((x_id, y_id) if x_type == 'drug' else (y_id, x_id))
            
        # Global PPI
        if self.use_global_ppi:
            for row in pp_edges.itertuples(index=False):
                x_id, x_name = str(row.x_id), row.x_name
                y_id, y_name = str(row.y_id), row.y_name
                
                # Proteins already added above
                
                self.protein_protein_edges.append((x_id, y_id))

    def _load_celltype_ppi(self):
        """Loads cell-type specific PPI network."""
        count = 0
        missed = 0
        with open(self.celltype_ppi_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                gene_a, gene_b = parts[0], parts[1]
                
                id_a = self.node_map.get(gene_a)
                id_b = self.node_map.get(gene_b)
                
                if id_a and id_b:
                    self.protein_protein_edges.append((id_a, id_b))
                    count += 1
                else:
                    missed += 1
        print(f"Loaded {count} PPI edges. Missed {missed} edges due to missing ID mapping.")

    def _assign_indices(self):
        """Assigns numerical indices to nodes."""
        self.drug_to_idx = {d: i for i, d in enumerate(sorted(list(self.drug_nodes)))}
        self.disease_to_idx = {d: i for i, d in enumerate(sorted(list(self.disease_nodes)))}
        self.protein_to_idx = {p: i for i, p in enumerate(sorted(list(self.protein_nodes)))}
        
        print(f"Drugs: {len(self.drug_nodes)}, Diseases: {len(self.disease_nodes)}, Proteins: {len(self.protein_nodes)}")
        print(f"Drug-Protein Edges: {len(self.drug_protein_edges)}")
        print(f"Disease-Protein Edges: {len(self.disease_protein_edges)}")
        print(f"Drug-Disease Edges (Labels): {len(self.drug_disease_edges)}")

if __name__ == "__main__":
    # Example usage
    primekg_file = "data/raw/PrimeKG/kg.csv"
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primekg_full = os.path.join(base_path, "data/raw/PrimeKG/kg.csv")
    ppi_file = os.path.join(base_path, "data/raw/celltype_ppi/22708126/networks/networks/ppi_edgelists/acinar_cell_of_salivary_gland.txt")
    
    if os.path.exists(primekg_full) and os.path.exists(ppi_file):
        print("Testing Cell-Type Specific:")
        loader = DataLoader(primekg_full, ppi_file, use_global_ppi=False)
        loader.load_data()
        
        print("\nTesting Global PPI:")
        loader_global = DataLoader(primekg_full, ppi_file, use_global_ppi=True)
        loader_global.load_data()
