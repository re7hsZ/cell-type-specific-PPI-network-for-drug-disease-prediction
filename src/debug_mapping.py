import pandas as pd
import os

def debug_mapping():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primekg_path = os.path.join(base_path, "data/raw/PrimeKG/kg.csv")
    celltype_ppi_path = os.path.join(base_path, "data/raw/celltype_ppi/22708126/networks/networks/ppi_edgelists/acinar_cell_of_salivary_gland.txt")
    
    print("Loading PrimeKG nodes...")
    usecols = ['x_type', 'x_name', 'y_type', 'y_name']
    df = pd.read_csv(primekg_path, usecols=usecols, low_memory=False)
    
    # Standardize types
    df['x_type'] = df['x_type'].replace('gene/protein', 'protein')
    df['y_type'] = df['y_type'].replace('gene/protein', 'protein')
    
    primekg_proteins = set()
    # Get all proteins from x
    primekg_proteins.update(df[df['x_type'] == 'protein']['x_name'].dropna().astype(str).unique())
    # Get all proteins from y
    primekg_proteins.update(df[df['y_type'] == 'protein']['y_name'].dropna().astype(str).unique())
    
    print(f"Total unique proteins in PrimeKG: {len(primekg_proteins)}")
    
    print("Loading Cell-type PPI genes...")
    celltype_genes = set()
    with open(celltype_ppi_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                celltype_genes.add(parts[0])
                celltype_genes.add(parts[1])
                
    print(f"Total unique genes in Cell-type PPI: {len(celltype_genes)}")
    
    # Intersection
    common = primekg_proteins.intersection(celltype_genes)
    missing = celltype_genes - primekg_proteins
    
    print(f"Common genes: {len(common)}")
    print(f"Missing genes: {len(missing)}")
    
    print("\nSample missing genes:")
    print(list(missing)[:20])
    
    # Check for case sensitivity
    print("\nChecking case sensitivity...")
    primekg_lower = {p.lower(): p for p in primekg_proteins}
    recoverable = []
    for m in missing:
        if m.lower() in primekg_lower:
            recoverable.append((m, primekg_lower[m.lower()]))
            
    print(f"Recoverable via case-insensitive match: {len(recoverable)}")
    if recoverable:
        print("Sample recoverable:", recoverable[:10])

if __name__ == "__main__":
    debug_mapping()
