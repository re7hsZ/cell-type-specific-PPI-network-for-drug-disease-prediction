# Cell-Type Specific PPI Network for Drug-Disease Prediction

This repository contains the implementation of a cell-type specific Protein-Protein Interaction (PPI) network for predicting drug-disease associations. The framework integrates general medical knowledge graphs with cell-type specific protein interaction networks to enhance prediction accuracy using Heterogeneous Graph Neural Networks (HeteroGCN).

## Overview

The project constructs a heterogeneous graph consisting of drugs, diseases, and proteins. While drug-protein and disease-protein associations are derived from a general knowledge graph (PrimeKG), the protein-protein interaction layer is dynamically replaced with cell-type specific networks derived from single-cell transcriptomics (PINNACLE). This allows for context-aware drug repurposing predictions.

## Methodology

-   **General Knowledge Graph**: [PrimeKG](https://github.com/mims-harvard/PrimeKG) is used for drug-target and disease-gene associations.
-   **Cell-Type Specific PPIs**: Context-specific protein interaction networks are sourced from [PINNACLE](https://www.nature.com/articles/s41592-024-02341-3).
-   **Model**: A Heterogeneous Graph Convolutional Network (HeteroGCN) with SAGEConv layers is employed to learn node embeddings.
-   **Prediction**: A link prediction head scores the likelihood of drug-disease indications.

## Requirements

-   Python 3.8+
-   PyTorch
-   PyTorch Geometric
-   pandas
-   scikit-learn
-   numpy

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Data Structure

Place raw data in the `data/raw` directory:
-   `data/raw/PrimeKG/kg.csv`: The PrimeKG dataset.
-   `data/raw/celltype_ppi/`: Directory containing cell-type specific edge lists (e.g., `acinar_cell_of_salivary_gland.txt`).

## Usage

### Training

To train the model on a specific cell type, use `src/train.py`. The script handles data loading, graph construction, training, and evaluation.

```bash
python src/train.py --celltype acinar_cell_of_salivary_gland --epochs 10
```

### Arguments

-   `--celltype`: Name of the cell-type PPI file (without extension). Default: `acinar_cell_of_salivary_gland`.
-   `--epochs`: Number of training epochs. Default: `10`.

## Evaluation Metrics

The model is evaluated using the following metrics on a held-out test set of drug-indication pairs:
-   Area Under the Receiver Operating Characteristic Curve (auROC)
-   Area Under the Precision-Recall Curve (auPRC)
-   F1 Score
-   Mean Reciprocal Rank (MRR)

## References

1.  **TxGNN**: Huang, K., et al. "A foundation model for clinician-centered drug repurposing." *Nature Medicine* (2024). [https://doi.org/10.1038/s41591-024-03233-x](https://doi.org/10.1038/s41591-024-03233-x)
2.  **PINNACLE**: Li, M.M., et al. "Contextual AI models for single-cell protein biology." *Nature Methods* (2024). [https://doi.org/10.1038/s41592-024-02341-3](https://doi.org/10.1038/s41592-024-02341-3)
