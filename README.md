## DARTS-GT: Differentiable Architecture Search for Graph Transformers with Quantifiable Instance-Specific Interpretability Analysis
<img width="553" height="787" alt="image" src="https://github.com/user-attachments/assets/6d47d711-f9f3-46b1-b90c-dcd9250af63c" />





## Overview

Graph Transformers combine attention with graph structure, but face key limitations:

**Current Limitations:**
- **Rigid architectures** - commit to fixed GNN types across all layers
- **Black-box predictions** - lack quantifiable methods to identify what drives decisions

**DARTS-GT redesigns address these limitations:**

- **DARTS-based architecture search** - automated depth-wise GNN selection within attention blocks
- **Asymmetric attention mechanism** - queries from features, keys/values from GNN transformations avoiding parallel MPNN+Transformer paths (GPS, UGAS etc)
- **First quantitative interpretability framework** - causal ablation via Head-deviation, Specialization, and Focus metrics

** Result:** Competitive SOTA without parallel MPNNs + quantifiable interpretability for Graph Transformers.

## Key Results

**State-of-the-art on 4/8 benchmarks:**

| Dataset | Metric | DARTS-GT | Previous Best |
|---------|--------|----------|---------------|
| ZINC | MAE ↓ | **0.066 ± 0.003** | 0.070 (GPS) |
| MalNet-Tiny | Acc ↑ | **0.9325 ± 0.006** | 92.64 (GPS) |
| Peptides-Func | AP ↑ | **0.669 ± 0.004** | 0.667 (UGAS) |
| Peptides-Struct | MAE ↓ | **0.246 ± 0.0006** | 0.247 (UGAS) |

Competitive performance on PATTERN, CLUSTER, MolHIV, MolPCBA.

**Architecture Discovery:**
- Dataset-specific patterns emerge: from highly specialized (81% GINE on MolPCBA) to balanced (ZINC)
- Heterogeneous architectures consistently produce more interpretable models than other baseline GTs 

**Interpretability Insight:**
- Visual attention salience does NOT always correlate with causal importance 
- Our metrics reveal which heads and nodes actually drive predictions

## Installation
```bash
# Clone repository
git clone https://github.com/shrutiOx/DARTS_GT.git
cd DARTS_GT

# Install dependencies (Python ≥3.8, PyTorch ≥1.12, PyG ≥2.1)
pip install -r requirements.txt
```
## Quick Start
```bash
# Navigate to the directory
cd DARTS_GT_NonLRGB

# Run DARTS search + training
python mainnas.py --cfg Configs/Zinc/confignas_sparse.yaml

# For LRGB datasets
cd ../DARTS_GT_LRGB
python mainnas.py --cfg Configs/Peptides-func/Darts-Gt/confignas_dense.yaml
```

**Expected runtime:** ~15 hours on single GPU (architecture search + training)

## Acknowledgments

This codebase builds upon and extends:
- **[GraphGPS](https://github.com/rampasek/GraphGPS)** (Rampášek et al., NeurIPS 2022) - MIT License
- **[LRGB Benchmarks](https://github.com/vijaydwivedi75/lrgb)** (Dwivedi et al., NeurIPS 2022) - MIT License

We extend sincere gratitude to the authors for open-sourcing their implementations.

### Our Contributions
This repository introduces several novel components:
- **Asymmetric attention mechanism** with depth-wise GNN selection
- **DARTS-based architecture search** for Graph Transformers  
- **Quantitative interpretability framework** (Head-deviation, Specialization, Focus metrics)
- **Causal ablation analysis** for GT interpretability

While our implementation leverages the GraphGPS infrastructure (data loaders, training loops, configuration system, hyperparameters), the core methodological contributions—architecture search, interpretability analysis, and asymmetric attention design—are original to this work.

### License
This project is released under the MIT License, consistent with the original GraphGPS and LRGB codebases. See [LICENSE](LICENSE) for details.
