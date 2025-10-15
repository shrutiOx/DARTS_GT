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
| MalNet-Tiny | Acc ↑ | **0.9325 ± 0.006** | 0.9264 (GPS) |
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
## Repository Structure
```
DARTS_GT/
├── DARTS_GT_LRGB/              # Long-Range Graph Benchmarks (Peptides)
│   ├── dartsgt/                # Core implementation
│   │   ├── layer/              # GT layers and GNN operators
│   │   ├── network/            # Model architectures
│   │   └── ..../               # others
│   ├── Configs/                # Configuration files by dataset
│   │   ├── Pep-func/           # Peptides-Func configs
│   │   └── Pep-struc/          # Peptides-Struct configs
│   ├── ResultsLogs/            # Experimental results (all seeds)
│   ├── ExplainerPepStruc.zip           # Interpretability analysis outputs
│   └── mainnas.py              # Main training script
│
├── DARTS_GT_NonLRGB/           # Standard benchmarks
│   ├── dartsgt/                # Core implementation
│   ├── Configs/                # Per-dataset configurations
│   │   ├── Zinc/               # ZINC configs (Darts_gt, Vanilla, etc.)
│   │   ├── MolHIV/             # MolHIV configs
│   │   ├── Cluster/            # CLUSTER configs
│   │   ├── Pattern/            # PATTERN configs
│   │   └── ...                 # MolPCBA, MalNet configs
│   ├── Result_Logs/            # Experimental results
│   ├── ExplainerMolHIV.zip           # Interpretability analysis outputs
│   └── mainnas.py              # Main training script
│
├── LICENSE                     # MIT License
└── README.md                   # This file
```

**For detailed code documentation, see [CODE_STRUCTURE.md](CODE_STRUCTURE.md)**  
**For configuration options, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md)**

## Reproducing Paper Results

### Benchmarking DARTS-GT

Configs for all datasets are organized in `Configs/` folders:
- **Non-LRGB datasets:** `DARTS_GT_NonLRGB/Configs/` (ZINC, MolHIV, MolPCBA, Pattern, Cluster, MalNet)
- **LRGB datasets:** `DARTS_GT_LRGB/Configs/` (Peptides-Func, Peptides-Struct)

Each dataset has multiple model variants: `Darts_gt/`, `Vanilla/`, `Uniform/`, `Random/`, `Symmetric/`, etc. - corresponding to different experimental tables in the paper.

**Example usage:**
```bash
# Navigate to appropriate directory
cd DARTS_GT_NonLRGB

# Run DARTS-GT (Table III-IV)
python mainnas.py --cfg Configs/Zinc/confignas_sparse.yaml

# Run Vanilla-GT baseline (Table II)
python mainnas.py --cfg Configs/MolHIV/Vanilla/confignas_sparse_V.yaml

# Run ablation studies (Table V-VI)
python mainnas.py --cfg Configs/Cluster/Random/confignas_rand.yaml
python mainnas.py --cfg Configs/Cluster/Symmetric/confignas_syym.yaml
```

**Run multiple seeds:** Modify `seed:` parameter in the config YAML file, or use shell scripts to automate.

**Our experimental logs:** The `ResultsLogs/` folders contain the actual logs and results we obtained for all seeds reported in the paper.

**Outputs:** Results saved to directory specified by `out_dir:` in config file, containing logs, metrics, and interpretability analysis.

## Interpretability Analysis

Our framework provides quantitative, causal interpretability via head ablation.

**Run interpretability analysis:**
```bash
cd DARTS_GT_NonLRGB
python mainnas.py --cfg Configs/MolHIV/DARTS_GT/confignas_sparse_interpret_on.yaml

# For LRGB
cd DARTS_GT_LRGB  
python mainnas.py --cfg Configs/Peptides-Struc/Darts-Gt/config_dense_interpret.yaml
```

**Enable in any config:** Set `cfg.gt.pk_explainer.enabled = True` in your YAML file.

**Output metrics:**

- **Head-Deviation (δ):** Prediction change when masking head m in layer ℓ (higher in positive number = more important)
- **Specialization:** Standard deviation of head impacts (high = few heads dominate; low = uniform importance)
- **Focus:** Jaccard similarity of top-k heads' attended nodes (high = consensus on substructures; low = divergent attention)

**Results location:**
- Live runs: `{out_dir}/pk_explainer_results/`
- Paper results: `Explainer.zip` files contain our interpretability analyses (MolHIV in `DARTS_GT_NonLRGB/`, Peptides-Struct in `DARTS_GT_LRGB/`) with per-instance metrics (Json files), attention heatmaps, and plots

**Key finding:** Visual attention salience does NOT always correlate with causal importance and  attention-based visualization or entropy heuristics cannot
 substitute for causal ablation methods.. Our metrics reveal which heads and nodes actually drive predictions.

**See paper Section III-C and IV-F,G for detailed methodology.**


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
