## DARTS-GT: Differentiable Architecture Search for Graph Transformers with Quantifiable Instance-Specific Interpretability Analysis
<img width="553" height="787" alt="image" src="https://github.com/user-attachments/assets/6d47d711-f9f3-46b1-b90c-dcd9250af63c" />





## Overview
Graph Transformers combine attention with graph inductive biases but suffer from:

Rigid architectures - commit to fixed GNN types across all layers
Black-box predictions - lack quantifiable methods to identify which components drive predictions

## DARTS-GT addresses both:

 **DARTS-GT** redesigns Graph Transformer attention through **asymmetry** and **differentiable architecture search**, while introducing the first **quantitative interpretability framework** for Graph Transformers.

**The problem:** Current GTs commit to fixed GNN types across layers and lack quantifiable methods to identify which components drive predictions—making it difficult to understand what the model actually learned.

**Our solution:** Causal ablation-based interpretability that reveals which heads and nodes matter for each prediction.



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
