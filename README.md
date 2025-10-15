# DARTS_GT
This repository contains code for our work DARTS-GT


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
