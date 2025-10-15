# Configuration Guide for DARTS-GT

This guide documents the **DARTS-GT specific parameters** and **actively modified GPS parameters**. For standard GraphGPS parameters (unchanged), refer to the [GraphGPS documentation](https://github.com/rampasek/GraphGPS).

---

## Table of Contents
- [DARTS-GT Core Parameters](#darts-gt-core-parameters)
- [Architecture Search (NAS)](#architecture-search-nas)
- [Interpretability (PK-Explainer)](#interpretability-pk-explainer)
- [Actively Modified GPS Parameters](#actively-modified-gps-parameters)
- [Complete LRGB Example](#complete-lrgb-example)

---

## DARTS-GT Core Parameters

All DARTS-GT specific parameters are under the `gt:` section.

### `routing_mode`
**Type:** `str`  
**Options:** `'uniform'`, `'random'`, `'nas'`  
**Default:** `'nas'`

Controls how GNN experts are selected at each layer:
- `'uniform'`: All experts equally weighted (baseline)
- `'random'`: Random expert selection each forward pass
- `'nas'`: DARTS-based architecture search to find optimal expert per layer
```yaml
gt:
  routing_mode: 'nas'  # Use DARTS to search for best experts
```

---

### `head_gnn_types`
**Type:** `list[str]`  
**Default:** `['GINE', 'CustomGatedGCN', 'GATV2']`

List of GNN expert types available for each layer's Key-Value computation. DARTS will search over these to find the best expert per layer.

**Supported GNN Types:**
- `'GINE'` - Graph Isomorphism Network with Edge features
- `'CustomGatedGCN'` - Gated Graph Convolutional Network
- `'GATV2'` - Graph Attention Network v2
- `'GCN'` - Standard Graph Convolutional Network
- `'SAGE'` - GraphSAGE
- `'GIN'` - Graph Isomorphism Network
- `'GENConv'` - Generalized Graph Convolution
- `'GAT'` - Graph Attention Network (v1)
- `'PNA'` - Principal Neighbourhood Aggregation
```yaml
gt:
  head_gnn_types: ['GINE', 'CustomGatedGCN', 'GATV2']
```

---

### `weight_fix`
**Type:** `str`  
**Options:** `'individual'`, `'all_single'`, or custom patterns like `'3:3:2'`  
**Default:** `'individual'`

Controls architecture weight sharing across layers:
- `'individual'`: Each layer has its own architecture weights (most flexible)
- `'all_single'`: All layers share the same architecture weights
- `'3:3:2'`: Custom grouping - first 3 layers share weights, next 3 share weights, last 2 share weights
```yaml
gt:
  weight_fix: '1:1'  # For 2 layers: each has independent weights
```

**Pattern Format:** Colon-separated integers where each number represents how many consecutive layers share weights. Sum must equal `gt.layers`.

---

### `weight_type`
**Type:** `str`  
**Options:** `'nas'`, `'softmax'`  
**Default:** `'nas'`

Determines how expert selection is implemented:
- `'nas'`: Uses DARTS architecture search
- `'softmax'`: Softmax-based expert mixing
```yaml
gt:
  weight_type: 'nas'
```

---

### `gnns_type_used`
**Type:** `int`  
**Default:** `2`

Number of GNN expert types actively used during architecture search. Must be ≤ `len(head_gnn_types)`.
```yaml
gt:
  gnns_type_used: 2  # Use 2 out of 3 available expert types
```

---

### `residual_mult`
**Type:** `float`  
**Default:** `0.1`

Multiplier for residual connections in the model. Controls the strength of skip connections.
```yaml
gt:
  residual_mult: 0.1
```

---

## Architecture Search (NAS)

All NAS-specific parameters are under `gt.nas:`.

### `enabled`
**Type:** `bool`  
**Default:** `true`

Enable/disable DARTS architecture search. When `false`, uses default expert selection based on `routing_mode`.
```yaml
gt:
  nas:
    enabled: true
```

---

### `darts_epochs`
**Type:** `int`  
**Default:** `50`

Number of epochs for DARTS architecture search phase. Only used when `routing_mode='nas'`.

**Note:** For `routing_mode='uniform'` or `'random'`, DARTS runs for 2 epochs regardless of this value.
```yaml
gt:
  nas:
    darts_epochs: 50
```

---

### `darts_split_ratio`
**Type:** `float`  
**Range:** `(0.0, 1.0)`  
**Default:** `0.6`

Training/validation split ratio for DARTS bilevel optimization. During DARTS phase, the training set is split into:
- DARTS train: `darts_split_ratio` (e.g., 60%)
- DARTS validation: `1 - darts_split_ratio` (e.g., 40%)
```yaml
gt:
  nas:
    darts_split_ratio: 0.6
```

---

### `arc_learning_rate`
**Type:** `float`  
**Default:** `4.0e-4`

Learning rate for architecture parameters (α) during DARTS search.
```yaml
gt:
  nas:
    arc_learning_rate: 4.0e-4
```

---

### `grad_clip`
**Type:** `float`  
**Default:** `5.0`

Gradient clipping value for architecture parameter updates. Prevents exploding gradients during DARTS.
```yaml
gt:
  nas:
    grad_clip: 5.0
```

---

### `unrolled`
**Type:** `bool`  
**Default:** `false`

Use second-order approximation (unrolled) vs. first-order approximation for DARTS.
- `true`: More accurate but slower and memory-intensive
- `false`: Faster first-order approximation (recommended)
```yaml
gt:
  nas:
    unrolled: false
```

---

### `darts_lr_schedule`
**Type:** `dict`

Learning rate scheduler parameters for DARTS model parameters (not architecture parameters).

#### `lr_reduce_factor`
**Type:** `float`  
**Default:** `0.5`

Factor by which to reduce learning rate on plateau.

#### `lr_schedule_patience`
**Type:** `int`  
**Default:** `10`

Number of epochs with no improvement before reducing LR.

#### `min_lr`
**Type:** `float`  
**Default:** `1.0e-6`

Minimum learning rate threshold.

#### `init_lr`
**Type:** `float`  
**Default:** `0.0025`

Initial learning rate for model parameters during DARTS.

#### `weight_decay`
**Type:** `float`  
**Default:** `3e-4`

L2 regularization for model parameters during DARTS.
```yaml
gt:
  nas:
    darts_lr_schedule:
      lr_reduce_factor: 0.5
      lr_schedule_patience: 10
      min_lr: 1.0e-6
      init_lr: 0.0025
      weight_decay: 3e-4
```

---

## Interpretability (PK-Explainer)

DARTS-GT introduces a quantitative interpretability framework. Parameters are under `gt.pk_explainer:`.

### `enabled`
**Type:** `bool`  
**Default:** `false`

Enable PK-Explainer analysis after training. Computes:
- **Specialization**: Std dev of head importance (higher = clearer roles)
- **Focus**: Node overlap among top-k heads (higher = consensus on important nodes)
- **Correlation**: Similarity between attention patterns of top heads
- Per-head attention statistics
```yaml
gt:
  pk_explainer:
    enabled: true
```

---

### `visualization`
**Type:** `bool`  
**Default:** `true`

Generate visualization plots including:
- Specialization vs. Focus scatter plot
- Metric distributions
- Per-graph attention heatmaps (for `graph_ids`)
```yaml
gt:
  pk_explainer:
    visualization: true
```

---

### `save_attention`
**Type:** `bool`  
**Default:** `false`

Save raw attention weights to disk. **Warning:** Very memory-intensive for large datasets.
```yaml
gt:
  pk_explainer:
    save_attention: false  # Recommended to keep false
```

---

### `k_heads` (Optional)
**Type:** `int`  
**Default:** `5`

Number of top/bottom heads to analyze for Focus metric.
```yaml
gt:
  pk_explainer:
    k_heads: 5
```

---

### `graph_ids` (Optional)
**Type:** `list[int]`  
**Default:** `[]`

Specific graph IDs to visualize attention heatmaps for. Empty list = no individual graph visualizations.
```yaml
gt:
  pk_explainer:
    graph_ids: [0, 10, 100]  # Visualize graphs 0, 10, and 100
```

---

## Actively Modified GPS Parameters

These GPS parameters are actively used/modified in DARTS-GT:

### `layer_type`
**Type:** `str`  
**Format:** `'{local_gnn}+{global_attention}'`

**DARTS-GT Usage:** We set `local_gnn='None'` and use experts for K/V computation instead.
```yaml
gt:
  layer_type: 'None+Transformer'  # No local GNN, experts handle K/V
```

**Supported global attention:**
- `Transformer` - Standard multi-head attention
- `SparseTransformer` - Graph-structure-aware sparse attention
- `Performer` - Linear attention
- `BigBird` - Sparse block attention

---

### `layers`
**Type:** `int`  
**Default:** `2`

Number of DARTS-GT layers. Each layer independently selects its best expert via NAS.
```yaml
gt:
  layers: 2
```

---

### `dim_hidden`, `n_heads`, `dropout`, etc.

Standard GraphGPS parameters actively used. See [GraphGPS docs](https://github.com/rampasek/GraphGPS) for full list.

---

## Complete LRGB Example

Here's a complete configuration for LRGB datasets:
```yaml
out_dir: results
train:
  mode: custom
  
model:
  type: NASModelEdge  # Or NASModelQ for Q-projection variant

gt:
  # === DARTS-GT Core ===
  routing_mode: 'nas'
  head_gnn_types: ['GINE', 'CustomGatedGCN', 'GATV2']
  weight_fix: '1:1'  # Independent weights for 2 layers
  weight_type: 'nas'
  gnns_type_used: 2
  residual_mult: 0.1
  
  # === Architecture Search ===
  nas:
    enabled: true
    darts_epochs: 50
    darts_split_ratio: 0.6
    arc_learning_rate: 4.0e-4
    grad_clip: 5.0
    unrolled: false
    darts_lr_schedule:
      lr_reduce_factor: 0.5
      lr_schedule_patience: 10
      min_lr: 1.0e-6
      init_lr: 0.0025
      weight_decay: 3e-4
  
  # === Interpretability ===
  pk_explainer:
    enabled: true
    visualization: true
    save_attention: false
    k_heads: 5
    # graph_ids: [0, 10, 100]  # Optional: specific graphs to visualize
  
  # === GPS Parameters (actively used) ===
  layer_type: 'None+Transformer'
  layers: 2
  dim_hidden: 64
  n_heads: 4
  dropout: 0.1
  attn_dropout: 0.1
```

---

## Usage Notes

1. **Routing Modes:**
   - Use `'nas'` for full architecture search
   - Use `'uniform'` or `'random'` as baselines (runs 2 DARTS epochs)

2. **Weight Sharing:**
   - `weight_fix='individual'`: Maximum flexibility, longer search
   - `weight_fix='all_single'`: Fastest search, less flexibility
   - Custom patterns like `'2:2:2'`: Balance flexibility and speed

3. **PK-Explainer:**
   - Enable after training completes
   - For large datasets, set `save_attention: false` to save memory
   - Use `graph_ids` to visualize specific interesting graphs

4. **Expert Selection:**
   - More experts in `head_gnn_types` = longer search but potentially better results
   - Limit `gnns_type_used` to speed up search

---

## See Also

- [CODE_STRUCTURE.md](CODE_STRUCTURE.md) - Code organization
- [README.md](README.md) - Quick start guide
- [GraphGPS Documentation](https://github.com/rampasek/GraphGPS) - Base framework

---

**Questions?** Check [CONFIG_GUIDE.md](CONFIG_GUIDE.md) or open an issue on GitHub.
