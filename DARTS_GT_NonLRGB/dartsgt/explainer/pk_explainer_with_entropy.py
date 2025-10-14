# -*- coding: utf-8 -*-
"""
PK-Explainer: Complete Implementation with Focus Fix and Graph Visualizations
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.graphgym.config import cfg
from tqdm import tqdm
import matplotlib.patches as mpatches


def compute_entropy_deviation_correlation(model, test_loader, checkpoint_path, all_deviations):
    """
    Compute correlation between attention entropy and head deviation for each graph.
    This tests whether attention concentration predicts functional importance.
    
    Returns:
        entropy_dev_correlations: List of Spearman correlations (one per graph)
        per_graph_entropy: Dict mapping graph_idx -> {head_name: entropy_value}
    """
    import scipy.stats as stats
    
    logging.info("Computing entropy-deviation correlations...")
    
    num_graphs = len(all_deviations)
    entropy_dev_correlations = []
    per_graph_entropy = {}
    
    # Suppress verbose logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    # Create model once for attention extraction
    model, _ = create_model_from_checkpoint(checkpoint_path)
    if hasattr(model, 'model'):
        layers = list(model.model.discrete_sequential) if hasattr(model.model, 'discrete_sequential') else model.model.layers
    else:
        layers = model.layers
    for layer in layers:
        if hasattr(layer, 'log_attn_weights'):
            layer.log_attn_weights = True
    
    logging.getLogger().setLevel(old_level)
    
    for graph_idx in tqdm(range(num_graphs), desc="Computing entropy correlations"):
        entropy_values = []
        deviation_values = []
        head_entropies = {}
        
        # For each head in this graph
        for layer_idx in all_deviations[graph_idx]:
            for head_idx in all_deviations[graph_idx][layer_idx]:
                # Get attention matrix
                attn_matrix = extract_attention_for_graph(
                    model, test_loader, graph_idx, layer_idx, head_idx
                )
                
                if attn_matrix is not None and attn_matrix.size > 0:
                    # Pool attention: sum across rows (incoming attention per node)
                    pooled_attn = np.sum(attn_matrix, axis=0)
                    
                    # Normalize to probability distribution
                    total = pooled_attn.sum()
                    if total > 0:
                        prob_dist = pooled_attn / total
                        
                        # Compute entropy: H = -Σ p_j log(p_j)
                        # Filter out zeros to avoid log(0)
                        prob_dist_nonzero = prob_dist[prob_dist > 0]
                        entropy = -np.sum(prob_dist_nonzero * np.log(prob_dist_nonzero))
                    else:
                        entropy = 0.0
                    
                    # Get deviation for this head
                    deviation = all_deviations[graph_idx][layer_idx][head_idx]
                    deviation = float(deviation) if hasattr(deviation, 'item') else float(deviation)
                    
                    # Store
                    head_name = f'L{layer_idx}_H{head_idx}'
                    head_entropies[head_name] = float(entropy)
                    entropy_values.append(entropy)
                    deviation_values.append(deviation)
        
        # Compute Spearman correlation for this graph
        if len(entropy_values) >= 3:  # Need at least 3 points for meaningful correlation
            corr, _ = stats.spearmanr(entropy_values, deviation_values)
            # Handle NaN (happens if all values identical)
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        entropy_dev_correlations.append(float(corr))
        per_graph_entropy[graph_idx] = head_entropies
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    logging.info(f"Computed entropy-deviation correlations for {num_graphs} graphs")
    
    return entropy_dev_correlations, per_graph_entropy


def visualize_entropy_deviation_analysis(entropy_dev_correlations, save_dir):
    """
    Create visualization showing the distribution of entropy-deviation correlations.
    Low/inconsistent correlation proves the visualization paradox.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of correlations
    axes[0].hist(entropy_dev_correlations, bins=30, alpha=0.7, color='purple', edgecolor='black')
    median_corr = np.median(entropy_dev_correlations)
    mean_corr = np.mean(entropy_dev_correlations)
    
    axes[0].axvline(median_corr, color='red', linestyle='--', linewidth=2,
                    label=f'Median: {median_corr:.3f}')
    axes[0].axvline(mean_corr, color='orange', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_corr:.3f}')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    axes[0].set_xlabel('Spearman Correlation (Entropy vs Deviation)', fontsize=12)
    axes[0].set_ylabel('Number of Graphs', fontsize=12)
    axes[0].set_title('Attention Entropy Does Not Predict Functional Importance', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(entropy_dev_correlations, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    axes[1].set_ylabel('Spearman Correlation', fontsize=12)
    axes[1].set_title('Distribution Summary', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Median: {median_corr:.3f}\nMean: {mean_corr:.3f}\nStd: {np.std(entropy_dev_correlations):.3f}'
    axes[1].text(1.15, 0.5, stats_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'entropy_deviation_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved entropy-deviation correlation plot")

def create_model_from_checkpoint(checkpoint_path):
    """Create a fresh model and load checkpoint weights : handles standard-gt, gps and darts-gt"""
    from torch_geometric.graphgym.model_builder import create_model
    import logging
    
    # Load checkpoint
    device = torch.device(cfg.accelerator)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get optimal weights if NAS model
    optimal_weights = checkpoint.get('optimal_weights', None)
    
    # FIX: Detect and fix dimension mismatch BEFORE creating model --reqd for MOLHIV etc
    actual_dim_out = None
    for key in checkpoint['model_state_dict'].keys():
        if 'post_mp.FC_layers.2.weight' in key:  # Look for final layer
            actual_dim_out = checkpoint['model_state_dict'][key].shape[0]
            break
        elif 'post_mp.layer_post_mp.model.1.model.weight' in key:  # PATTERN structure
            actual_dim_out = checkpoint['model_state_dict'][key].shape[0]
            break
    
    original_dim_out = cfg.share.dim_out
    if actual_dim_out and actual_dim_out != cfg.share.dim_out:
        logging.info(f"Detected dim_out mismatch: config={cfg.share.dim_out}, saved={actual_dim_out}")
        cfg.share.dim_out = actual_dim_out
    
    # Suppress verbose logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    

    
    # Create model with corrected dimensions -- placeholder
    model = create_model(cfg)
    
    # For NAS models, convert to discrete version
    if cfg.model.type == 'NASModelEdge' and optimal_weights is not None:
        from dartsgt.network.NAS_model_edge import NASModelEdge
        temp_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
        discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
        discrete_model.optimal_weights_dict = optimal_weights
        model.model = discrete_model
    elif cfg.model.type == 'NASModel' and optimal_weights is not None:
        from dartsgt.network.NAS_model import NASModel
        temp_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
        discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
        discrete_model.optimal_weights_dict = optimal_weights
        model.model = discrete_model
    if cfg.model.type == 'NASModelQE' and optimal_weights is not None:
        from dartsgt.network.NAS_model_qproj_edge import NASModelEdge
        temp_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
        discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
        discrete_model.optimal_weights_dict = optimal_weights
        model.model = discrete_model
    elif cfg.model.type == 'NASModelQ' and optimal_weights is not None:
        from dartsgt.network.NAS_model_qproj import NASModel
        temp_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
        discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
        discrete_model.optimal_weights_dict = optimal_weights
        model.model = discrete_model
    
    # Restore logging level
    logging.getLogger().setLevel(old_level)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Restore original dimension for safety
    cfg.share.dim_out = original_dim_out
    
    return model, optimal_weights


def mask_attention_head(model, layer_idx, head_idx):
    """Mask a specific attention head - preserves sparse attention handling"""
    # Get layers
    if hasattr(model, 'model'):
        if hasattr(model.model, 'discrete_sequential'):
            layers = list(model.model.discrete_sequential)
        else:
            layers = model.model.layers
    else:
        layers = model.layers
    
    layer = layers[layer_idx]
    if hasattr(layer, 'log_attn_weights'):
        layer.log_attn_weights = True
    
    if hasattr(layer, 'self_attn') and layer.self_attn is not None:
        # Check if sparse attention FIRST
        if hasattr(layer, 'global_model_type') and layer.global_model_type == 'SparseTransformer':
            # Handle sparse attention
            sparse_attn = layer.self_attn
            head_dim = sparse_attn.head_dim
            
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
            # Zero out sparse attention weights
            sparse_attn.W_k.weight.data[start_idx:end_idx, :] = 0
            sparse_attn.W_v.weight.data[start_idx:end_idx, :] = 0
            sparse_attn.W_o.weight.data[:, start_idx:end_idx] = 0
            
        elif hasattr(layer.self_attn, 'in_proj_weight'):
            # Handle regular MultiheadAttention
            embed_dim = layer.dim_h
            head_dim = embed_dim // layer.num_heads
            
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
            # Zero out the weights for this head
            layer.self_attn.in_proj_weight.data[start_idx:end_idx, :] = 0  # Q
            layer.self_attn.in_proj_weight.data[embed_dim + start_idx:embed_dim + end_idx, :] = 0  # K
            layer.self_attn.in_proj_weight.data[2*embed_dim + start_idx:2*embed_dim + end_idx, :] = 0  # V
            layer.self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0  # Output


def evaluate_model_on_dataset(model, test_loader):
    """Evaluate model and return predictions for all graphs"""
    device = torch.device(cfg.accelerator)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Get predictions
            pred = model(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # Store predictions and labels
            all_predictions.append(pred.cpu())
            all_labels.append(batch.y.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Squeeze if needed
    if len(all_labels.shape) > 1 and all_labels.shape[1] == 1:
        all_labels = all_labels.squeeze(-1)
    if len(all_predictions.shape) > 1 and all_predictions.shape[1] == 1:
        all_predictions = all_predictions.squeeze(-1)
    
    return all_predictions.numpy(), all_labels.numpy()

def extract_attention_for_graph(model, test_loader, graph_idx, layer_idx, head_idx):
    """
    Extract attention weights using saved layer.attn_weights (like IAC does)
    Returns attention matrix with ACTUAL graph size (no padding)
    """
    device = torch.device(cfg.accelerator)
    model.eval()
    
    # Import needed for sparse attention
    from torch_scatter import scatter_add
    
    # Find which batch contains this graph
    graph_counter = 0
    target_batch = None
    local_graph_idx = None
    
    for batch in test_loader:
        batch = batch.to(device)
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        if graph_counter <= graph_idx < graph_counter + num_graphs:
            target_batch = batch
            local_graph_idx = graph_idx - graph_counter #getting local graph idx in that batch
            break
        
        graph_counter += num_graphs
    
    if target_batch is None:
        return None
    
    # GET ACTUAL GRAPH SIZE FROM BATCH
    batch_idx = target_batch.batch
    unique_graphs, graph_sizes = torch.unique(batch_idx, return_counts=True)
    actual_graph_size = graph_sizes[local_graph_idx].item()
    
    logging.info(f"Graph {graph_idx}: Actual size from batch = {actual_graph_size} nodes")
    
    # FIX 1: Ensure correct dtype for embeddings (prevents float/int error)
    '''
    if hasattr(target_batch, 'x') and target_batch.x is not None:
        if target_batch.x.dtype in [torch.float32, torch.float64]:
            if target_batch.x.min() >= 0 and target_batch.x.max() < 10000:
                target_batch.x = target_batch.x.long()'''
                
        # FIX 1: Ensure correct dtype for embeddings (dataset-specific)
    if hasattr(target_batch, 'x') and target_batch.x is not None:
        # Only convert to long for datasets with categorical features
        if cfg.dataset.name in ['PATTERN', 'pattern', 'CLUSTER', 'cluster']:
            # These datasets have continuous features - keep as float
            if target_batch.x.dtype not in [torch.float32, torch.float64]:
                target_batch.x = target_batch.x.float()
        else:
            if target_batch.x.dtype in [torch.float32, torch.float64]:
                if target_batch.x.min() >= 0 and target_batch.x.max() < 10000:
                    target_batch.x = target_batch.x.long()
        # For other datasets, keep the original dtype
    
    # Get layers structure
    if hasattr(model, 'model'):
        if hasattr(model.model, 'discrete_sequential'):
            layers = list(model.model.discrete_sequential)
        else:
            layers = model.model.layers
    else:
        layers = model.layers
    
    # FIX 2: Enable attention logging for ALL layers BEFORE forward pass
    for layer in layers:
        if hasattr(layer, 'log_attn_weights'):
            layer.log_attn_weights = True
    
    # NOW run forward pass to populate attn_weights
    with torch.no_grad():
        _ = model(target_batch)
    
    # Get the specific layer we want
    layer = layers[layer_idx]
    
    # Access saved attention weights (like IAC does)
    if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
        if isinstance(layer.attn_weights, tuple):
            # Sparse attention handling - already returns correct size
            sparse_weights, edge_index = layer.attn_weights
            
            # Get batch information
            batch_idx = target_batch.batch
            unique_graphs = torch.unique(batch_idx)
            
            # Find nodes for our specific graph
            target_graph_id = unique_graphs[local_graph_idx]
            node_mask = (batch_idx == target_graph_id)
            graph_nodes = torch.where(node_mask)[0]
            num_graph_nodes = len(graph_nodes)
            
            # Verify size matches
            assert num_graph_nodes == actual_graph_size, f"Size mismatch: {num_graph_nodes} vs {actual_graph_size}"
            
            # Create node mapping (global to local indices)
            node_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(graph_nodes)}
            
            # Filter edges for this graph - sparse handling
            src, dst = edge_index
            graph_edges_mask = node_mask[src] & node_mask[dst]
            graph_src = src[graph_edges_mask]
            graph_dst = dst[graph_edges_mask]
            
            # Get attention weights for this graph and head
            graph_attn_weights = sparse_weights[graph_edges_mask, head_idx]
            
            # Create dense attention matrix for this graph (already correct size)
            attn_matrix = torch.zeros(actual_graph_size, actual_graph_size, device=device)
            
            # Fill the attention matrix
            for s, d, w in zip(graph_src, graph_dst, graph_attn_weights):
                local_src = node_map[s.item()]
                local_dst = node_map[d.item()]
                attn_matrix[local_src, local_dst] = w
            
            # Normalize if needed (sparse attention might not be normalized)
            row_sums = attn_matrix.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            attn_matrix = attn_matrix / row_sums
            
            return attn_matrix.cpu().numpy()
            
        else:
            # Dense attention [batch, num_heads, seq, seq]
            attn = layer.attn_weights
            if local_graph_idx < attn.shape[0] and head_idx < attn.shape[1]:
                # CRITICAL: Return ONLY the actual graph portion (no padding!)
                full_attn = attn[local_graph_idx, head_idx]
                
                # Crop to actual graph size
                actual_attn = full_attn[:actual_graph_size, :actual_graph_size]
                
                logging.info(f"Dense attention: cropped from {full_attn.shape} to {actual_attn.shape}")
                
                return actual_attn.cpu().numpy()
    
    return None

def compute_focus_for_graph(model, test_loader, graph_idx, top_heads, checkpoint_path, 
                           all_deviations=None, vis_graphs=None, k_heads=5):
    """
    Compute Focus metric using fixed attention extraction
    Now also creates visualization if graph_idx is in vis_graphs
    NOW ALSO COMPUTES: Correlation, Total Attention, and Stdev per head
    """
    # Suppress verbose logging
    import logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    # Create ONE model for all attention extractions
    model, _ = create_model_from_checkpoint(checkpoint_path)
    if hasattr(model, 'model'):
        layers = list(model.model.discrete_sequential) if hasattr(model.model, 'discrete_sequential') else model.model.layers
    else:
        layers = model.layers
    for layer in layers:
        if hasattr(layer, 'log_attn_weights'):
            layer.log_attn_weights = True
    
    # Restore logging level
    logging.getLogger().setLevel(old_level)
    
    top_nodes_per_head = []
    head_to_nodes = {}
    attention_matrices = {}  # Store actual attention matrices
    actual_graph_size = None  # Will be determined from first valid matrix
    
    # NEW: Dictionaries for new metrics
    total_attn_dict = {}
    stdev_dict = {}
    
    for layer_idx, head_idx in top_heads:
        # Get attention weights - ALREADY CORRECT SIZE from extract_attention_for_graph
        attn_weights = extract_attention_for_graph(model, test_loader, graph_idx, layer_idx, head_idx)
        
        if attn_weights is not None:
            # Store the attention matrix (already correct size, no cropping needed!)
            head_key = f'L{layer_idx}_H{head_idx}'
            attention_matrices[head_key] = attn_weights
            
            # NEW: Compute total attention and stdev for this head
            total_attn_dict[head_key] = float(np.sum(attn_weights))
            stdev_dict[head_key] = float(np.std(attn_weights))
            
            # Get actual size from the matrix shape (first time only)
            if actual_graph_size is None:
                actual_graph_size = attn_weights.shape[0]
                logging.info(f"Graph {graph_idx}: Size = {actual_graph_size} nodes")
            
            # Sum attention across source nodes to get importance per target node : summing over all rows ie across coloumns
            node_attention = np.sum(attn_weights, axis=0)
            
            # Get top 10% of nodes
            num_nodes = len(node_attention)
            num_top = max(1, num_nodes // 10)
            top_node_indices = np.argsort(node_attention)[-num_top:]
            
            # Store nodes WITH attention scores in descending order
            nodes_with_scores = []
            for idx in top_node_indices:
                nodes_with_scores.append({
                    'node': int(idx),
                    'attention_score': float(node_attention[idx])
                })
            
            # Sort by attention score descending
            nodes_with_scores.sort(key=lambda x: x['attention_score'], reverse=True)
            top_nodes_per_head.append(set(top_node_indices))
            head_to_nodes[head_key] = nodes_with_scores
            
            # Debug output for graph 100
            if graph_idx == 100 and layer_idx == 0 and head_idx == 3:
                print(f"\n=== Graph 100, L0_H3 Debug ===")
                print(f"Shape: {attn_weights.shape}")
                print(f"Total nodes: {len(node_attention)}")
                print(f"Top 10% = {num_top} nodes")
                
                # Show actual top nodes
                top_5 = np.argsort(node_attention)[-5:]
                print(f"Actual top 5 nodes: {top_5}")
                print(f"Their attention sums: {node_attention[top_5]}")
    
    # NEW: Compute correlation between attention patterns of top heads
    correlation_scores = []
    if len(attention_matrices) >= 2:
        head_keys = list(attention_matrices.keys())
        for i in range(len(head_keys)):
            for j in range(i + 1, len(head_keys)):
                # Flatten attention matrices
                attn_i = attention_matrices[head_keys[i]].flatten()
                attn_j = attention_matrices[head_keys[j]].flatten()
                
                # Compute Pearson correlation
                if len(attn_i) > 0 and len(attn_j) > 0:
                    # Avoid issues with constant arrays
                    if np.std(attn_i) > 0 and np.std(attn_j) > 0:
                        corr = np.corrcoef(attn_i, attn_j)[0, 1]
                        correlation_scores.append(corr)
                    else:
                        correlation_scores.append(0.0)
    
    correlation = np.mean(correlation_scores) if correlation_scores else 0.0

    # CREATE VISUALIZATION IF THIS GRAPH IS IN vis_graphs
    if vis_graphs and graph_idx in vis_graphs and all_deviations:
        logging.info(f"Creating visualization for graph {graph_idx} (size: {actual_graph_size})...")
        
        vis_dir = Path(cfg.run_dir) / 'pk_explainer_results' / 'graph_visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get deviations for this graph
        graph_devs = all_deviations[graph_idx]
        
        # Flatten deviations
        all_heads = []
        for layer_idx in graph_devs:
            for head_idx in graph_devs[layer_idx]:
                all_heads.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'deviation': graph_devs[layer_idx][head_idx]
                })
        
        # Sort heads by deviation value
        sorted_by_dev = sorted(all_heads, key=lambda x: x['deviation'], reverse=True)
        
        # Get top-k heads (for coloring) :positive deviaton - good heads;
        positive_heads = [h for h in sorted_by_dev if h['deviation'] > 0]
        negative_heads = [h for h in sorted_by_dev if h['deviation'] < 0]
        
        if len(positive_heads) >= k_heads:
            top_k_heads_viz = positive_heads[:k_heads]
        else:
            top_k_heads_viz = positive_heads + negative_heads[:k_heads - len(positive_heads)]
        
        # Get bottom-k heads
        sorted_ascending = sorted(all_heads, key=lambda x: x['deviation'])
        neg_heads = [h for h in sorted_ascending if h['deviation'] < 0]
        pos_heads = [h for h in sorted_ascending if h['deviation'] > 0]
        
        if len(neg_heads) >= k_heads:
            bottom_k_heads = neg_heads[:k_heads]
        else:
            bottom_k_heads = neg_heads + pos_heads[:k_heads - len(neg_heads)]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Deviation Profile (top)
        ax1 = plt.subplot(3, 1, 1)
        
        x_labels = [f"L{h['layer']}_H{h['head']}" for h in all_heads]
        deviations = [h['deviation'] for h in all_heads]
        
        # Create color map
        colors = []
        for h in all_heads:
            if h in top_k_heads_viz:
                colors.append('green')
            elif h in bottom_k_heads:
                colors.append('red')
            else:
                colors.append('blue')
        
        bars = ax1.bar(range(len(x_labels)), deviations, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(x_labels)))
        ax1.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontweight='bold')  # Increased
        ax1.set_ylabel('Deviation from Baseline', fontsize=14, fontweight='bold')  # Increased
        ax1.set_title(f'Graph {graph_idx}: Head Deviation Profile', fontsize=16, fontweight='bold')  # Increased
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        # Increase y-axis tick label size
        ax1.tick_params(axis='y', labelsize=12)
        # Update legend with larger font
        green_patch = mpatches.Patch(color='green', label=f'Top-{k_heads} Heads')
        red_patch = mpatches.Patch(color='red', label=f'Bottom-{k_heads} Heads')
        blue_patch = mpatches.Patch(color='blue', label='Other Heads')
        ax1.legend(handles=[green_patch, red_patch, blue_patch], 
          fontsize=12, loc='upper left')  # Increased font, specified location
        
        # MAKE MORE COMPACT - reduce subplot spacing
        plt.subplots_adjust(hspace=0.3)  # Reduce vertical spacing between subplots
        
        # 2. Attention Heatmaps - Already correct size!
        for i, (layer_idx, head_idx) in enumerate(top_heads[:k_heads]):
            ax = plt.subplot(3, k_heads, 2*k_heads + 1 + i)
            
            head_key = f'L{layer_idx}_H{head_idx}'
            if head_key in attention_matrices:
                attn_weights = attention_matrices[head_key]
                
                # Get actual size of attention matrix (already correct!)
                actual_size = attn_weights.shape[0]
                
                # Plot with proper extent
                im = ax.imshow(attn_weights, cmap='hot', interpolation='nearest',
                              extent=[0, actual_size, actual_size, 0])
                
                # Find deviation for this head
                deviation = graph_devs[layer_idx][head_idx]
                
                # Show actual node count in title
                ax.set_title(f'L{layer_idx}_H{head_idx}\nδ={deviation:.3f}\nNodes: {actual_size}', 
                           fontsize=14)
                ax.set_xlabel('Target', fontsize=12)
                ax.set_ylabel('Source', fontsize=12)
                
                ax.set_xlim(0, actual_size)
                ax.set_ylim(actual_size, 0)
                
                # Adjust tick marks for readability
                if actual_size <= 50:
                    tick_spacing = 10
                elif actual_size <= 100:
                    tick_spacing = 20
                else:
                    tick_spacing = 50
                
                ax.set_xticks(range(0, actual_size + 1, tick_spacing))
                ax.set_yticks(range(0, actual_size + 1, tick_spacing))
                
                ax.grid(True, alpha=0.2, linestyle='--')
                
                # Add colorbar with max value
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Add max value annotation
                max_val = attn_weights.max()
                ax.text(0.02, 0.98, f'Max: {max_val:.3f}', 
                       transform=ax.transAxes, fontsize=7,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Attention\nData', ha='center', va='center')
                ax.set_title(f'L{layer_idx}_H{head_idx}', fontsize=9)
                ax.axis('off')
        
        # Update main title to show actual graph size
        plt.suptitle(f'Graph {graph_idx} Analysis (Actual Size: {actual_graph_size} nodes)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(vis_dir / f'graph_{graph_idx}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved visualization for graph {graph_idx} (actual size: {actual_graph_size} nodes)")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Calculate pairwise Jaccard similarities
    if len(top_nodes_per_head) < 2:
        return 0.0, head_to_nodes if head_to_nodes else {}, 0.0, total_attn_dict, stdev_dict
    
    jaccard_scores = []
    for i in range(len(top_nodes_per_head)):#contains a list of sets
        for j in range(i + 1, len(top_nodes_per_head)):
            set_i = top_nodes_per_head[i]
            set_j = top_nodes_per_head[j]
            
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0
            
            jaccard_scores.append(jaccard)
    
    # Return average Jaccard similarity AND NEW METRICS
    focus = np.mean(jaccard_scores) if jaccard_scores else 0.0
    return focus, head_to_nodes, correlation, total_attn_dict, stdev_dict



def compute_all_head_deviations(checkpoint_path, test_loader):
    """
    Compute deviation for every head on every graph
    Handles regression, binary, multi-class, and multi-label tasks
    """
    import logging
    
    # Get baseline predictions
    logging.info("Getting baseline predictions...")
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    model, _ = create_model_from_checkpoint(checkpoint_path)
    logging.getLogger().setLevel(old_level)
    
    baseline_preds, true_labels = evaluate_model_on_dataset(model, test_loader)
    del model
    torch.cuda.empty_cache()
    
    # Get model structure
    logging.getLogger().setLevel(logging.WARNING)
    model, _ = create_model_from_checkpoint(checkpoint_path)
    logging.getLogger().setLevel(old_level)
    
    if hasattr(model, 'model'):
        if hasattr(model.model, 'discrete_sequential'):
            layers = list(model.model.discrete_sequential)
        else:
            layers = model.model.layers
    else:
        layers = model.layers
    
    # Count layers with attention
    layer_indices = []
    for idx, layer in enumerate(layers):
        if hasattr(layer, 'num_heads'):
            layer_indices.append((idx, layer.num_heads))
    
    del model
    torch.cuda.empty_cache()
    
    # Store all deviations
    num_graphs = len(baseline_preds)
    all_deviations = {}
    
    # Initialize structure
    for graph_idx in range(num_graphs):
        all_deviations[graph_idx] = {}
    
    # Determine task type for error calculation
    task_type = cfg.dataset.task_type
    is_binary = getattr(cfg.dataset, 'is_binary', None)
    
    logging.info(f"Computing deviations for task: {task_type}, is_binary: {is_binary}")
    
    # Test each head
    logging.info("Testing all attention heads...")
    for layer_idx, num_heads in tqdm(layer_indices, desc="Layers"):
        for head_idx in range(num_heads):
            # Create fresh model (suppress logs)
            logging.getLogger().setLevel(logging.WARNING)
            model, _ = create_model_from_checkpoint(checkpoint_path)
            logging.getLogger().setLevel(old_level)
            
            # Mask this head
            mask_attention_head(model, layer_idx, head_idx)
            
            # Get predictions with masked head
            masked_preds, _ = evaluate_model_on_dataset(model, test_loader)
            
            # Calculate deviations for all graphs
            for graph_idx in range(num_graphs):
                true_val = true_labels[graph_idx]
                base_pred = baseline_preds[graph_idx]
                masked_pred = masked_preds[graph_idx]
                
                # Task-aware error calculation
                # Task-aware error calculation
                if task_type == 'regression':
                    # FIX: Handle multi-dimensional regression outputs
                    if isinstance(base_pred, np.ndarray) and base_pred.ndim > 0:
                        # Multi-target regression: average across all targets
                        base_error = np.mean(np.abs(base_pred - true_val))
                        masked_error = np.mean(np.abs(masked_pred - true_val))
                    else:
                        # Single-target regression
                        base_error = abs(float(base_pred) - float(true_val))
                        masked_error = abs(float(masked_pred) - float(true_val))
                    
                elif task_type == 'classification_multilabel' or is_binary == 'multi_label':
                    # Multi-label: Hamming distance (average error across labels)
                    # Convert predictions to binary using sigmoid
                    base_pred_binary = (torch.sigmoid(torch.tensor(base_pred)) > 0.5).float().numpy()
                    masked_pred_binary = (torch.sigmoid(torch.tensor(masked_pred)) > 0.5).float().numpy()
                    
                    # Handle NaN values in true labels (if any)
                    if np.any(np.isnan(true_val)):
                        mask = ~np.isnan(true_val)
                        if mask.sum() > 0:
                            base_error = np.mean(np.abs(base_pred_binary[mask] - true_val[mask]))
                            masked_error = np.mean(np.abs(masked_pred_binary[mask] - true_val[mask]))
                        else:
                            base_error = 0.0
                            masked_error = 0.0
                    else:
                        base_error = np.mean(np.abs(base_pred_binary - true_val))
                        masked_error = np.mean(np.abs(masked_pred_binary - true_val))
                    
                elif task_type in ['classification_binary', 'classification_multi', 'classification']:
                    if is_binary == 'binary' or (len(base_pred.shape) == 0 or 
                                                  (len(base_pred.shape) == 1 and base_pred.shape[0] == 1)):
                        # Binary classification - USE SOFT PREDICTIONS
                        base_pred_scalar = base_pred.item() if hasattr(base_pred, 'item') else float(base_pred)
                        masked_pred_scalar = masked_pred.item() if hasattr(masked_pred, 'item') else float(masked_pred)
                        true_val_scalar = true_val.item() if hasattr(true_val, 'item') else float(true_val)
                        
                        # Use probability difference instead of hard classification
                        base_prob = torch.sigmoid(torch.tensor(base_pred_scalar)).item()
                        masked_prob = torch.sigmoid(torch.tensor(masked_pred_scalar)).item()
                        
                        # Error = distance from true probability (0 or 1)
                        base_error = abs(base_prob - true_val_scalar)
                        masked_error = abs(masked_prob - true_val_scalar)
                        
                    else:
                        # Multi-class classification
                        if len(base_pred.shape) > 0 and base_pred.shape[0] > 1:
                            base_pred_class = np.argmax(base_pred)
                            masked_pred_class = np.argmax(masked_pred)
                        else:
                            base_pred_class = int(base_pred)
                            masked_pred_class = int(masked_pred)
                        
                        true_val_int = int(true_val.item() if hasattr(true_val, 'item') else true_val)
                        
                        base_error = float(base_pred_class != true_val_int)
                        masked_error = float(masked_pred_class != true_val_int)
                
                else:
                    # Fallback to regression-style error
                    logging.warning(f"Unknown task type {task_type}, using regression error")
                    base_error = abs(float(base_pred) - float(true_val))
                    masked_error = abs(float(masked_pred) - float(true_val))
                
                # Deviation = masked_error - baseline_error
                deviation = masked_error - base_error
                # FIX: Ensure deviation is always a scalar
                deviation = float(deviation) if not isinstance(deviation, (int, float)) else deviation

                # Store deviation
                if layer_idx not in all_deviations[graph_idx]:
                    all_deviations[graph_idx][layer_idx] = {}
                all_deviations[graph_idx][layer_idx][head_idx] = deviation
            
            # Clean up
            del model
            torch.cuda.empty_cache()
    
    return all_deviations, baseline_preds, true_labels


def save_per_graph_results(graph_idx, specialization, focus, head_deviations, 
                          top_k_heads, bottom_k_heads, baseline_pred, true_label, save_dir,
                          head_to_nodes, correlation, total_attn_dict, stdev_dict,head_entropy_dict=None):
    """Save individual graph results to JSON - NOW WITH NEW METRICS"""
    graph_results_dir = save_dir / 'graph_results'
    graph_results_dir.mkdir(parents=True, exist_ok=True)
    
    # FIX: Handle multi-dimensional arrays
    if isinstance(baseline_pred, np.ndarray) and baseline_pred.ndim > 0:
        baseline_pred = baseline_pred.tolist()  # Convert array to list
        true_label = true_label.tolist() if isinstance(true_label, np.ndarray) else true_label
        base_error = float(np.mean(np.abs(np.array(baseline_pred) - np.array(true_label))))
    else:
        baseline_pred = float(baseline_pred) if hasattr(baseline_pred, 'item') else float(baseline_pred)
        true_label = float(true_label) if hasattr(true_label, 'item') else float(true_label)
        base_error = abs(baseline_pred - true_label)
    
    # Convert top_k_heads tuples (might contain numpy floats)
    top_k_heads_clean = []
    for head_name, deviation in top_k_heads:
        dev_float = float(deviation) if hasattr(deviation, 'item') else float(deviation)
        top_k_heads_clean.append((head_name, dev_float))
    
    # Convert bottom_k_heads tuples
    bottom_k_heads_clean = []
    for head_name, deviation in bottom_k_heads:
        dev_float = float(deviation) if hasattr(deviation, 'item') else float(deviation)
        bottom_k_heads_clean.append((head_name, dev_float))
    
    graph_result = {
        'graph_id': int(graph_idx),
        'baseline_prediction': baseline_pred,
        'true_label': true_label,
        'baseline_error': float(base_error),
        'specialization': float(specialization),
        'focus': float(focus),
        'correlation': float(correlation),  # NEW
        'total_attention_per_head': total_attn_dict,  # NEW
        'stdev_attention_per_head': stdev_dict,  # NEW
        'entropy_per_head': head_entropy_dict if head_entropy_dict else {},
        'head_deviations': head_deviations,
        'top_k_heads': top_k_heads_clean,
        'bottom_k_heads': bottom_k_heads_clean,
        'num_heads_tested': len(head_deviations),
        'interpretability': categorize_interpretability(specialization, focus),
        'task_type': cfg.dataset.task_type,
        'is_binary': cfg.dataset.is_binary if hasattr(cfg.dataset, 'is_binary') else None,
        'top_nodes_per_head': head_to_nodes,  # Dict mapping head names to their top node indices
    }
    
    # Save JSON
    graph_file = graph_results_dir / f'graph_{graph_idx}_result.json'
    with open(graph_file, 'w') as f:
        json.dump(graph_result, f, indent=2)




def categorize_interpretability(specialization, focus, threshold_spec=0.1, threshold_focus=0.3):
    """Categorize into one of 4 interpretability scenarios"""
    high_spec = specialization > threshold_spec
    high_focus = focus > threshold_focus
    
    if high_spec and high_focus:
        return "High Spec + High Focus: Clear interpretability"
    elif high_spec and not high_focus:
        return "High Spec + Low Focus: Multiple mechanisms"
    elif not high_spec and high_focus:
        return "Low Spec + High Focus: Distributed but consistent"
    else:
        return "Low Spec + Low Focus: Challenging interpretation"


def visualize_aggregate_results(specializations, focus_scores, save_dir):
    """Create aggregate visualizations"""
    # Scatter plot of Specialization vs Focus
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(specializations, focus_scores, alpha=0.6, s=50, 
                        c=range(len(specializations)), cmap='viridis')
    ax.set_xlabel('Specialization (std of head deviations)', fontsize=12)
    ax.set_ylabel('Focus (node overlap among top heads)', fontsize=12)
    ax.set_title(f'Interpretability Analysis of all graphs', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    spec_threshold = np.median(specializations)
    focus_threshold = np.median(focus_scores)
    ax.axvline(x=spec_threshold, color='red', linestyle='--', alpha=0.5, 
              label=f'Median Spec: {spec_threshold:.6f}')
    ax.axhline(y=focus_threshold, color='blue', linestyle='--', alpha=0.5, 
              label=f'Median Focus: {focus_threshold:.6f}')
    
    ax.legend()
    plt.colorbar(scatter, label='Graph Index')
    plt.tight_layout()
    plt.savefig(save_dir / 'specialization_vs_focus.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(specializations, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(specializations), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(specializations):.6f}')
    axes[0].set_xlabel('Specialization')
    axes[0].set_ylabel('Number of Graphs')
    axes[0].set_title('Specialization Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(focus_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(focus_scores), color='red', linestyle='--',
                    label=f'Mean: {np.mean(focus_scores):.6f}')
    axes[1].set_xlabel('Focus')
    axes[1].set_ylabel('Number of Graphs')
    axes[1].set_title('Focus Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
def run_pk_analysis(model, train_loader, test_loader):
    """
    Main entry point - Complete implementation WITH NEW METRICS
    """
    logging.info("="*60)
    logging.info("Starting PK-Explainer Analysis (Complete Version)")
    logging.info("="*60)
    logging.info(f"Dataset: {cfg.dataset.name}")
    logging.info(f"Model type: {cfg.model.type}")
    
    # Get k value from config
    k_heads = getattr(cfg.gt.pk_explainer, 'k_heads', 5)  # Default to 5 if not specified
    
    # Step 1: Save model checkpoint
    checkpoint_path = Path(cfg.run_dir) / 'pk_checkpoint.pt'
    optimal_weights = None
    if hasattr(model, 'model') and hasattr(model.model, 'optimal_weights_dict'):
        optimal_weights = model.model.optimal_weights_dict
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimal_weights': optimal_weights
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Step 2: Compute all head deviations
    all_deviations, baseline_preds, true_labels = compute_all_head_deviations(checkpoint_path, test_loader)
    # Create results directory early (needed for entropy visualization)
    vis_dir = Path(cfg.run_dir) / 'pk_explainer_results'
    vis_dir.mkdir(parents=True, exist_ok=True)
    # Step 2.5: Compute entropy-deviation correlations (proves visualization paradox)
    logging.info("Computing entropy-deviation correlations...")
    entropy_dev_correlations, per_graph_entropy = compute_entropy_deviation_correlation(
        model, test_loader, checkpoint_path, all_deviations
    )
    visualize_entropy_deviation_analysis(entropy_dev_correlations, vis_dir)
    
    median_entropy_corr = np.median(entropy_dev_correlations)
    mean_entropy_corr = np.mean(entropy_dev_correlations)
    std_entropy_corr = np.std(entropy_dev_correlations)
    
    logging.info(f"Entropy-Deviation Correlation: {median_entropy_corr:.3f} (median), {mean_entropy_corr:.3f} (mean)")
    
    # Step 3: Calculate metrics for each graph
    num_graphs = len(all_deviations)
    specializations = []
    focus_scores = []
    correlation_scores = []  # NEW
    vis_dir = Path(cfg.run_dir) / 'pk_explainer_results'
    
    logging.info(f"Calculating metrics for {num_graphs} graphs...")
    
    for graph_idx in tqdm(range(num_graphs), desc="Computing metrics"):
        # Get all deviations for this graph
        graph_devs = []
        head_deviations = {}
        
        for layer_idx in all_deviations[graph_idx]:
            for head_idx in all_deviations[graph_idx][layer_idx]:
                dev = all_deviations[graph_idx][layer_idx][head_idx]
                dev_float = float(dev) if hasattr(dev, 'item') else float(dev)
                graph_devs.append(dev_float)
                head_deviations[f'L{layer_idx}_H{head_idx}'] = dev_float
        
        # Calculate Specialization
        if graph_devs:
            specialization = np.std(graph_devs)
        else:
            specialization = 0.0
        specializations.append(specialization)
        
        # Get all heads with their deviations
        head_info = []
        for layer_idx in all_deviations[graph_idx]:
            for head_idx in all_deviations[graph_idx][layer_idx]:
                dev = all_deviations[graph_idx][layer_idx][head_idx]
                dev_float = float(dev) if hasattr(dev, 'item') else float(dev)
                head_info.append((layer_idx, head_idx, dev_float))
        
        # Sort all heads by deviation
        head_info_sorted = sorted(head_info, key=lambda x: x[2], reverse=True)  # Sorted by actual deviation value
        
        # Get top-k heads (most beneficial or least harmful)
        positive_devs = [h for h in head_info_sorted if h[2] > 0]
        negative_devs = [h for h in head_info_sorted if h[2] < 0]
        
        if len(positive_devs) >= k_heads:
            top_k_info = positive_devs[:k_heads]
        else:
            # Not enough positive, fill with least negative
            top_k_info = positive_devs + negative_devs[:k_heads - len(positive_devs)]
        
        # Get bottom-k heads (most harmful or least beneficial)
        negative_devs_reversed = sorted(head_info, key=lambda x: x[2])  # Sort ascending (most negative first)
        neg_devs = [h for h in negative_devs_reversed if h[2] < 0]
        pos_devs = [h for h in negative_devs_reversed if h[2] > 0]
        
        if len(neg_devs) >= k_heads:
            bottom_k_info = neg_devs[:k_heads]
        else:
            # Not enough negative, fill with least positive
            bottom_k_info = neg_devs + pos_devs[:k_heads - len(neg_devs)]
        
        # For focus calculation - use only top-k heads
        top_k_heads = [(h[0], h[1]) for h in top_k_info]
        top_k_heads_with_dev = [(f'L{h[0]}_H{h[1]}', h[2]) for h in top_k_info]
        bottom_k_heads_with_dev = [(f'L{h[0]}_H{h[1]}', h[2]) for h in bottom_k_info]
        
        # Calculate Focus AND NEW METRICS (now with visualization if needed)
        vis_graphs = None
        if hasattr(cfg.gt.pk_explainer, 'graph_ids') and cfg.gt.pk_explainer.graph_ids:
            vis_graphs = cfg.gt.pk_explainer.graph_ids
        
        focus, head_to_nodes, correlation, total_attn_dict, stdev_dict = compute_focus_for_graph(
            None, test_loader, graph_idx, top_k_heads, checkpoint_path,
            all_deviations=all_deviations,  # Pass deviations for visualization
            vis_graphs=vis_graphs,  # Pass which graphs to visualize
            k_heads=k_heads  # Pass k value
        )
        focus_scores.append(focus)
        correlation_scores.append(correlation)  # NEW
        
        # Save per-graph results WITH NEW METRICS
        save_per_graph_results(
            graph_idx, specialization, focus, head_deviations, 
            top_k_heads_with_dev, bottom_k_heads_with_dev,
            baseline_preds[graph_idx], 
            true_labels[graph_idx], vis_dir,
            head_to_nodes,   
            correlation,  # NEW
            total_attn_dict,  # NEW
            stdev_dict,# NEW
            per_graph_entropy.get(graph_idx, {})# NEW
        )
    
    # Step 5: Dataset-level statistics WITH NEW METRICS
    mean_spec = np.mean(specializations)
    std_spec = np.std(specializations)
    mean_focus = np.mean(focus_scores)
    std_focus = np.std(focus_scores)
    mean_corr = np.mean(correlation_scores)  # NEW
    std_corr = np.std(correlation_scores)  # NEW
    
    logging.info(f"\n=== RESULTS ===")
    logging.info(f"Specialization: {mean_spec:.6f} +/- {std_spec:.6f}")
    logging.info(f"Focus: {mean_focus:.6f} +/- {std_focus:.6f}")
    logging.info(f"Correlation: {mean_corr:.6f} +/- {std_corr:.6f}")  # NEW
    logging.info(f"Entropy-Deviation Corr: {median_entropy_corr:.3f} +/- {std_entropy_corr:.3f}")
    # Step 6: Aggregate visualizations
    visualize_aggregate_results(specializations, focus_scores, vis_dir)
    
    # Step 7: Save summary WITH NEW METRICS
    overall_category = categorize_interpretability(mean_spec, mean_focus)
    
    results = {
        'dataset': cfg.dataset.name,
        'model_type': cfg.model.type,
        'num_graphs': num_graphs,
        'metrics': {
            'specialization_mean': float(mean_spec),
            'specialization_std': float(std_spec),
            'focus_mean': float(mean_focus),
            'focus_std': float(std_focus),
            'correlation_mean': float(mean_corr),  # NEW
            'correlation_std': float(std_corr),  # NEW
            'entropy_deviation_correlation_median': float(median_entropy_corr),
            'entropy_deviation_correlation_mean': float(mean_entropy_corr),
            'entropy_deviation_correlation_std': float(std_entropy_corr)
        },
        'interpretability_category': overall_category
    }
    
    results_file = Path(cfg.run_dir) / 'pk_explainer_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to {results_file}")
    logging.info(f"Per-graph results saved in {vis_dir / 'graph_results'}")
    logging.info("="*60)
    
    # Clean up
    checkpoint_path.unlink()
    
    return results


