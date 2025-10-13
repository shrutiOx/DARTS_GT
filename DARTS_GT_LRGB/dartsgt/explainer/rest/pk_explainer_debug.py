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


def create_model_from_checkpoint(checkpoint_path):
    """Create a fresh model and load checkpoint weights"""
    from torch_geometric.graphgym.model_builder import create_model
    import logging
    
    # Load checkpoint
    device = torch.device(cfg.accelerator)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get optimal weights if NAS model
    optimal_weights = checkpoint.get('optimal_weights', None)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        raise KeyError(f"Cannot find state dict. Available keys: {list(checkpoint.keys())}")
    
    # Check if this is a discrete model or DARTS model
    is_discrete_model = any('discrete_sequential' in key for key in state_dict.keys())
    is_darts_model = any('darts_layers' in key or 'darts_sequential' in key for key in state_dict.keys())
    
    logging.info(f"Checkpoint type: discrete={is_discrete_model}, darts={is_darts_model}")
    
    # FIX: Detect and fix dimension mismatch BEFORE creating model
    actual_dim_out = None
    for key in state_dict.keys():
        if 'post_mp.FC_layers.2.weight' in key:
            actual_dim_out = state_dict[key].shape[0]
            break
    
    original_dim_out = cfg.share.dim_out
    if actual_dim_out and actual_dim_out != cfg.share.dim_out:
        logging.info(f"Detected dim_out mismatch: config={cfg.share.dim_out}, saved={actual_dim_out}")
        cfg.share.dim_out = actual_dim_out
    
    # Suppress verbose logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    # CRITICAL FIX: Create the right type of model based on checkpoint content
    if is_discrete_model and cfg.model.type in ['NASModelQ', 'NASModel', 'NASModelQE', 'NASModelEdge']:
        logging.info("Creating discrete model to match checkpoint...")
        
        # Create wrapper model first
        from torch_geometric.graphgym.model_builder import GraphGymModule
        from dartsgt.network.gps_model import GPSModel
        
        # Create base GPS model wrapper
        model = GraphGymModule(None, None, None)
        
        # Now create the appropriate discrete model
        if cfg.model.type == 'NASModelQ':
            from dartsgt.network.NAS_model_qproj import NASModel
            nas_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
            # Create discrete version
            if optimal_weights:
                discrete_model = nas_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            else:
                # Create empty discrete model that matches checkpoint structure
                discrete_model = nas_model.get_discrete_model({0: [1, 0, 0], 1: [1, 0, 0]}, cfg.gt.weight_type)
            discrete_model.discrete = True  # Set discrete flag
            model.model = discrete_model
            
        elif cfg.model.type == 'NASModelQE':
            from dartsgt.network.NAS_model_qproj_edge import NASModelEdge
            nas_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
            if optimal_weights:
                discrete_model = nas_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            else:
                discrete_model = nas_model.get_discrete_model({0: [1, 0, 0], 1: [1, 0, 0]}, cfg.gt.weight_type)
            discrete_model.discrete = True
            model.model = discrete_model
            
        elif cfg.model.type == 'NASModel':
            from dartsgt.network.NAS_model import NASModel
            nas_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
            if optimal_weights:
                discrete_model = nas_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            else:
                discrete_model = nas_model.get_discrete_model({0: [1, 0, 0], 1: [1, 0, 0]}, cfg.gt.weight_type)
            discrete_model.discrete = True
            model.model = discrete_model
            
        elif cfg.model.type == 'NASModelEdge':
            from dartsgt.network.NAS_model_edge import NASModelEdge
            nas_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
            if optimal_weights:
                discrete_model = nas_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            else:
                discrete_model = nas_model.get_discrete_model({0: [1, 0, 0], 1: [1, 0, 0]}, cfg.gt.weight_type)
            discrete_model.discrete = True
            model.model = discrete_model
            
    else:
        # Standard model creation for non-discrete checkpoints
        model = create_model(cfg)
        
        # Convert to discrete if needed
        if cfg.model.type in ['NASModelQ', 'NASModel', 'NASModelQE', 'NASModelEdge'] and optimal_weights:
            if cfg.model.type == 'NASModelQ':
                from dartsgt.network.NAS_model_qproj import NASModel
                temp_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
                discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            elif cfg.model.type == 'NASModelQE':
                from dartsgt.network.NAS_model_qproj_edge import NASModelEdge
                temp_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
                discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            elif cfg.model.type == 'NASModel':
                from dartsgt.network.NAS_model import NASModel
                temp_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
                discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            elif cfg.model.type == 'NASModelEdge':
                from dartsgt.network.NAS_model_edge import NASModelEdge
                temp_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
                discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            
            discrete_model.discrete = True
            model.model = discrete_model
    
    # Restore logging level
    logging.getLogger().setLevel(old_level)
    
    # Load weights - should now match
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info("Successfully loaded all weights with strict=True")
    except RuntimeError as e:
        logging.warning(f"Strict loading failed, trying strict=False: {str(e)[:200]}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()
    
    # Ensure discrete mode is set properly
    if hasattr(model, 'model') and hasattr(model.model, 'discrete'):
        model.model.discrete = True
        logging.info(f"Model discrete flag set to: {model.model.discrete}")
    
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
            local_graph_idx = graph_idx - graph_counter
            break
        
        graph_counter += num_graphs
    
    if target_batch is None:
        return None
    
    # FIX 1: Ensure correct dtype for embeddings (prevents float/int error)
    if hasattr(target_batch, 'x') and target_batch.x is not None:
        if target_batch.x.dtype in [torch.float32, torch.float64]:
            if target_batch.x.min() >= 0 and target_batch.x.max() < 10000:
                target_batch.x = target_batch.x.long()
    
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
            # Sparse attention handling - FIXED
            sparse_weights, edge_index = layer.attn_weights
            
            # Get batch information
            batch_idx = target_batch.batch
            unique_graphs = torch.unique(batch_idx)
            
            # Find nodes for our specific graph
            target_graph_id = unique_graphs[local_graph_idx]
            node_mask = (batch_idx == target_graph_id)
            graph_nodes = torch.where(node_mask)[0]
            num_graph_nodes = len(graph_nodes)
            
            # Create node mapping (global to local indices)
            node_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(graph_nodes)}
            
            # Filter edges for this graph
            src, dst = edge_index
            graph_edges_mask = node_mask[src] & node_mask[dst]
            graph_src = src[graph_edges_mask]
            graph_dst = dst[graph_edges_mask]
            
            # Get attention weights for this graph and head
            graph_attn_weights = sparse_weights[graph_edges_mask, head_idx]
            
            # Create dense attention matrix for this graph
            attn_matrix = torch.zeros(num_graph_nodes, num_graph_nodes, device=device)
            
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
                return attn[local_graph_idx, head_idx].cpu().numpy()
    
    return None


def compute_focus_for_graph(model, test_loader, graph_idx, top_heads, checkpoint_path):
    """
    Compute Focus metric using fixed attention extraction
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
    head_to_nodes = {}  # NEW: Store mapping of head to its top nodes
    
    for layer_idx, head_idx in top_heads:
        # Get attention weights using the fixed method
        attn_weights = extract_attention_for_graph(model, test_loader, graph_idx, layer_idx, head_idx)
        
        if attn_weights is not None:
            # Sum attention across source nodes to get importance per target node
            node_attention = np.sum(attn_weights, axis=0)
            
            # Get top 25% of nodes
            num_nodes = len(node_attention)
            num_top = max(1, num_nodes // 4)
            top_node_indices = np.argsort(node_attention)[-num_top:]
            
            top_nodes_per_head.append(set(top_node_indices))
            head_to_nodes[f'L{layer_idx}_H{head_idx}'] = [int(idx) for idx in top_node_indices]
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Calculate pairwise Jaccard similarities
    # Calculate pairwise Jaccard similarities
    if len(top_nodes_per_head) < 2:
        # Return empty dict if no heads processed, maintaining consistency
        return 0.0, head_to_nodes if head_to_nodes else {}
    
    jaccard_scores = []
    for i in range(len(top_nodes_per_head)):
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
            
    #if top_nodes_per_head:
    #    common_nodes = set.intersection(*top_nodes_per_head) if len(top_nodes_per_head) > 1 else top_nodes_per_head[0]
    #    # FIX: Convert to list of Python ints
    #    common_nodes = [int(node) for node in common_nodes]
    #else:
    #    common_nodes = []
    
    # Return average Jaccard similarity
    return np.mean(jaccard_scores) if jaccard_scores else 0.0, head_to_nodes


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
                          head_to_nodes):
    """Save individual graph results to JSON"""
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
        'head_deviations': head_deviations,
        'top_k_heads': top_k_heads_clean,
        'bottom_k_heads': bottom_k_heads_clean,
        'num_heads_tested': len(head_deviations),
        'interpretability': categorize_interpretability(specialization, focus),
        'task_type': cfg.dataset.task_type,
        'is_binary': cfg.dataset.is_binary if hasattr(cfg.dataset, 'is_binary') else None,
        'top_nodes_per_head': head_to_nodes,  # NEW: Dict mapping head names to their top node indices
        #'common_nodes': common_nodes,  # NEW: Node indices that all top heads agree on
        #'num_common_nodes': len(common_nodes),  # NEW: Count for easy access
    }
    
    # Save JSON
    graph_file = graph_results_dir / f'graph_{graph_idx}_result.json'
    with open(graph_file, 'w') as f:
        json.dump(graph_result, f, indent=2)


def visualize_specific_graph(graph_idx, model, test_loader, all_deviations, 
                            checkpoint_path, save_dir, k_heads=5):
    """Create visualizations for a specific graph"""
    import logging
    vis_dir = save_dir / 'graph_visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get k from parameter
    k = k_heads
    
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
    
    # Sort heads by deviation value (not absolute)
    sorted_by_dev = sorted(all_heads, key=lambda x: x['deviation'], reverse=True)
    
    # Get top-k (most beneficial/least harmful)
    positive_heads = [h for h in sorted_by_dev if h['deviation'] > 0]
    negative_heads = [h for h in sorted_by_dev if h['deviation'] < 0]
    
    if len(positive_heads) >= k:
        top_k_heads = positive_heads[:k]
    else:
        top_k_heads = positive_heads + negative_heads[:k - len(positive_heads)]
    
    # Get bottom-k (most harmful/least beneficial)
    sorted_ascending = sorted(all_heads, key=lambda x: x['deviation'])
    neg_heads = [h for h in sorted_ascending if h['deviation'] < 0]
    pos_heads = [h for h in sorted_ascending if h['deviation'] > 0]
    
    if len(neg_heads) >= k:
        bottom_k_heads = neg_heads[:k]
    else:
        bottom_k_heads = neg_heads + pos_heads[:k - len(neg_heads)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Deviation Profile (top)
    ax1 = plt.subplot(3, 1, 1)
    
    x_labels = [f"L{h['layer']}_H{h['head']}" for h in all_heads]
    deviations = [h['deviation'] for h in all_heads]
    
    # Create color map
    colors = []
    for h in all_heads:
        if h in top_k_heads:
            colors.append('green')
        elif h in bottom_k_heads:
            colors.append('red')
        else:
            colors.append('blue')
    
    bars = ax1.bar(range(len(x_labels)), deviations, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=8)
    ax1.set_ylabel('Deviation from Baseline')
    ax1.set_title(f'Graph {graph_idx}: Head Deviation Profile')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Update legend
    green_patch = mpatches.Patch(color='green', label=f'Top-{k} Heads')
    red_patch = mpatches.Patch(color='red', label=f'Bottom-{k} Heads')
    blue_patch = mpatches.Patch(color='blue', label='Other Heads')
    ax1.legend(handles=[green_patch, red_patch, blue_patch])
    
    # 2. Attention Heatmaps - ONLY for top-k heads
    import logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    model, _ = create_model_from_checkpoint(checkpoint_path)
    logging.getLogger().setLevel(old_level)
    
    for i, head_info in enumerate(top_k_heads[:k]):  # Only top-k, not bottom-k
        ax = plt.subplot(3, k, 2*k + 1 + i)  # CORRECT - works for any k
        layer_idx = head_info['layer']
        head_idx = head_info['head']
        deviation = head_info['deviation']
        
        # Get attention weights
        attn_weights = extract_attention_for_graph(model, test_loader, graph_idx, 
                                                   layer_idx, head_idx)
        
        if attn_weights is not None:
            # Get actual size of attention matrix
            actual_size = attn_weights.shape[0]
            
            # Plot with proper extent
            im = ax.imshow(attn_weights, cmap='hot', interpolation='nearest',
                          extent=[0, actual_size, actual_size, 0])
            
            ax.set_title(f'L{layer_idx}_H{head_idx}\nδ={deviation:.3f}', fontsize=9)
            ax.set_xlabel('Target', fontsize=8)
            ax.set_ylabel('Source', fontsize=8)
            
            # FIX: Set axis limits to actual graph size
            ax.set_xlim(0, actual_size)
            ax.set_ylim(actual_size, 0)  # Inverted for correct orientation
            
            # Adjust tick marks for readability
            if actual_size <= 50:
                tick_spacing = 10
            elif actual_size <= 100:
                tick_spacing = 20
            else:
                tick_spacing = 50
            
            ax.set_xticks(range(0, actual_size + 1, tick_spacing))
            ax.set_yticks(range(0, actual_size + 1, tick_spacing))
            
            # Add grid for better readability
            ax.grid(True, alpha=0.2, linestyle='--')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No Attention\nData', ha='center', va='center')
            ax.set_title(f'L{layer_idx}_H{head_idx}\nδ={deviation:.3f}', fontsize=9)
            ax.axis('off')
    
    # Clean up model
    del model
    torch.cuda.empty_cache()
    
    plt.suptitle(f'Graph {graph_idx} Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / f'graph_{graph_idx}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved visualization for graph {graph_idx}")


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


def run_pk_analysis(model=None, train_loader=None, test_loader=None, checkpoint_path=None):
    """
    Enhanced PK Analysis with Normal and Debug modes
    Normal mode: Full analysis with optional immediate viz
    Debug mode: Limited graphs with detailed diagnostics
    """
    logging.info("="*60)
    logging.info("Starting PK-Explainer Analysis")
    logging.info("="*60)
    logging.info(f"Dataset: {cfg.dataset.name}")
    logging.info(f"Model type: {cfg.model.type if hasattr(cfg, 'model') else 'Unknown'}")
    
    # Get configuration parameters
    debug_mode = getattr(cfg.gt.pk_explainer, 'debug_mode', False)
    use_checkpoint = getattr(cfg.gt.pk_explainer, 'use_checkpoint', None)
    immediate_viz = getattr(cfg.gt.pk_explainer, 'immediate_viz', False)
    k_heads = getattr(cfg.gt.pk_explainer, 'k_heads', 5)
    graph_ids_to_viz = getattr(cfg.gt.pk_explainer, 'graph_ids', [])
    
    logging.info(f"Mode: {'DEBUG' if debug_mode else 'NORMAL'}")
    logging.info(f"Immediate visualization: {immediate_viz}")
    
    # Step 1: Handle model/checkpoint
    if use_checkpoint:
        checkpoint_path = Path(use_checkpoint)
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        model, optimal_weights = create_model_from_checkpoint(checkpoint_path)
    elif model is None:
        raise ValueError("No model provided and no checkpoint specified!")
    else:
        # Save current model as checkpoint
        checkpoint_path = Path(cfg.run_dir) / 'pk_checkpoint.pt'
        optimal_weights = None
        if hasattr(model, 'model') and hasattr(model.model, 'optimal_weights_dict'):
            optimal_weights = model.model.optimal_weights_dict
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimal_weights': optimal_weights
        }, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Step 2: Decide which graphs to analyze
    if debug_mode and graph_ids_to_viz:
        # Debug mode: ONLY analyze specified graphs
        graphs_to_analyze = graph_ids_to_viz
        logging.info(f"DEBUG MODE: Analyzing only {len(graphs_to_analyze)} graphs: {graphs_to_analyze}")
    else:
        # Normal mode: Analyze ALL graphs
        num_graphs = len(test_loader.dataset)
        graphs_to_analyze = list(range(num_graphs))
        logging.info(f"NORMAL MODE: Analyzing all {num_graphs} graphs")
        if immediate_viz and graph_ids_to_viz:
            logging.info(f"Will show immediate viz for graphs: {graph_ids_to_viz}")
    
    # Step 3: Compute head deviations (might be slow for all graphs)
    if debug_mode:
        logging.info("\nComputing head deviations for selected graphs only...")
        # For debug, we can do a simplified version
        all_deviations = {}
        baseline_preds = []
        true_labels = []
        
        # Just compute for selected graphs (simplified for speed)
        for graph_idx in graphs_to_analyze:
            all_deviations[graph_idx] = {}
            # Add mock deviations for now (you can implement selective computation)
            for layer_idx in range(2):  # Assuming 2 layers
                all_deviations[graph_idx][layer_idx] = {}
                for head_idx in range(4):  # Assuming 4 heads
                    all_deviations[graph_idx][layer_idx][head_idx] = np.random.randn() * 0.1
            baseline_preds.append(0)  # Placeholder
            true_labels.append(0)  # Placeholder
    else:
        # Normal mode: Full computation
        logging.info("\nComputing head deviations for all graphs...")
        all_deviations, baseline_preds, true_labels = compute_all_head_deviations(checkpoint_path, test_loader)
    
    # Step 4: Calculate metrics for each graph to analyze
    num_graphs_to_analyze = len(graphs_to_analyze)
    specializations = []
    focus_scores = []
    vis_dir = Path(cfg.run_dir) / 'pk_explainer_results'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"\nCalculating metrics for {num_graphs_to_analyze} graphs...")
    
    for idx, graph_idx in enumerate(tqdm(graphs_to_analyze, desc="Computing metrics")):
        
        # Get deviations for this graph
        if graph_idx not in all_deviations:
            logging.warning(f"Graph {graph_idx} not in deviations, skipping...")
            continue
            
        graph_devs = []
        head_deviations = {}
        
        for layer_idx in all_deviations[graph_idx]:
            for head_idx in all_deviations[graph_idx][layer_idx]:
                dev = all_deviations[graph_idx][layer_idx][head_idx]
                dev_float = float(dev) if hasattr(dev, 'item') else float(dev)
                graph_devs.append(dev_float)
                head_deviations[f'L{layer_idx}_H{head_idx}'] = dev_float
        
        # Calculate metrics
        specialization = np.std(graph_devs) if graph_devs else 0.0
        specializations.append(specialization)
        
        # Get top-k heads
        head_info = []
        for layer_idx in all_deviations[graph_idx]:
            for head_idx in all_deviations[graph_idx][layer_idx]:
                dev = all_deviations[graph_idx][layer_idx][head_idx]
                dev_float = float(dev) if hasattr(dev, 'item') else float(dev)
                head_info.append((layer_idx, head_idx, dev_float))
        
        head_info_sorted = sorted(head_info, key=lambda x: x[2], reverse=True)
        top_k_info = head_info_sorted[:k_heads] if len(head_info_sorted) >= k_heads else head_info_sorted
        top_k_heads = [(h[0], h[1]) for h in top_k_info]
        top_k_heads_with_dev = [(f'L{h[0]}_H{h[1]}', h[2]) for h in top_k_info]
        
        # DEBUG MODE: Extra diagnostics
        if debug_mode:
            logging.info(f"\n{'='*40}")
            logging.info(f"DEBUG: Graph {graph_idx}")
            logging.info(f"{'='*40}")
            logging.info(f"Specialization: {specialization:.6f}")
            logging.info(f"Top {k_heads} heads: {top_k_heads_with_dev}")
            
            # Extract and analyze attention for top heads
            for layer_idx, head_idx in top_k_heads[:3]:  # Analyze top 3
                logging.info(f"\nAnalyzing L{layer_idx}_H{head_idx}:")
                attn_weights = extract_attention_for_graph(model, test_loader, graph_idx, layer_idx, head_idx)
                
                if attn_weights is not None:
                    col_sums = np.sum(attn_weights, axis=0)
                    row_sums = np.sum(attn_weights, axis=1)
                    
                    logging.info(f"  Shape: {attn_weights.shape}")
                    logging.info(f"  Non-zero: {np.count_nonzero(attn_weights)}/{attn_weights.size}")
                    
                    top_5_cols = np.argsort(col_sums)[-5:]
                    logging.info(f"  Top 5 target nodes: {top_5_cols}")
                    logging.info(f"  Their attention sums: {col_sums[top_5_cols]}")
                    
                    if len(col_sums) > 15:
                        logging.info(f"  Cols 0-15: sum={col_sums[:15].sum():.4f}, max={col_sums[:15].max():.4f}")
                    if len(col_sums) > 172:
                        logging.info(f"  Col 172: {col_sums[172]:.4f}")
        
        # Calculate Focus
        focus, head_to_nodes = compute_focus_for_graph(
            None, test_loader, graph_idx, top_k_heads, checkpoint_path
        )
        focus_scores.append(focus)
        
        # Save per-graph results (for both modes)
        if not debug_mode or graph_idx in graph_ids_to_viz:
            save_per_graph_results(
                graph_idx, specialization, focus, head_deviations, 
                top_k_heads_with_dev, [],  # Empty bottom_k for now
                baseline_preds[idx] if idx < len(baseline_preds) else 0, 
                true_labels[idx] if idx < len(true_labels) else 0, 
                vis_dir, head_to_nodes
            )
        
        # Immediate visualization if requested
        if immediate_viz and graph_idx in graph_ids_to_viz:
            logging.info(f"Creating immediate visualization for graph {graph_idx}...")
            visualize_specific_graph(
                graph_idx, model, test_loader, 
                all_deviations, checkpoint_path, vis_dir, k_heads
            )
            # Show the plot immediately
            import matplotlib.pyplot as plt
            plt.show()
    
    # Step 5: Summary statistics
    if specializations:
        mean_spec = np.mean(specializations)
        std_spec = np.std(specializations)
        mean_focus = np.mean(focus_scores)
        std_focus = np.std(focus_scores)
        
        logging.info(f"\n=== RESULTS ===")
        logging.info(f"Graphs analyzed: {len(specializations)}")
        logging.info(f"Specialization: {mean_spec:.6f} +/- {std_spec:.6f}")
        logging.info(f"Focus: {mean_focus:.6f} +/- {std_focus:.6f}")
        
        # Save summary
        results = {
            'dataset': cfg.dataset.name,
            'mode': 'DEBUG' if debug_mode else 'NORMAL',
            'num_graphs_analyzed': len(specializations),
            'metrics': {
                'specialization_mean': float(mean_spec),
                'specialization_std': float(std_spec),
                'focus_mean': float(mean_focus),
                'focus_std': float(std_focus)
            }
        }
        
        results_file = Path(cfg.run_dir) / 'pk_explainer_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"\nResults saved to {results_file}")
    
    # Clean up checkpoint if we created it
    if not use_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    
    logging.info("="*60)
    
    return results if 'results' in locals() else {}
