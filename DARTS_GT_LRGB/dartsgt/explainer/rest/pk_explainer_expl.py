import torch
import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader


def compute_importance_scores(model, train_loader) -> Dict[int, Dict[int, float]]:
    """
    Compute importance scores using PK-Explainer formulation
    Handles GPS, DARTS-GT, VANILLA models with both dense and sparse attention
    """
    model.eval()
    device = torch.device(cfg.accelerator)
    
    # Determine model structure - HANDLE ALL MODEL TYPES
    if hasattr(model, 'model'):  # GraphGym wrapper
        inner_model = model.model
        if hasattr(inner_model, 'discrete_sequential') and inner_model.discrete_sequential is not None:
            # NAS model in discrete mode
            layers = list(inner_model.discrete_sequential)
        elif hasattr(inner_model, 'layers'):
            # GPS/Vanilla models
            layers = inner_model.layers
        else:
            raise AttributeError(f"Model {type(inner_model)} has no recognizable layer structure")
    else:
        if hasattr(model, 'discrete_sequential') and model.discrete_sequential is not None:
            layers = list(model.discrete_sequential)
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise AttributeError(f"Model {type(model)} has no recognizable layer structure")
    
    num_layers = len(layers)
    layer_attentions = {i: [] for i in range(num_layers)}
    #print('layer_attentions ',len(layer_attentions))
    
    # Collect attention weights
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            _ = model(batch)
            
            # Collect attention from each layer:specific handling for sparse (tuple) vs dense
            for i, layer in enumerate(layers):
                if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
                    # Check if sparse (tuple) or dense (tensor)
                    if isinstance(layer.attn_weights, tuple):
                        # Sparse attention: (weights, edge_index)
                        sparse_weights, edge_index = layer.attn_weights
                        layer_attentions[i].append(('sparse', sparse_weights.cpu(), edge_index.cpu(), batch.batch.cpu()))
                    else:
                        # Dense attention: regular tensor
                        layer_attentions[i].append(layer.attn_weights.cpu())
    
    # Compute importance for each layer
    importance_scores = {}
    
    for layer_idx, attentions in layer_attentions.items():
        if not attentions:
            continue
        
        if isinstance(attentions[0], tuple) and attentions[0][0] == 'sparse':
            # SPARSE ATTENTION PROCESSING
            all_kl_divs = []
            
            for attn_idx, (_, sparse_weights, edge_index, batch_idx) in enumerate(attentions):
                # Process all graphs in batch together
                unique_graphs = torch.unique(batch_idx)
                
                # Compute S^h for all nodes at once
                num_nodes_total = batch_idx.size(0)
                num_heads = sparse_weights.size(1)
                #print('num_heads ',num_heads)
                #print('num_nodes_total ',num_nodes_total)
                
                S_h = torch.zeros(num_heads, num_nodes_total)
                #print('S_h shape ',S_h.shape)
                #print('S_h  before ',S_h)
                for head in range(num_heads):
                    S_h[head] = scatter_add(
                        sparse_weights[:, head],
                        edge_index[1],  # destination nodes (global indices)
                        dim=0,
                        dim_size=num_nodes_total
                    )
                    #print('S_h shape ',S_h[head].shape)
                    #print('S_h  after ',S_h)
                S_h = S_h + 1e-10  # Avoid division by zero
                
                # Now process each graph
                for graph_idx, graph_id in enumerate(unique_graphs):
                    node_mask = (batch_idx == graph_id)
                    graph_S_h = S_h[:, node_mask]  # Extract this graph's S_h
                    
                    # NORMALIZE TO PROBABILITY DISTRIBUTIONS (following POE formulation)
                    # P^h[j] = S^h[j] / sum_j S^h[j]
                    P_h = torch.zeros_like(graph_S_h)
                    for h in range(num_heads):
                        P_h[h] = graph_S_h[h] / (graph_S_h[h].sum() + 1e-10)
                    
                    # Compute Product of Experts: P[j] = prod_h P^h[j] normalized
                    # Use log-space for numerical stability
                    log_P_h = torch.log(P_h + 1e-20)
                    log_P_all = log_P_h.sum(dim=0)  # Sum in log space = product in normal space
                    #print('log_P_h ',log_P_h)
                    #print('log_P_all ',log_P_all)

                    
                    
                    # Normalize to get P[j]
                    P_tilde = torch.exp(log_P_all - log_P_all.max())  # Subtract max for stability
                    P_all = P_tilde / (P_tilde.sum() + 1e-10)
                    
                    # KL divergence for each head
                    head_kl_divs = []
                    for head_idx in range(num_heads):
                        # Compute P_{-λ}[j] = normalized product without head λ
                        log_P_minus = torch.zeros_like(log_P_h[0])
                        for h in range(num_heads):
                            if h != head_idx:
                                log_P_minus = log_P_minus + log_P_h[h]
                                #print('log_P_minus ',log_P_minus)
                        
                        # Normalize P_minus
                        P_minus_tilde = torch.exp(log_P_minus - log_P_minus.max())
                        P_minus = P_minus_tilde / (P_minus_tilde.sum() + 1e-10)
                        
                        # KL divergence: D_KL(P || P_{-λ})
                        kl_div = (P_all * torch.log(P_all / (P_minus + 1e-10))).sum()
                        #print('kl_div ',kl_div)
                        
                        
                        head_kl_divs.append(kl_div.item() if torch.isfinite(kl_div) else 0.0)
                    
                    all_kl_divs.append(head_kl_divs)
                    
        else:
            # DENSE ATTENTION PROCESSING
            all_kl_divs = []
            
            for attn_batch in attentions:
                batch_size, num_heads, num_nodes, _ = attn_batch.shape
                
                for b in range(batch_size):
                    # Get attention for this graph: [H, N, N]
                    attn_graph = attn_batch[b]
                    #print('attn_graph shape ',attn_graph.shape)
                    
                    # Compute S^h[j] = sum_i A^h[i,j] (pooled attention)
                    S_h = attn_graph.sum(dim=1)  # [H, N]
                    S_h = S_h + 1e-10  # Avoid division by zero
                    
                    #print('S_h  ',S_h)
                    
                    # NORMALIZE TO PROBABILITY DISTRIBUTIONS (following POE formulation)
                    # P^h[j] = S^h[j] / sum_j S^h[j]
                    P_h = torch.zeros_like(S_h)
                    for h in range(num_heads):
                        P_h[h] = S_h[h] / (S_h[h].sum() + 1e-10)
                    
                    # Compute Product of Experts: P[j] = prod_h P^h[j] normalized
                    # Use log-space for numerical stability
                    log_P_h = torch.log(P_h + 1e-20)
                    log_P_all = log_P_h.sum(dim=0)  # Sum in log space = product in normal space
                    
                    # Normalize to get P[j]
                    P_tilde = torch.exp(log_P_all - log_P_all.max())  # Subtract max for stability
                    P_all = P_tilde / (P_tilde.sum() + 1e-10)
                    
                    # For each head, compute KL divergence
                    head_kl_divs = []
                    for head_idx in range(num_heads):
                        # Compute P_{-λ}[j] = normalized product without head λ
                        log_P_minus = torch.zeros_like(log_P_h[0])
                        for h in range(num_heads):
                            if h != head_idx:
                                log_P_minus = log_P_minus + log_P_h[h]
                        
                        # Normalize P_minus
                        P_minus_tilde = torch.exp(log_P_minus - log_P_minus.max())
                        P_minus = P_minus_tilde / (P_minus_tilde.sum() + 1e-10)
                        
                        # KL divergence: D_KL(P || P_{-λ})
                        kl_div = (P_all * torch.log(P_all / (P_minus + 1e-10))).sum()
                        head_kl_divs.append(kl_div.item() if torch.isfinite(kl_div) else 0.0)
                    
                    all_kl_divs.append(head_kl_divs)
        
        # Average KL divergence across all graphs for each head
        if all_kl_divs:
            avg_kl_per_head = np.mean(all_kl_divs, axis=0)
            layer_importance = {h: avg_kl_per_head[h] for h in range(num_heads)}
            importance_scores[layer_idx] = layer_importance
    
    return importance_scores




def mask_attention_head(model, layer_idx: int, head_idx: int, device):
    """
    Mask a specific attention head by zeroing its weights
    """
    # Get layers based on model type
    if hasattr(model, 'model'):  # GraphGym wrapper
        inner_model = model.model
        if hasattr(inner_model, 'discrete_sequential') and inner_model.discrete_sequential is not None:
            layers = list(inner_model.discrete_sequential)
        elif hasattr(inner_model, 'layers'):
            layers = inner_model.layers
        else:
            raise AttributeError(f"Model {type(inner_model)} has no recognizable layer structure")
    else:
        if hasattr(model, 'discrete_sequential') and model.discrete_sequential is not None:
            layers = list(model.discrete_sequential)
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise AttributeError(f"Model {type(model)} has no recognizable layer structure")
    
    layer = layers[layer_idx]
    
    if hasattr(layer, 'self_attn') and layer.self_attn is not None:
        # Check if it's SparseGraphAttention
        if hasattr(layer, 'global_model_type') and layer.global_model_type == 'SparseTransformer':
            # SparseGraphAttention has W_k, W_v, W_o
            sparse_attn = layer.self_attn
            embed_dim = sparse_attn.dim_h
            head_dim = sparse_attn.head_dim
            
            #print('sparse_attn ',sparse_attn.shape)
            #print('embed_dim ',embed_dim.shape)
            #print('head_dim ',head_dim.shape)
            
            # Calculate slice for this head
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
            #print('start_idx ',start_idx)
            #print('end_idx ',end_idx)
            
            # Store original weights
            orig_weights = {
                'W_k': sparse_attn.W_k.weight.data.clone(),
                'W_v': sparse_attn.W_v.weight.data.clone(),
                'W_o': sparse_attn.W_o.weight.data.clone(),
            }
            
            # Zero out this head's weights
            # For K and V projections (output dimension)
            sparse_attn.W_k.weight.data[start_idx:end_idx, :] = 0
            sparse_attn.W_v.weight.data[start_idx:end_idx, :] = 0
            
            # For output projection (input dimension)
            sparse_attn.W_o.weight.data[:, start_idx:end_idx] = 0
            
            return orig_weights, None  # Return dict for sparse
            
        # For regular MultiheadAttention
        elif hasattr(layer.self_attn, 'in_proj_weight'):
            # Get dimensions
            embed_dim = layer.dim_h
            head_dim = embed_dim // layer.num_heads
            
            #print('embed_dim ',embed_dim.shape)
            #print('head_dim ',head_dim.shape)
            
            # Calculate slice for this head
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
           # print('start_idx ',start_idx)
           # print('end_idx ',end_idx)
            
            # Store original weights
            orig_in_proj = layer.self_attn.in_proj_weight.data.clone()
            orig_out_proj = layer.self_attn.out_proj.weight.data.clone()
            
            # Zero out this head's weights in Q, K, V projections
            layer.self_attn.in_proj_weight.data[start_idx:end_idx, :] = 0  # Q
            layer.self_attn.in_proj_weight.data[embed_dim + start_idx:embed_dim + end_idx, :] = 0  # K
            layer.self_attn.in_proj_weight.data[2*embed_dim + start_idx:2*embed_dim + end_idx, :] = 0  # V
            
            # Zero out this head's weights in output projection
            layer.self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0
            
            return orig_in_proj, orig_out_proj
    
    return None, None

def restore_attention_head(model, layer_idx: int, orig_in_proj, orig_out_proj):
    """
    Restore original weights after masking
    """
    # Get layers (same as mask_attention_head)
    if hasattr(model, 'model'):  # GraphGym wrapper
        inner_model = model.model
        if hasattr(inner_model, 'discrete_sequential') and inner_model.discrete_sequential is not None:
            layers = list(inner_model.discrete_sequential)
        elif hasattr(inner_model, 'layers'):
            layers = inner_model.layers
        else:
            return
    else:
        if hasattr(model, 'discrete_sequential') and model.discrete_sequential is not None:
            layers = list(model.discrete_sequential)
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            return
    
    layer = layers[layer_idx]
    
    if hasattr(layer, 'self_attn') and layer.self_attn is not None:
        # Check if sparse (orig_in_proj is dict)
        if isinstance(orig_in_proj, dict):
            # Restore SparseGraphAttention weights
            sparse_attn = layer.self_attn
            sparse_attn.W_k.weight.data = orig_in_proj['W_k']
            sparse_attn.W_v.weight.data = orig_in_proj['W_v']
            sparse_attn.W_o.weight.data = orig_in_proj['W_o']
        else:
            # Restore MultiheadAttention weights
            layer.self_attn.in_proj_weight.data = orig_in_proj
            layer.self_attn.out_proj.weight.data = orig_out_proj
        



def ablation_study(model_path, test_loader, importance_scores):
    """
    Perform ablation study by masking heads and measuring performance
    """
    device = torch.device(cfg.accelerator)
    
    # Import eval function based on training mode
    if cfg.train.mode == 'custom':
        from dartsgt.train.custom_train import eval_epoch
    elif cfg.train.mode == 'NoMixNas_uncertainty_train':
        from dartsgt.train.NoMixNas_uncertainty_train import eval_epoch_with_uncertainty as eval_epoch
    
    # Import the CustomLogger from your logger module
    from dartsgt.logger import CustomLogger
    from torch_geometric.graphgym.logger import infer_task
    
    # Create temporary logger using CustomLogger
    logger = CustomLogger(name='pk_ablation', task_type=infer_task())
    
    # Load checkpoint ONCE
    checkpoint = torch.load(model_path, map_location=device)
    optimal_weights = checkpoint.get('optimal_weights')
    
    # Helper function to create the right model
    def create_correct_model():
        from torch_geometric.graphgym.model_builder import create_model
        model = create_model(cfg)
        
        # If it's a discrete NAS model, swap the inner model
        if cfg.model.type == 'NASModelEdge' and optimal_weights is not None:
            from dartsgt.network.NAS_model_edge import NASModelEdge
            
            # Create DARTS model
            temp_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
            
            # Convert to discrete model
            discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            
            # Swap inner model in the GraphGym wrapper
            model.model = discrete_model
        
        return model
    
    # Get baseline performance
    logging.info("Getting baseline performance...")
    
    # Create model with correct structure
    model = create_correct_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.reset()
    eval_epoch(logger, test_loader, model, split='test')
    baseline_stats = logger.write_epoch(0)
    
    # Determine metric
    if cfg.metric_best == 'auto':
        if cfg.dataset.task_type == 'classification_multilabel':
            metric_key = 'ap'
        elif 'classification' in cfg.dataset.task_type:
            metric_key = 'accuracy'
        else:
            metric_key = 'mae'
    else:
        metric_key = cfg.metric_best
    
    baseline_perf = baseline_stats.get(metric_key, baseline_stats.get('loss'))
    logging.info(f"Baseline {metric_key}: {baseline_perf:.8f}")
    
    # Test each head
    ablation_results = {}
    
    for layer_idx, head_scores in importance_scores.items():
        layer_results = {}
        
        for head_idx, importance in head_scores.items():
            # Create fresh model for each test
            model = create_correct_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            # Mask the head
            orig_in_proj, orig_out_proj = mask_attention_head(model, layer_idx, head_idx, device)
            
            if orig_in_proj is not None:
                # Evaluate with masked head
                logger.reset()
                eval_epoch(logger, test_loader, model, split='test')
                masked_stats = logger.write_epoch(0)
                masked_perf = masked_stats.get(metric_key, masked_stats.get('loss'))
                
                # Calculate performance drop
                if 'loss' in metric_key or metric_key == 'mae':
                    # Lower is better
                    performance_drop = (masked_perf - baseline_perf) / abs(baseline_perf)
                else:
                    # Higher is better
                    performance_drop = (baseline_perf - masked_perf) / baseline_perf
                
                layer_results[head_idx] = {
                    'importance': importance,
                    'performance_drop': performance_drop,
                    'masked_performance': masked_perf
                }

                restore_attention_head(model, layer_idx, orig_in_proj, orig_out_proj) # <<<< ADDED
                logging.info(f"Layer {layer_idx}, Head {head_idx}: "
                            f"importance={importance:.9f}, drop={performance_drop:.9f}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        ablation_results[layer_idx] = layer_results
    
    logger.close()
    return ablation_results, baseline_perf, metric_key


def compute_interpretability_metrics(importance_scores, ablation_results):
    """
    Compute IAC and interpretability classification
    """
    # Collect all data points
    all_data = []
    
    for layer_idx in importance_scores.keys():
        if layer_idx in ablation_results:
            for head_idx in importance_scores[layer_idx].keys():
                if head_idx in ablation_results[layer_idx]:
                    all_data.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'kl_divergence': importance_scores[layer_idx][head_idx],
                        'drop': ablation_results[layer_idx][head_idx]['performance_drop']
                    })
    
    if len(all_data) < 2:
        return {
            'hypothesis': 'insufficient_data',
            'iac': 0.0,
            'raw_correlation': 0.0,
            'classification': 'Non-interpretable',
            'fidelity_plus': 0,
            'fidelity_minus': 1,
            'num_heads_tested': len(all_data)
        }
    
    # Calculate correlation
    kl_divergences = [item['kl_divergence'] for item in all_data]
    drops = [item['drop'] for item in all_data]
    correlation, _ = spearmanr(kl_divergences, drops)
    
    logging.info("\n" + "="*60)
    logging.info("PK-EXPLAINER INTERPRETABILITY ANALYSIS")
    logging.info("="*60)
    
    logging.info("\nMETHOD: Measuring correlation between attention pattern uniqueness and performance impact")
    logging.info("- KL Divergence: How unique a head's attention patterns are (higher = more unique)")
    logging.info("- Performance Drop: How much performance decreases when head is removed")
    
    # Display the data sorted by KL divergence
    logging.info("\nHeads sorted by KL divergence (low to high) WITHIN EACH LAYER:")
    num_layers = max(item['layer'] for item in all_data) + 1
    
    for layer in range(num_layers):
        layer_data = [item for item in all_data if item['layer'] == layer]
        if layer_data:
            layer_data_sorted = sorted(layer_data, key=lambda x: x['kl_divergence'])
            logging.info(f"\n  Layer {layer}:")
            for item in layer_data_sorted:
                logging.info(f"    Head {item['head']}: KL-div={item['kl_divergence']:.9f}, perf-drop={item['drop']:.9f}")
    
    # Report the correlation
    logging.info(f"\n" + "-"*40)
    logging.info(f"CORRELATION ANALYSIS:")
    logging.info(f"Spearman correlation between KL divergence and performance drop: {correlation:.9f}")
    
    # Interpret what this means
    if correlation < -0.1:  # Negative correlation
        logging.info("\nINTERPRETATION:")
        logging.info("- Negative correlation indicates: As KL divergence increases, performance drop decreases")
        logging.info("- This means: Heads with REDUNDANT attention patterns (low KL) cause HIGH performance drops")
        logging.info("- These redundant-looking heads are actually FOUNDATIONAL components")
        hypothesis = 'foundation'
        iac = correlation  
        
    elif correlation > 0.1:  # Positive correlation
        logging.info("\nINTERPRETATION:")
        logging.info("- Positive correlation indicates: As KL divergence increases, performance drop increases")
        logging.info("- This means: Heads with UNIQUE attention patterns (high KL) cause HIGH performance drops")
        logging.info("- These specialized heads are critical for specific patterns")
        hypothesis = 'specialization'
        iac = correlation  
        
    else:  # Near zero correlation
        logging.info("\nINTERPRETATION:")
        logging.info("- Near-zero correlation indicates: No clear relationship between attention uniqueness and importance")
        logging.info("- Head importance cannot be predicted from attention patterns alone")
        hypothesis = 'no_clear_pattern'
        iac = correlation

    # Classification based on correlation strength
    abs_corr = abs(correlation)
    if abs_corr > 0.5:
        classification = "Interpretable"
        logging.info(f"\nCLASSIFICATION: {classification} (strong correlation)")
    elif abs_corr > 0.1:
        classification = "Partially Interpretable"
        logging.info(f"\nCLASSIFICATION: {classification} (moderate correlation)")
    else:
        classification = "Non-interpretable"
        logging.info(f"\nCLASSIFICATION: {classification} (weak correlation)")
    
    # CORRECTED FIDELITY CALCULATION - Per Layer
    important_drops_all = []
    unimportant_drops_all = []
    
    # Process each layer separately
    for layer_idx in range(num_layers):
        # Get data for this layer
        layer_data = [item for item in all_data if item['layer'] == layer_idx]
        
        if layer_data:
            # Sort by KL divergence for this layer
            layer_sorted = sorted(layer_data, key=lambda x: x['kl_divergence'])
            
            # Split heads in this layer
            k = len(layer_sorted)  # Number of heads in this layer
            k_half = k // 2 + (k % 2)  # Round up for odd numbers
            
            if hypothesis == 'foundation':
                # Low KL heads are important
                important_heads = layer_sorted[:k_half]
                unimportant_heads = layer_sorted[k_half:]
            elif hypothesis == 'specialization':
                # High KL heads are important
                important_heads = layer_sorted[-k_half:]
                unimportant_heads = layer_sorted[:-k_half]
            else:
                # No clear pattern, use first half as important
                important_heads = layer_sorted[:k_half]
                unimportant_heads = layer_sorted[k_half:]
            
            # Collect drops
            important_drops_all.extend([h['drop'] for h in important_heads])
            unimportant_drops_all.extend([h['drop'] for h in unimportant_heads])
    
    # Calculate fidelity metrics
    fid_plus = np.mean(important_drops_all) if important_drops_all else 0
    fid_minus = np.mean(unimportant_drops_all) if unimportant_drops_all else 1
    
    logging.info(f"\nFIDELITY METRICS (Per-Layer Analysis):")
    logging.info(f"- Total heads analyzed: {len(all_data)}")
    logging.info(f"- Important heads (top k/2 per layer): {len(important_drops_all)}")
    logging.info(f"- Unimportant heads (bottom k/2 per layer): {len(unimportant_drops_all)}")
    logging.info(f"- Fidelity+ (avg drop for important heads): {fid_plus:.9f}")
    logging.info(f"- Fidelity- (avg drop for unimportant heads): {fid_minus:.9f}")
    
    # Summary
    logging.info(f"\n" + "="*60)
    logging.info(f"SUMMARY:")
    if hypothesis == 'foundation':
        logging.info("- Model relies on FOUNDATIONAL heads with redundant attention patterns")
        logging.info("- Removing seemingly redundant heads causes major performance degradation")
    elif hypothesis == 'specialization':
        logging.info("- Model relies on SPECIALIZED heads with unique attention patterns")
        logging.info("- Removing heads with unique patterns causes major performance degradation")
    else:
        logging.info("- No clear relationship between attention patterns and head importance")
    logging.info(f"- Interpretability Assessment: {classification}")
    logging.info("="*60 + "\n")
    
    results = {
        'hypothesis': hypothesis,
        'iac': iac,
        'raw_correlation': correlation,
        'classification': classification,
        'fidelity_plus': fid_plus,
        'fidelity_minus': fid_minus,
        'num_heads_tested': len(all_data),
        'sorted_heads': all_data
    }
    
    return results

def visualize_results(importance_scores, ablation_results, interpretability_metrics, save_dir):
    """
    Create visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get dimensions
    num_layers = len(importance_scores)
    if num_layers == 0:
        return
    num_heads = len(next(iter(importance_scores.values())))
    
    # Create matrices
    importance_matrix = np.zeros((num_layers, num_heads))
    drop_matrix = np.zeros((num_layers, num_heads))
    
    for layer_idx, head_scores in importance_scores.items():
        for head_idx, score in head_scores.items():
            importance_matrix[layer_idx, head_idx] = score
            if layer_idx in ablation_results and head_idx in ablation_results[layer_idx]:
                drop_matrix[layer_idx, head_idx] = ablation_results[layer_idx][head_idx]['performance_drop']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Importance heatmap
    sns.heatmap(importance_matrix, 
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'L{i}' for i in range(num_layers)],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Importance Score'},
                ax=ax1)
    ax1.set_title('Head Importance Scores (KL Divergence)')
    ax1.set_xlabel('Head')
    ax1.set_ylabel('Layer')
    
    # Performance drop heatmap
    sns.heatmap(drop_matrix,
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'L{i}' for i in range(num_layers)],
                cmap='Blues',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Performance Drop'},
                ax=ax2)
    ax2.set_title('Performance Drop When Masked')
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Layer')
    
    fig.suptitle(f"PK-Explainer Analysis - {interpretability_metrics['classification']} "
                 f"(IAC={interpretability_metrics['iac']:.3f}, "
                 f"Hypothesis: {interpretability_metrics['hypothesis']})",
                 fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'pk_explainer_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlation plot
    plt.figure(figsize=(8, 6))
    
    importances = []
    drops = []
    for layer_idx, layer_heads in importance_scores.items():
        for head_idx, imp in layer_heads.items():
            if layer_idx in ablation_results and head_idx in ablation_results[layer_idx]:
                importances.append(imp)
                drops.append(ablation_results[layer_idx][head_idx]['performance_drop'])
    
    plt.scatter(importances, drops, alpha=0.7, s=100)
    plt.xlabel('Head Importance Score')
    plt.ylabel('Performance Drop When Masked')
    plt.title(f'Importance vs Performance Drop\n'
              f'Spearman Correlation: {interpretability_metrics["iac"]:.3f}')
    
    if len(importances) > 1:
        z = np.polyfit(importances, drops, 1)
        p = np.poly1d(z)
        plt.plot(sorted(importances), p(sorted(importances)), "r--", alpha=0.8)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'importance_vs_drop_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved visualizations to {save_dir}")







def run_pk_analysis(model, train_loader, test_loader):
    """
    Main entry point for PK-Explainer analysis
    """
    logging.info("="*60)
    logging.info("Starting PK-Explainer Analysis")
    logging.info("="*60)
    
    # DEBUG: Check model output dimension
    for name, param in model.named_parameters():
        if 'post_mp' in name and 'weight' in name and 'model.1' in name:
            logging.info(f"DEBUG: Model output layer {name} shape: {param.shape}")
    logging.info(f"DEBUG: cfg.share.dim_out = {cfg.share.dim_out}")
    
    # Create subset of training data
    train_dataset = train_loader.dataset
    subset_size = int(len(train_dataset) * cfg.gt.pk_explainer.sample_ratio)
    
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    
    from torch.utils.data import Subset
    subset_dataset = Subset(train_dataset, indices)
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Step 1: Compute importance scores
    logging.info(f"Computing importance scores on {subset_size} training samples...")
    importance_scores = compute_importance_scores(model, subset_loader)
    
    # Log importance scores
    for layer_idx, head_scores in importance_scores.items():
        logging.info(f"\nLayer {layer_idx} Head Importance:")
        sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
        for head_idx, score in sorted_heads:
            logging.info(f"  Head {head_idx}: {score:.9f}")
    

    # Save model for ablation study
    # Save model state dict for ablation study
    model_path = Path(cfg.run_dir) / 'model_for_ablation.pt'
    # Extract optimal weights if it's a NAS model
    optimal_weights = None
    if hasattr(model, 'model') and hasattr(model.model, 'optimal_weights_dict'):
        optimal_weights = model.model.optimal_weights_dict
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        'optimal_weights': optimal_weights,  # Save for recreation
        'config': cfg
    }, model_path)
    logging.info(f"Saved model state to {model_path}")
    
    # Step 2: Ablation study
    logging.info("\nPerforming ablation study...")
    ablation_results, baseline_perf, metric_name = ablation_study(
        model_path, test_loader, importance_scores
    )
    
    # Step 3: Compute interpretability metrics
    interpretability_metrics = compute_interpretability_metrics(
        importance_scores, ablation_results
    )
    
    logging.info(f"\nInterpretability Analysis:")
    logging.info(f"  Winning Hypothesis: {interpretability_metrics['hypothesis']}")
    logging.info(f"  IAC Score: {interpretability_metrics['iac']:.9f}")
    logging.info(f"  Classification: {interpretability_metrics['classification']}")
    logging.info(f"  Fidelity+: {interpretability_metrics['fidelity_plus']:.9f}")
    logging.info(f"  Fidelity-: {interpretability_metrics['fidelity_minus']:.9f}")
    
    # Step 4: Visualization
    if cfg.gt.pk_explainer.visualization:
        vis_dir = Path(cfg.run_dir) / 'pk_explainer_results'
        visualize_results(importance_scores, ablation_results, interpretability_metrics, vis_dir)
    
    # Save results
    results = {
        'importance_scores': {
            str(layer): {str(head): float(score) 
                        for head, score in heads.items()}
            for layer, heads in importance_scores.items()
        },
        'ablation_results': {
            str(layer): {
                str(head): {
                    'importance': float(data['importance']),
                    'performance_drop': float(data['performance_drop']),
                    'masked_performance': float(data['masked_performance'])
                }
                for head, data in heads.items()
            }
            for layer, heads in ablation_results.items()
        },
        'interpretability_metrics': interpretability_metrics,
        'baseline_performance': {
            metric_name: float(baseline_perf)
        }
    }
    
    results_file = Path(cfg.run_dir) / 'pk_explainer_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nPK-Explainer results saved to {results_file}")
    
    # Clean up
    model_path.unlink()  # Delete temporary model file
    
    return results
