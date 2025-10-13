# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 00:47:44 2025

@author: ADMIN
"""

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
    Compute POE-based importance scores for IAC calculation only
    """
    model.eval()
    device = torch.device(cfg.accelerator)
    
    # Determine model structure
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
    
    num_layers = len(layers)
    layer_attentions = {i: [] for i in range(num_layers)}
    
    # Collect attention weights
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            _ = model(batch)
            
            for i, layer in enumerate(layers):
                if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
                    if isinstance(layer.attn_weights, tuple):
                        # Sparse attention
                        sparse_weights, edge_index = layer.attn_weights
                        layer_attentions[i].append(('sparse', sparse_weights.cpu(), edge_index.cpu(), batch.batch.cpu()))
                    else:
                        # Dense attention
                        layer_attentions[i].append(layer.attn_weights.cpu())
    
    # Compute POE importance for each layer
    importance_scores = {}
    
    for layer_idx, attentions in layer_attentions.items():
        if not attentions:
            continue
        
        if isinstance(attentions[0], tuple) and attentions[0][0] == 'sparse':
            # Sparse attention processing
            all_kl_divs = []
            
            for attn_idx, (_, sparse_weights, edge_index, batch_idx) in enumerate(attentions):
                unique_graphs = torch.unique(batch_idx)
                num_nodes_total = batch_idx.size(0)
                num_heads = sparse_weights.size(1)
                
                S_h = torch.zeros(num_heads, num_nodes_total)
                for head in range(num_heads):
                    S_h[head] = scatter_add(
                        sparse_weights[:, head],
                        edge_index[1],
                        dim=0,
                        dim_size=num_nodes_total
                    )
                S_h = S_h + 1e-10
                
                for graph_idx, graph_id in enumerate(unique_graphs):
                    node_mask = (batch_idx == graph_id)
                    graph_S_h = S_h[:, node_mask]
                    
                    # Normalize to probability
                    P_h = torch.zeros_like(graph_S_h)
                    for h in range(num_heads):
                        P_h[h] = graph_S_h[h] / (graph_S_h[h].sum() + 1e-10)
                    
                    # Product of Experts
                    log_P_h = torch.log(P_h + 1e-20)
                    log_P_all = log_P_h.sum(dim=0)
                    P_tilde = torch.exp(log_P_all - log_P_all.max())
                    P_all = P_tilde / (P_tilde.sum() + 1e-10)
                    
                    # KL divergence for each head
                    head_kl_divs = []
                    for head_idx in range(num_heads):
                        log_P_minus = torch.zeros_like(log_P_h[0])
                        for h in range(num_heads):
                            if h != head_idx:
                                log_P_minus = log_P_minus + log_P_h[h]
                        
                        P_minus_tilde = torch.exp(log_P_minus - log_P_minus.max())
                        P_minus = P_minus_tilde / (P_minus_tilde.sum() + 1e-10)
                        
                        kl_div = (P_all * torch.log(P_all / (P_minus + 1e-10))).sum()
                        head_kl_divs.append(kl_div.item() if torch.isfinite(kl_div) else 0.0)
                    
                    all_kl_divs.append(head_kl_divs)
                    
        else:
            # Dense attention processing
            all_kl_divs = []
            
            for attn_batch in attentions:
                batch_size, num_heads, num_nodes, _ = attn_batch.shape
                
                for b in range(batch_size):
                    attn_graph = attn_batch[b]
                    
                    # Pooled attention
                    S_h = attn_graph.sum(dim=1)
                    S_h = S_h + 1e-10
                    
                    # Normalize to probability
                    P_h = torch.zeros_like(S_h)
                    for h in range(num_heads):
                        P_h[h] = S_h[h] / (S_h[h].sum() + 1e-10)
                    
                    # Product of Experts
                    log_P_h = torch.log(P_h + 1e-20)
                    log_P_all = log_P_h.sum(dim=0)
                    P_tilde = torch.exp(log_P_all - log_P_all.max())
                    P_all = P_tilde / (P_tilde.sum() + 1e-10)
                    
                    # KL divergence for each head
                    head_kl_divs = []
                    for head_idx in range(num_heads):
                        log_P_minus = torch.zeros_like(log_P_h[0])
                        for h in range(num_heads):
                            if h != head_idx:
                                log_P_minus = log_P_minus + log_P_h[h]
                        
                        P_minus_tilde = torch.exp(log_P_minus - log_P_minus.max())
                        P_minus = P_minus_tilde / (P_minus_tilde.sum() + 1e-10)
                        
                        kl_div = (P_all * torch.log(P_all / (P_minus + 1e-10))).sum()
                        head_kl_divs.append(kl_div.item() if torch.isfinite(kl_div) else 0.0)
                    
                    all_kl_divs.append(head_kl_divs)
        
        # Average KL divergence across all graphs
        if all_kl_divs:
            avg_kl_per_head = np.mean(all_kl_divs, axis=0)
            layer_importance = {h: avg_kl_per_head[h] for h in range(num_heads)}
            importance_scores[layer_idx] = layer_importance
    
    return importance_scores


def mask_attention_head(model, layer_idx: int, head_idx: int, device):
    """
    Mask a specific attention head
    """
    # Get layers
    if hasattr(model, 'model'):
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
        # Check if sparse attention
        if hasattr(layer, 'global_model_type') and layer.global_model_type == 'SparseTransformer':
            sparse_attn = layer.self_attn
            embed_dim = sparse_attn.dim_h
            head_dim = sparse_attn.head_dim
            
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
            orig_weights = {
                'W_k': sparse_attn.W_k.weight.data.clone(),
                'W_v': sparse_attn.W_v.weight.data.clone(),
                'W_o': sparse_attn.W_o.weight.data.clone(),
            }
            
            # Zero out weights
            sparse_attn.W_k.weight.data[start_idx:end_idx, :] = 0
            sparse_attn.W_v.weight.data[start_idx:end_idx, :] = 0
            sparse_attn.W_o.weight.data[:, start_idx:end_idx] = 0
            
            return orig_weights, None
            
        elif hasattr(layer.self_attn, 'in_proj_weight'):
            # Regular MultiheadAttention
            embed_dim = layer.dim_h
            head_dim = embed_dim // layer.num_heads
            
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
            orig_in_proj = layer.self_attn.in_proj_weight.data.clone()
            orig_out_proj = layer.self_attn.out_proj.weight.data.clone()
            
            # Zero out weights
            layer.self_attn.in_proj_weight.data[start_idx:end_idx, :] = 0  # Q
            layer.self_attn.in_proj_weight.data[embed_dim + start_idx:embed_dim + end_idx, :] = 0  # K
            layer.self_attn.in_proj_weight.data[2*embed_dim + start_idx:2*embed_dim + end_idx, :] = 0  # V
            layer.self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0
            
            return orig_in_proj, orig_out_proj
    
    return None, None


def restore_attention_head(model, layer_idx: int, orig_in_proj, orig_out_proj):
    """
    Restore original weights
    """
    if hasattr(model, 'model'):
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
        if isinstance(orig_in_proj, dict):
            # Restore sparse attention
            sparse_attn = layer.self_attn
            sparse_attn.W_k.weight.data = orig_in_proj['W_k']
            sparse_attn.W_v.weight.data = orig_in_proj['W_v']
            sparse_attn.W_o.weight.data = orig_in_proj['W_o']
        else:
            # Restore regular attention
            layer.self_attn.in_proj_weight.data = orig_in_proj
            layer.self_attn.out_proj.weight.data = orig_out_proj


def ablation_study(model_path, test_loader):
    """
    Perform ablation study - measure actual performance drops
    """
    device = torch.device(cfg.accelerator)
    
    # Import eval function
    if cfg.train.mode == 'custom':
        from dartsgt.train.custom_train import eval_epoch
    elif cfg.train.mode == 'NoMixNas_uncertainty_train':
        from dartsgt.train.NoMixNas_uncertainty_train import eval_epoch_with_uncertainty as eval_epoch
    elif cfg.train.mode == 'nas_uncertainty_train':
        from dartsgt.train.NAS_uncertainty_train import eval_epoch_with_uncertainty as eval_epoch
    
    from dartsgt.logger import CustomLogger
    from torch_geometric.graphgym.logger import infer_task
    
    logger = CustomLogger(name='pk_ablation', task_type=infer_task())
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    optimal_weights = checkpoint.get('optimal_weights')
    
    # Fix dimension mismatch
    actual_dim_out = None
    for key in checkpoint['model_state_dict'].keys():
        if 'post_mp' in key and 'model.1' in key and 'weight' in key:
            actual_dim_out = checkpoint['model_state_dict'][key].shape[0]
            break
    
    original_dim_out = cfg.share.dim_out
    if actual_dim_out and actual_dim_out != cfg.share.dim_out:
        logging.info(f"Detected dim_out mismatch: config={cfg.share.dim_out}, saved={actual_dim_out}")
        cfg.share.dim_out = actual_dim_out
    
    # Create model
    def create_correct_model():
        from torch_geometric.graphgym.model_builder import create_model
        model = create_model(cfg)
        
        if cfg.model.type == 'NASModelEdge' and optimal_weights is not None:
            from dartsgt.network.NAS_model_edge import NASModelEdge
            temp_model = NASModelEdge(cfg.share.dim_in, cfg.share.dim_out)
            discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            model.model = discrete_model
        if cfg.model.type == 'NASModel' and optimal_weights is not None:
            from dartsgt.network.NAS_model import NASModel
            temp_model = NASModel(cfg.share.dim_in, cfg.share.dim_out)
            discrete_model = temp_model.get_discrete_model(optimal_weights, cfg.gt.weight_type)
            model.model = discrete_model
        
        return model
    
    # Get baseline performance
    logging.info("Getting baseline performance...")
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
    logging.info(f"Baseline {metric_key}: {baseline_perf:.4f}")
    
    # Test each head
    ablation_results = {}
    
    # Get layer structure
    if hasattr(model, 'model'):
        inner_model = model.model
        if hasattr(inner_model, 'discrete_sequential') and inner_model.discrete_sequential is not None:
            layers = list(inner_model.discrete_sequential)
        elif hasattr(inner_model, 'layers'):
            layers = inner_model.layers
    else:
        if hasattr(model, 'discrete_sequential') and model.discrete_sequential is not None:
            layers = list(model.discrete_sequential)
        elif hasattr(model, 'layers'):
            layers = model.layers
    
    # Test each head
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        if hasattr(layer, 'num_heads'):
            num_heads = layer.num_heads
            layer_results = {}
            
            for head_idx in range(num_heads):
                # Create fresh model
                model = create_correct_model()
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                
                # Mask head
                orig_in_proj, orig_out_proj = mask_attention_head(model, layer_idx, head_idx, device)
                
                if orig_in_proj is not None:
                    # Evaluate
                    logger.reset()
                    eval_epoch(logger, test_loader, model, split='test')
                    masked_stats = logger.write_epoch(0)
                    masked_perf = masked_stats.get(metric_key, masked_stats.get('loss'))
                    
                    # Calculate drop
                    if 'loss' in metric_key or metric_key == 'mae':
                        performance_drop = (masked_perf - baseline_perf) / abs(baseline_perf)
                    else:
                        performance_drop = (baseline_perf - masked_perf) / baseline_perf
                    
                    layer_results[head_idx] = {
                        'performance_drop': performance_drop,
                        'masked_performance': masked_perf
                    }
                    
                    restore_attention_head(model, layer_idx, orig_in_proj, orig_out_proj)
                    logging.info(f"Layer {layer_idx}, Head {head_idx}: drop={performance_drop:.4f}")
                
                del model
                torch.cuda.empty_cache()
            
            ablation_results[layer_idx] = layer_results
    
    # Restore config
    if actual_dim_out and actual_dim_out != original_dim_out:
        cfg.share.dim_out = original_dim_out
    
    logger.close()
    return ablation_results, baseline_perf, metric_key


def compute_interpretability_metrics(ablation_results, poe_scores):
    """
    Compute fidelity and IAC metrics
    """
    # Collect all heads with their drops
    all_heads = []
    
    for layer_idx in ablation_results.keys():
        for head_idx in ablation_results[layer_idx].keys():
            all_heads.append({
                'layer': layer_idx,
                'head': head_idx,
                'drop': ablation_results[layer_idx][head_idx]['performance_drop']
            })
    
    if len(all_heads) < 2:
        return {
            'iac': 0.0,
            'correlation': 0.0,
            'fidelity_plus': 0,
            'fidelity_minus': 1,
            'num_heads_tested': len(all_heads)
        }
    
    # Sort by actual performance drop
    sorted_heads = sorted(all_heads, key=lambda x: x['drop'], reverse=True)
    
    # Calculate fidelity
    k = len(sorted_heads) // 2
    important_heads = sorted_heads[:k]
    unimportant_heads = sorted_heads[k:]
    
    fid_plus = np.mean([h['drop'] for h in important_heads]) if important_heads else 0
    fid_minus = np.mean([h['drop'] for h in unimportant_heads]) if unimportant_heads else 1
    
    logging.info(f"\nFIDELITY METRICS:")
    logging.info(f"Fidelity+ (top {k} heads): {fid_plus:.4f}")
    logging.info(f"Fidelity- (bottom {k} heads): {fid_minus:.4f}")
    
    # Calculate IAC if POE scores available
    iac = 0.0
    correlation = 0.0
    
    if poe_scores:
        poe_values = []
        drop_values = []
        
        for head in all_heads:
            if head['layer'] in poe_scores and head['head'] in poe_scores[head['layer']]:
                poe_values.append(poe_scores[head['layer']][head['head']])
                drop_values.append(head['drop'])
        
        if len(poe_values) >= 2:
            correlation, _ = spearmanr(poe_values, drop_values)
            iac = correlation
            
            logging.info(f"\nCORRELATION ANALYSIS:")
            logging.info(f"IAC (POE vs Drop correlation): {iac:.4f}")
            
            if correlation < -0.1:
                logging.info("Pattern: Foundation - redundant attention heads are important")
            elif correlation > 0.1:
                logging.info("Pattern: Specialization - unique attention heads are important")
            else:
                logging.info("Pattern: No clear relationship between attention and importance")
    
    return {
        'iac': iac,
        'correlation': correlation,
        'fidelity_plus': fid_plus,
        'fidelity_minus': fid_minus,
        'num_heads_tested': len(all_heads),
        'sorted_heads': sorted_heads
    }


def visualize_results(ablation_results, poe_scores, interpretability_metrics, save_dir):
    """
    Create visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    num_layers = len(ablation_results)
    if num_layers == 0:
        return
    num_heads = len(next(iter(ablation_results.values())))
    
    # Create matrices
    drop_matrix = np.zeros((num_layers, num_heads))
    poe_matrix = np.zeros((num_layers, num_heads))
    
    for layer_idx, layer_results in ablation_results.items():
        for head_idx, results in layer_results.items():
            drop_matrix[layer_idx, head_idx] = results['performance_drop']
            if layer_idx in poe_scores and head_idx in poe_scores[layer_idx]:
                poe_matrix[layer_idx, head_idx] = poe_scores[layer_idx][head_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance drop heatmap
    sns.heatmap(drop_matrix,
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'L{i}' for i in range(num_layers)],
                cmap='Blues',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Performance Drop'},
                ax=ax1)
    ax1.set_title('Actual Performance Drop When Masked')
    ax1.set_xlabel('Head')
    ax1.set_ylabel('Layer')
    
    # POE scores heatmap
    sns.heatmap(poe_matrix, 
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'L{i}' for i in range(num_layers)],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'POE Score'},
                ax=ax2)
    ax2.set_title('POE Attention Scores')
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Layer')
    
    fig.suptitle(f"PK-Explainer Analysis - IAC={interpretability_metrics['iac']:.3f}")
    plt.tight_layout()
    plt.savefig(save_dir / 'pk_explainer_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlation plot
    if interpretability_metrics['correlation'] != 0:
        plt.figure(figsize=(8, 6))
        
        poe_vals = []
        drop_vals = []
        for layer_idx in ablation_results:
            for head_idx in ablation_results[layer_idx]:
                if layer_idx in poe_scores and head_idx in poe_scores[layer_idx]:
                    poe_vals.append(poe_scores[layer_idx][head_idx])
                    drop_vals.append(ablation_results[layer_idx][head_idx]['performance_drop'])
        
        plt.scatter(poe_vals, drop_vals, alpha=0.7, s=100)
        plt.xlabel('POE Score')
        plt.ylabel('Performance Drop')
        plt.title(f'POE Score vs Performance Drop\nCorrelation: {interpretability_metrics["iac"]:.3f}')
        
        if len(poe_vals) > 1:
            z = np.polyfit(poe_vals, drop_vals, 1)
            p = np.poly1d(z)
            plt.plot(sorted(poe_vals), p(sorted(poe_vals)), "r--", alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'correlation_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Saved visualizations to {save_dir}")


def run_pk_analysis(model, train_loader, test_loader):
    """
    Main entry point - ablation first approach
    """
    logging.info("="*60)
    logging.info("Starting PK-Explainer Analysis")
    logging.info("="*60)
    
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
    
    # Step 1: Save model for ablation
    model_path = Path(cfg.run_dir) / 'model_for_ablation.pt'
    optimal_weights = None
    if hasattr(model, 'model') and hasattr(model.model, 'optimal_weights_dict'):
        optimal_weights = model.model.optimal_weights_dict
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        'optimal_weights': optimal_weights,
        'config': cfg
    }, model_path)
    logging.info(f"Saved model state to {model_path}")
    
    # Step 2: Ablation study FIRST
    logging.info("\nPerforming ablation study...")
    ablation_results, baseline_perf, metric_name = ablation_study(model_path, test_loader)
    
    # Step 3: Compute POE scores for correlation
    logging.info(f"\nComputing POE scores on {subset_size} training samples...")
    poe_scores = compute_importance_scores(model, subset_loader)
    
    # Log POE scores for each layer
    logging.info("\nPOE-based Attention Importance Scores (KL divergence):")
    for layer_idx in sorted(poe_scores.keys()):
        logging.info(f"\nLayer {layer_idx}:")
        layer_scores = poe_scores[layer_idx]
        # Sort heads by KL divergence (high to low)
        sorted_heads = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        for head_idx, kl_div in sorted_heads:
            logging.info(f"  Head {head_idx}: KL-div = {kl_div:.9f}")
    
    # Step 4: Compute metrics
    interpretability_metrics = compute_interpretability_metrics(ablation_results, poe_scores)
    
    logging.info(f"\nInterpretability Analysis:")
    logging.info(f"  IAC Score: {interpretability_metrics['iac']:.4f}")
    logging.info(f"  Fidelity+: {interpretability_metrics['fidelity_plus']:.4f}")
    logging.info(f"  Fidelity-: {interpretability_metrics['fidelity_minus']:.4f}")
    
    # Step 5: Visualization
    if cfg.gt.pk_explainer.visualization:
        vis_dir = Path(cfg.run_dir) / 'pk_explainer_results'
        visualize_results(ablation_results, poe_scores, interpretability_metrics, vis_dir)
    
    # Save results
    results = {
        'ablation_results': {
            str(layer): {
                str(head): {
                    'performance_drop': float(data['performance_drop']),
                    'masked_performance': float(data['masked_performance'])
                }
                for head, data in heads.items()
            }
            for layer, heads in ablation_results.items()
        },
        'poe_scores': {
            str(layer): {str(head): float(score) 
                        for head, score in heads.items()}
            for layer, heads in poe_scores.items()
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
    model_path.unlink()
    
    return results
