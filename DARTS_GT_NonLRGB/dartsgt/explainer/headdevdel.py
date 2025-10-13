# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:13:07 2025

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Head Deletion Experiments
Validates pk_explainer rankings by actually removing heads and measuring impact
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from torch_geometric.graphgym.config import cfg
from tqdm import tqdm
import random

def permanently_delete_heads(model, heads_to_delete):
    """
    Permanently delete specified heads by zeroing their weights
    heads_to_delete: list of (layer_idx, head_idx) tuples
    """
    # Get layers
    if hasattr(model, 'model'):
        if hasattr(model.model, 'discrete_sequential'):
            layers = list(model.model.discrete_sequential)
        else:
            layers = model.model.layers
    else:
        layers = model.layers
    
    for layer_idx, head_idx in heads_to_delete:
        layer = layers[layer_idx]
        
        if hasattr(layer, 'self_attn') and layer.self_attn is not None:
            # Check if sparse attention
            if hasattr(layer, 'global_model_type') and layer.global_model_type == 'SparseTransformer':
                # Handle sparse attention
                sparse_attn = layer.self_attn
                head_dim = sparse_attn.head_dim
                
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim
                
                # Zero out sparse attention weights permanently
                with torch.no_grad():
                    sparse_attn.W_k.weight.data[start_idx:end_idx, :] = 0
                    sparse_attn.W_v.weight.data[start_idx:end_idx, :] = 0
                    sparse_attn.W_o.weight.data[:, start_idx:end_idx] = 0
                    if sparse_attn.W_k.bias is not None:
                        sparse_attn.W_k.bias.data[start_idx:end_idx] = 0
                        sparse_attn.W_v.bias.data[start_idx:end_idx] = 0
                
            elif hasattr(layer.self_attn, 'in_proj_weight'):
                # Handle regular MultiheadAttention
                embed_dim = layer.dim_h
                head_dim = embed_dim // layer.num_heads
                
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim
                
                # Zero out the weights permanently
                with torch.no_grad():
                    layer.self_attn.in_proj_weight.data[start_idx:end_idx, :] = 0  # Q
                    layer.self_attn.in_proj_weight.data[embed_dim + start_idx:embed_dim + end_idx, :] = 0  # K
                    layer.self_attn.in_proj_weight.data[2*embed_dim + start_idx:2*embed_dim + end_idx, :] = 0  # V
                    layer.self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0  # Output
                    
                    if layer.self_attn.in_proj_bias is not None:
                        layer.self_attn.in_proj_bias.data[start_idx:end_idx] = 0  # Q bias
                        layer.self_attn.in_proj_bias.data[embed_dim + start_idx:embed_dim + end_idx] = 0  # K bias
                        layer.self_attn.in_proj_bias.data[2*embed_dim + start_idx:2*embed_dim + end_idx] = 0  # V bias


def evaluate_model_performance(model, test_loader):
    """Evaluate model and return performance metrics"""
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
            
            all_predictions.append(pred.cpu())
            all_labels.append(batch.y.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics based on task type
    task_type = cfg.dataset.task_type
    is_binary = getattr(cfg.dataset, 'is_binary', None)
    
    if task_type == 'regression':
        # Handle multi-dimensional regression
        if len(all_predictions.shape) > 1 and all_predictions.shape[1] > 1:
            # Multi-target: use mean absolute error across all targets
            errors = torch.abs(all_predictions - all_labels)
            # Handle NaN values
            mask = ~torch.isnan(all_labels)
            if mask.sum() > 0:
                mae = errors[mask].mean().item()
            else:
                mae = 0.0
            return {'mae': mae}
        else:
            # Single target
            mae = torch.abs(all_predictions - all_labels).mean().item()
            mse = ((all_predictions - all_labels) ** 2).mean().item()
            return {'mae': mae, 'mse': mse}
    
    elif task_type == 'classification_multilabel' or is_binary == 'multi_label':
        # Multi-label: use average precision or hamming distance
        pred_binary = (torch.sigmoid(all_predictions) > 0.5).float()
        
        # Handle NaN values
        mask = ~torch.isnan(all_labels)
        if mask.sum() > 0:
            # Hamming distance (proportion of incorrect labels)
            hamming = (pred_binary[mask] != all_labels[mask]).float().mean().item()
            accuracy = 1.0 - hamming
        else:
            accuracy = 0.0
        return {'accuracy': accuracy, 'hamming_distance': hamming}
    
    elif task_type in ['classification_binary', 'classification_multi', 'classification']:
        if is_binary == 'binary' or all_predictions.shape[-1] == 1:
            # Binary classification
            pred_probs = torch.sigmoid(all_predictions.squeeze())
            pred_class = (pred_probs > 0.5).float()
            accuracy = (pred_class == all_labels.squeeze()).float().mean().item()
            
            # AUROC if we have both classes
            if len(torch.unique(all_labels)) > 1:
                from sklearn.metrics import roc_auc_score
                try:
                    auroc = roc_auc_score(all_labels.numpy(), pred_probs.numpy())
                except:
                    auroc = 0.0
            else:
                auroc = 0.0
            
            return {'accuracy': accuracy, 'auroc': auroc}
        else:
            # Multi-class
            pred_class = all_predictions.argmax(dim=-1)
            accuracy = (pred_class == all_labels.squeeze()).float().mean().item()
            return {'accuracy': accuracy}
    
    # Fallback
    return {'error': 0.0}


def get_all_head_rankings(checkpoint_path, test_loader):
    """
    Get ranking of all heads by their deviation (importance)
    Reuses logic from pk_explainer
    """
    from pk_explainer import compute_all_head_deviations
    
    # Get deviations for all heads
    all_deviations, baseline_preds, true_labels = compute_all_head_deviations(checkpoint_path, test_loader)
    
    # Aggregate deviations across all graphs
    head_importance = {}
    
    for graph_idx in all_deviations:
        for layer_idx in all_deviations[graph_idx]:
            for head_idx in all_deviations[graph_idx][layer_idx]:
                key = (layer_idx, head_idx)
                if key not in head_importance:
                    head_importance[key] = []
                head_importance[key].append(all_deviations[graph_idx][layer_idx][head_idx])
    
    # Average importance per head
    avg_importance = {}
    for key in head_importance:
        avg_importance[key] = np.mean(head_importance[key])
    
    # Sort by importance (descending)
    sorted_heads = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_heads, avg_importance


def run_head_deletion_experiments(model, train_loader, test_loader):
    """
    Main entry point for head deletion experiments
    """
    logging.info("="*60)
    logging.info("Starting Head Deletion Experiments")
    logging.info("="*60)
    
    # Get config parameters
    k_heads = getattr(cfg.gt.head_deletion, 'k_heads', 5)
    if isinstance(k_heads, int):
        k_values = [k_heads]
    else:
        k_values = k_heads
    
    modes = getattr(cfg.gt.head_deletion, 'modes', ['top_k', 'random'])
    num_random_trials = getattr(cfg.gt.head_deletion, 'num_random_trials', 5)
    
    # Create results directory
    results_dir = Path(cfg.run_dir) / 'head_deletion_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Save checkpoint
    checkpoint_path = Path(cfg.run_dir) / 'deletion_checkpoint.pt'
    optimal_weights = None
    if hasattr(model, 'model') and hasattr(model.model, 'optimal_weights_dict'):
        optimal_weights = model.model.optimal_weights_dict
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimal_weights': optimal_weights
    }, checkpoint_path)
    
    # Step 2: Get baseline performance
    logging.info("Evaluating baseline model...")
    baseline_metrics = evaluate_model_performance(model, test_loader)
    logging.info(f"Baseline performance: {baseline_metrics}")
    
    # Step 3: Get head rankings
    logging.info("Computing head importance rankings...")
    sorted_heads, avg_importance = get_all_head_rankings(checkpoint_path, test_loader)
    
    # Get total number of heads
    total_heads = len(sorted_heads)
    logging.info(f"Total heads in model: {total_heads}")
    
    # Save head rankings
    rankings_file = results_dir / 'head_rankings.json'
    rankings_data = {
        'rankings': [(f'L{h[0]}_H{h[1]}', float(imp)) for (h, imp) in sorted_heads],
        'total_heads': total_heads
    }
    with open(rankings_file, 'w') as f:
        json.dump(rankings_data, f, indent=2)
    
    # Step 4: Run deletion experiments
    all_results = {
        'baseline': baseline_metrics,
        'experiments': {}
    }
    
    for k in k_values:
        logging.info(f"\n--- Testing with k={k} heads ---")
        all_results['experiments'][f'k_{k}'] = {}
        
        for mode in modes:
            logging.info(f"\nMode: {mode}")
            
            if mode == 'top_k':
                # Delete top k most important heads
                heads_to_delete = [h[0] for h in sorted_heads[:k]]
                
                # Create fresh model and delete heads
                from pk_explainer import create_model_from_checkpoint
                model_copy, _ = create_model_from_checkpoint(checkpoint_path)
                permanently_delete_heads(model_copy, heads_to_delete)
                
                # Evaluate
                metrics = evaluate_model_performance(model_copy, test_loader)
                
                result = {
                    'deleted_heads': [f'L{h[0]}_H{h[1]}' for h in heads_to_delete],
                    'metrics': metrics,
                    'performance_change': calculate_change(baseline_metrics, metrics)
                }
                all_results['experiments'][f'k_{k}'][mode] = result
                logging.info(f"Top-{k} deletion: {metrics}")
                
                del model_copy
                torch.cuda.empty_cache()
                
            elif mode == 'bottom_k':
                # Delete bottom k least important (or harmful) heads
                heads_to_delete = [h[0] for h in sorted_heads[-k:]]
                
                from pk_explainer import create_model_from_checkpoint
                model_copy, _ = create_model_from_checkpoint(checkpoint_path)
                permanently_delete_heads(model_copy, heads_to_delete)
                
                metrics = evaluate_model_performance(model_copy, test_loader)
                
                result = {
                    'deleted_heads': [f'L{h[0]}_H{h[1]}' for h in heads_to_delete],
                    'metrics': metrics,
                    'performance_change': calculate_change(baseline_metrics, metrics)
                }
                all_results['experiments'][f'k_{k}'][mode] = result
                logging.info(f"Bottom-{k} deletion: {metrics}")
                
                del model_copy
                torch.cuda.empty_cache()
                
            elif mode == 'random':
                # Multiple random trials
                random_results = []
                
                for trial in range(num_random_trials):
                    # Randomly select k heads
                    all_head_tuples = [h[0] for h in sorted_heads]
                    heads_to_delete = random.sample(all_head_tuples, k)
                    
                    from pk_explainer import create_model_from_checkpoint
                    model_copy, _ = create_model_from_checkpoint(checkpoint_path)
                    permanently_delete_heads(model_copy, heads_to_delete)
                    
                    metrics = evaluate_model_performance(model_copy, test_loader)
                    
                    random_results.append({
                        'trial': trial,
                        'deleted_heads': [f'L{h[0]}_H{h[1]}' for h in heads_to_delete],
                        'metrics': metrics
                    })
                    
                    del model_copy
                    torch.cuda.empty_cache()
                
                # Aggregate random results
                avg_metrics = {}
                for key in random_results[0]['metrics']:
                    values = [r['metrics'][key] for r in random_results]
                    avg_metrics[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                
                result = {
                    'num_trials': num_random_trials,
                    'avg_metrics': avg_metrics,
                    'performance_change': calculate_change(baseline_metrics, 
                                                          {k: v['mean'] for k, v in avg_metrics.items()}),
                    'individual_trials': random_results
                }
                all_results['experiments'][f'k_{k}'][mode] = result
                logging.info(f"Random-{k} deletion (avg of {num_random_trials} trials): {avg_metrics}")
                
            elif mode == 'progressive':
                # Progressive deletion: 1, 2, ..., k
                progressive_results = []
                
                for i in range(1, k+1):
                    heads_to_delete = [h[0] for h in sorted_heads[:i]]
                    
                    from pk_explainer import create_model_from_checkpoint
                    model_copy, _ = create_model_from_checkpoint(checkpoint_path)
                    permanently_delete_heads(model_copy, heads_to_delete)
                    
                    metrics = evaluate_model_performance(model_copy, test_loader)
                    
                    progressive_results.append({
                        'num_deleted': i,
                        'deleted_heads': [f'L{h[0]}_H{h[1]}' for h in heads_to_delete],
                        'metrics': metrics,
                        'performance_change': calculate_change(baseline_metrics, metrics)
                    })
                    
                    del model_copy
                    torch.cuda.empty_cache()
                
                all_results['experiments'][f'k_{k}'][mode] = progressive_results
                logging.info(f"Progressive deletion complete (1 to {k} heads)")
    
    # Step 5: Create visualizations
    create_deletion_visualizations(all_results, results_dir)
    
    # Step 6: Save results
    results_file = results_dir / 'deletion_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"\nResults saved to {results_file}")
    logging.info("="*60)
    
    # Clean up
    checkpoint_path.unlink()
    
    return all_results


def calculate_change(baseline, current):
    """Calculate relative performance change"""
    changes = {}
    for key in baseline:
        if key in current:
            if isinstance(current[key], dict) and 'mean' in current[key]:
                curr_val = current[key]['mean']
            else:
                curr_val = current[key]
            
            base_val = baseline[key]
            
            # For accuracy/AUROC: decrease is bad
            if 'accuracy' in key.lower() or 'auroc' in key.lower():
                change = ((curr_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            # For error metrics: increase is bad
            else:
                change = ((curr_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            
            changes[key] = round(change, 2)
    
    return changes


def create_deletion_visualizations(results, save_dir):
    """Create comparison plots"""
    experiments = results['experiments']
    
    for k_key in experiments:
        k_value = int(k_key.split('_')[1])
        
        # Extract metrics for plotting
        modes_data = experiments[k_key]
        
        if 'progressive' in modes_data:
            # Progressive deletion plot
            progressive = modes_data['progressive']
            num_deleted = [r['num_deleted'] for r in progressive]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each metric
            metric_keys = progressive[0]['metrics'].keys()
            for metric in metric_keys:
                values = [r['metrics'][metric] for r in progressive]
                ax.plot(num_deleted, values, marker='o', label=metric)
            
            # Add baseline
            for metric in results['baseline']:
                ax.axhline(y=results['baseline'][metric], linestyle='--', 
                          alpha=0.5, label=f'Baseline {metric}')
            
            ax.set_xlabel('Number of Heads Deleted')
            ax.set_ylabel('Performance')
            ax.set_title(f'Progressive Head Deletion (Top-k Important Heads)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'progressive_deletion_k{k_value}.png', dpi=150)
            plt.close()
        
        # Comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mode_names = []
        performances = []
        errors = []
        
        for mode in ['top_k', 'bottom_k', 'random']:
            if mode in modes_data:
                mode_names.append(mode.replace('_', ' ').title())
                
                if mode == 'random':
                    # Use first metric for comparison
                    first_metric = list(modes_data[mode]['avg_metrics'].keys())[0]
                    performances.append(modes_data[mode]['avg_metrics'][first_metric]['mean'])
                    errors.append(modes_data[mode]['avg_metrics'][first_metric]['std'])
                else:
                    first_metric = list(modes_data[mode]['metrics'].keys())[0]
                    performances.append(modes_data[mode]['metrics'][first_metric])
                    errors.append(0)
        
        # Add baseline
        mode_names.insert(0, 'Baseline')
        performances.insert(0, results['baseline'][first_metric])
        errors.insert(0, 0)
        
        x = np.arange(len(mode_names))
        bars = ax.bar(x, performances, yerr=errors, capsize=5, alpha=0.7)
        
        # Color code bars
        bars[0].set_color('green')  # Baseline
        if len(bars) > 1:
            bars[1].set_color('red')  # Top-k (should be worst)
        if len(bars) > 2:
            bars[2].set_color('blue')  # Bottom-k (might be better)
        if len(bars) > 3:
            bars[3].set_color('orange')  # Random
        
        ax.set_xlabel('Deletion Mode')
        ax.set_ylabel(first_metric.replace('_', ' ').title())
        ax.set_title(f'Performance After Deleting {k_value} Heads')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_names)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'deletion_comparison_k{k_value}.png', dpi=150)
        plt.close()
    
    logging.info(f"Visualizations saved to {save_dir}")
    
    
'''
gt:
  head_deletion:
    enable: True
    k_heads: [3, 5, 10]  # or just 5
    modes: ['top_k', 'bottom_k', 'random', 'progressive']
    num_random_trials: 5
'''
