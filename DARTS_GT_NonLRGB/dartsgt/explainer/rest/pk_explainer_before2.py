# -*- coding: utf-8 -*-
"""
PK-Explainer: Quantitative Head Importance Analysis via Ablation
Enhanced version with proper GNN type tracking and visualization
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader


def get_gnn_type_mapping(model):
    """
    Extract GNN type for each layer from the model's optimal weights
    """
    gnn_mapping = {}
    
    # Check if model has optimal weights stored
    if hasattr(model, 'model'):
        inner_model = model.model
        if hasattr(inner_model, 'optimal_weights_dict'):
            optimal_weights = inner_model.optimal_weights_dict
            logging.info(f"Found optimal weights in model: {optimal_weights}")
            
            # Map each layer to its GNN type
            for layer_key, gnn_idx in optimal_weights.items():
                layer_num = int(layer_key.split('_')[1])
                if gnn_idx < len(cfg.gt.head_gnn_types):
                    gnn_type = cfg.gt.head_gnn_types[gnn_idx]
                    gnn_mapping[layer_num] = gnn_type
                    logging.info(f"Layer {layer_num}: {gnn_type} (index {gnn_idx})")
                else:
                    logging.warning(f"Invalid GNN index {gnn_idx} for layer {layer_num}")
                    gnn_mapping[layer_num] = f"GNN_{gnn_idx}"
        else:
            # Try to extract from discrete layers directly
            if hasattr(inner_model, 'discrete_sequential') and inner_model.discrete_sequential is not None:
                layers = list(inner_model.discrete_sequential)
                for idx, layer in enumerate(layers):
                    if hasattr(layer, 'selected_gnn'):
                        gnn_mapping[idx] = layer.selected_gnn
                    elif hasattr(layer, 'best_expert_idx'):
                        gnn_idx = layer.best_expert_idx
                        gnn_mapping[idx] = cfg.gt.head_gnn_types[gnn_idx] if gnn_idx < len(cfg.gt.head_gnn_types) else f"Expert_{gnn_idx}"
    
    # If still empty, try to infer from layer properties
    if not gnn_mapping:
        logging.warning("Could not extract GNN mapping from optimal weights, attempting to infer from layers...")
        if hasattr(model, 'model'):
            inner_model = model.model
            if hasattr(inner_model, 'discrete_sequential'):
                layers = list(inner_model.discrete_sequential)
                for idx, layer in enumerate(layers):
                    # Check various attributes that might indicate GNN type
                    if hasattr(layer, 'kv_model'):
                        # Try to identify GNN type from the module
                        kv_model = layer.kv_model
                        if kv_model is not None:
                            module_name = type(kv_model).__name__
                            if 'GINE' in module_name:
                                gnn_mapping[idx] = 'GINE'
                            elif 'GAT' in module_name:
                                gnn_mapping[idx] = 'GATv2' if 'v2' in module_name else 'GAT'
                            elif 'GatedGCN' in module_name:
                                gnn_mapping[idx] = 'GatedGCN'
                            elif 'GCN' in module_name:
                                gnn_mapping[idx] = 'GCN'
                            elif 'SAGE' in module_name:
                                gnn_mapping[idx] = 'SAGE'
                            else:
                                gnn_mapping[idx] = module_name
    
    return gnn_mapping


def mask_attention_head(model, layer_idx: int, head_idx: int, device):
    """
    Mask a specific attention head by zeroing its weights
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
    Restore original weights after masking
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
    
    # Extract GNN mapping from checkpoint
    gnn_mapping_from_checkpoint = {}
    if optimal_weights:
        logging.info(f"Optimal weights from checkpoint: {optimal_weights}")
        for layer_key, gnn_idx in optimal_weights.items():
            layer_num = int(layer_key.split('_')[1])
            if gnn_idx < len(cfg.gt.head_gnn_types):
                gnn_type = cfg.gt.head_gnn_types[gnn_idx]
                gnn_mapping_from_checkpoint[layer_num] = gnn_type
                logging.info(f"Checkpoint - Layer {layer_num}: {gnn_type}")
    
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
        
        # Fix dimensions BEFORE model creation
        if cfg.dataset.name in ['MOLHIV', 'ogbg-molhiv']:
            cfg.share.dim_out = 1
        elif cfg.dataset.name in ['MOLPCBA', 'ogbg-molpcba']:  
            cfg.share.dim_out = 128
        
        model = create_model(cfg)  # Now creates with correct dim_out
        
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
        
        return model
    
    # Get baseline performance
    logging.info("Getting baseline performance...")
    model = create_correct_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get GNN mapping from model or checkpoint
    gnn_mapping = get_gnn_type_mapping(model)
    if not gnn_mapping and gnn_mapping_from_checkpoint:
        gnn_mapping = gnn_mapping_from_checkpoint
        logging.info("Using GNN mapping from checkpoint")
    
    logging.info(f"Final GNN mapping: {gnn_mapping}")
    
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
    ablation_results = {}
    
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        if hasattr(layer, 'num_heads'):
            num_heads = layer.num_heads
            layer_results = {}
            
            # Get GNN type for this layer
            gnn_type = gnn_mapping.get(layer_idx, f"Layer_{layer_idx}")
            
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
                        'masked_performance': masked_perf,
                        'gnn_type': gnn_type
                    }
                    
                    restore_attention_head(model, layer_idx, orig_in_proj, orig_out_proj)
                    logging.info(f"Layer {layer_idx} ({gnn_type}), Head {head_idx}: drop={performance_drop:.4f}")
                
                del model
                torch.cuda.empty_cache()
            
            ablation_results[layer_idx] = layer_results
    
    # Restore config
    if actual_dim_out and actual_dim_out != original_dim_out:
        cfg.share.dim_out = original_dim_out
    
    logger.close()
    return ablation_results, baseline_perf, metric_key, gnn_mapping


def compute_interpretability_metrics(ablation_results):
    """
    Compute fidelity metrics with GNN type information
    """
    # Collect all heads with their drops
    all_heads = []
    
    for layer_idx in ablation_results.keys():
        for head_idx in ablation_results[layer_idx].keys():
            all_heads.append({
                'layer': layer_idx,
                'head': head_idx,
                'drop': ablation_results[layer_idx][head_idx]['performance_drop'],
                'gnn_type': ablation_results[layer_idx][head_idx].get('gnn_type', f'Layer_{layer_idx}')
            })
    
    if len(all_heads) < 2:
        return {
            'fidelity': 0.0,
            'fidelity_minus': 0.0,
            'num_heads_tested': len(all_heads),
            'sorted_heads': all_heads
        }
    
    # Sort by actual performance drop (descending)
    sorted_heads = sorted(all_heads, key=lambda x: x['drop'], reverse=True)
    
    # Calculate fidelity
    k = len(sorted_heads) // 2
    important_heads = sorted_heads[:k]
    unimportant_heads = sorted_heads[k:]
    
    fidelity = np.mean([h['drop'] for h in important_heads]) if important_heads else 0
    fidelity_minus = np.mean([h['drop'] for h in unimportant_heads]) if unimportant_heads else 0
    
    logging.info(f"\nFIDELITY METRICS:")
    logging.info(f"Fidelity (top {k} heads): {fidelity:.4f}")
    logging.info(f"Fidelity- (bottom {k} heads): {fidelity_minus:.4f}")
    
    # Log GNN distribution in important heads
    gnn_counts = {}
    for head in important_heads:
        gnn_type = head['gnn_type']
        gnn_counts[gnn_type] = gnn_counts.get(gnn_type, 0) + 1
    
    logging.info(f"\nGNN distribution in important heads:")
    for gnn_type, count in sorted(gnn_counts.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {gnn_type}: {count} heads")
    
    return {
        'fidelity': fidelity,
        'fidelity_minus': fidelity_minus,
        'num_heads_tested': len(all_heads),
        'sorted_heads': sorted_heads,
        'gnn_distribution': gnn_counts
    }


def export_to_excel(ablation_results, interpretability_metrics, baseline_perf, metric_name, gnn_mapping, save_path):
    """
    Export results to Excel with detailed head importance analysis and GNN types
    """
    # Prepare data for DataFrame
    data_rows = []
    
    for head_info in interpretability_metrics['sorted_heads']:
        layer_idx = head_info['layer']
        head_idx = head_info['head']
        drop = head_info['drop']
        gnn_type = head_info['gnn_type']
        
        data_rows.append({
            'Layer': layer_idx,
            'Head': head_idx,
            'Performance_Drop': drop,
            'GNN_Type': gnn_type,
            'Rank': len(data_rows) + 1,
            'Importance': 'High' if len(data_rows) < len(interpretability_metrics['sorted_heads']) // 2 else 'Low'
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Create Excel writer
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # Write main data
        df.to_excel(writer, sheet_name='Head_Importance', index=False)
        
        # Add GNN mapping sheet
        gnn_df = pd.DataFrame([
            {'Layer': layer, 'GNN_Type': gnn_type}
            for layer, gnn_type in sorted(gnn_mapping.items())
        ])
        gnn_df.to_excel(writer, sheet_name='Layer_GNN_Mapping', index=False)
        
        # Add summary sheet
        summary_data = {
            'Metric': ['Dataset', 'Baseline_Performance', f'Baseline_{metric_name}', 
                      'Fidelity', 'Fidelity_Minus', 'Total_Heads', 'Important_Heads'],
            'Value': [cfg.dataset.name, baseline_perf, baseline_perf,
                     interpretability_metrics['fidelity'], 
                     interpretability_metrics['fidelity_minus'],
                     interpretability_metrics['num_heads_tested'],
                     interpretability_metrics['num_heads_tested'] // 2]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the Excel file
        workbook = writer.book
        
        # Format Head_Importance sheet
        worksheet = writer.sheets['Head_Importance']
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            if col_idx < 26:  # Handle column naming
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width
    
    logging.info(f"Excel results saved to {save_path}")


def visualize_results(ablation_results, interpretability_metrics, gnn_mapping, save_dir):
    """
    Create enhanced visualizations with GNN type information
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    num_layers = len(ablation_results)
    if num_layers == 0:
        return
    num_heads = len(next(iter(ablation_results.values())))
    
    # Create performance drop matrix
    drop_matrix = np.zeros((num_layers, num_heads))
    
    for layer_idx, layer_results in ablation_results.items():
        for head_idx, results in layer_results.items():
            drop_matrix[layer_idx, head_idx] = results['performance_drop']
    
    # Create figure for performance drops with GNN annotations
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create y-axis labels with GNN types
    y_labels = [f'L{i} ({gnn_mapping.get(i, f"L{i}")})' for i in range(num_layers)]
    
    # Performance drop heatmap
    sns.heatmap(drop_matrix,
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=y_labels,
                cmap='RdYlBu_r',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Performance Drop'},
                center=0,
                ax=ax)
    ax.set_title(f'Head Importance via Ablation (Dataset: {cfg.dataset.name})\nFidelity={interpretability_metrics["fidelity"]:.4f}')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer (GNN Type)')
    plt.tight_layout()
    plt.savefig(save_dir / 'head_importance_heatmap_with_gnn.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create bar plot of top heads with GNN colors
    sorted_heads = interpretability_metrics['sorted_heads'][:15]  # Top 15 most important
    
    # Define colors for each GNN type
    gnn_colors = {
        'GINE': 'royalblue',
        'GATv2': 'forestgreen',
        'GatedGCN': 'crimson',
        'GAT': 'orange',
        'GCN': 'purple',
        'SAGE': 'brown'
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = [f'L{h["layer"]}-H{h["head"]}' for h in sorted_heads]
    drops = [h['drop'] for h in sorted_heads]
    colors = [gnn_colors.get(h['gnn_type'], 'gray') for h in sorted_heads]
    
    bars = ax.bar(range(len(x_labels)), drops, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Performance Drop')
    ax.set_title(f'Top {len(sorted_heads)} Most Important Attention Heads by GNN Type')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add GNN type annotations
    for i, (bar, head) in enumerate(zip(bars, sorted_heads)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{head["gnn_type"]}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, rotation=0)
    
    # Add legend for GNN types
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0,0),1,1, fc=color, alpha=0.7, edgecolor='black', label=gnn_type) 
                       for gnn_type, color in gnn_colors.items() if gnn_type in [h['gnn_type'] for h in sorted_heads]]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'top_important_heads_by_gnn.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create GNN distribution pie chart
    if 'gnn_distribution' in interpretability_metrics:
        fig, ax = plt.subplots(figsize=(8, 8))
        gnn_dist = interpretability_metrics['gnn_distribution']
        colors_pie = [gnn_colors.get(gnn, 'gray') for gnn in gnn_dist.keys()]
        
        wedges, texts, autotexts = ax.pie(gnn_dist.values(), 
                                           labels=gnn_dist.keys(), 
                                           colors=colors_pie,
                                           autopct='%1.1f%%',
                                           startangle=90)
        ax.set_title(f'GNN Distribution in Important Heads\n(Top {len(interpretability_metrics["sorted_heads"])//2} heads)')
        plt.savefig(save_dir / 'gnn_distribution_important_heads.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Saved visualizations to {save_dir}")


def run_pk_analysis(model, train_loader, test_loader):
    """
    Main entry point - enhanced with proper GNN type tracking
    """
    logging.info("="*60)
    logging.info("Starting PK-Explainer Analysis (Enhanced Version)")
    logging.info("="*60)
    logging.info(f"Dataset: {cfg.dataset.name}")
    logging.info(f"Model type: {cfg.model.type}")
    logging.info(f"GNN types available: {cfg.gt.head_gnn_types}")
    
    # Step 1: Save model for ablation
    model_path = Path(cfg.run_dir) / 'model_for_ablation.pt'
    optimal_weights = None
    if hasattr(model, 'model') and hasattr(model.model, 'optimal_weights_dict'):
        optimal_weights = model.model.optimal_weights_dict
        logging.info(f"Extracted optimal weights: {optimal_weights}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        'optimal_weights': optimal_weights,
        'config': cfg
    }, model_path)
    logging.info(f"Saved model state to {model_path}")
    
    # Step 2: Ablation study
    logging.info("\nPerforming ablation study...")
    ablation_results, baseline_perf, metric_name, gnn_mapping = ablation_study(model_path, test_loader)
    
    # Step 3: Compute metrics
    interpretability_metrics = compute_interpretability_metrics(ablation_results)
    
    logging.info(f"\nInterpretability Analysis:")
    logging.info(f"  Fidelity: {interpretability_metrics['fidelity']:.4f}")
    logging.info(f"  Fidelity-: {interpretability_metrics['fidelity_minus']:.4f}")
    logging.info(f"  Total heads tested: {interpretability_metrics['num_heads_tested']}")
    
    # Step 4: Export to Excel
    excel_path = Path(cfg.run_dir) / 'pk_explainer_results.xlsx'
    export_to_excel(ablation_results, interpretability_metrics, baseline_perf, 
                   metric_name, gnn_mapping, excel_path)
    
    # Step 5: Visualization
    if cfg.gt.pk_explainer.visualization:
        vis_dir = Path(cfg.run_dir) / 'pk_explainer_results'
        visualize_results(ablation_results, interpretability_metrics, gnn_mapping, vis_dir)
    
    # Step 6: Save JSON results (enhanced)
    results = {
        'dataset': cfg.dataset.name,
        'model_type': cfg.model.type,
        'gnn_mapping': gnn_mapping,
        'ablation_results': {
            str(layer): {
                str(head): {
                    'performance_drop': float(data['performance_drop']),
                    'masked_performance': float(data['masked_performance']),
                    'gnn_type': data.get('gnn_type', f'Layer_{layer}')
                }
                for head, data in heads.items()
            }
            for layer, heads in ablation_results.items()
        },
        'interpretability_metrics': {
            'fidelity': float(interpretability_metrics['fidelity']),
            'fidelity_minus': float(interpretability_metrics['fidelity_minus']),
            'num_heads_tested': interpretability_metrics['num_heads_tested'],
            'gnn_distribution': interpretability_metrics.get('gnn_distribution', {})
        },
        'baseline_performance': {
            metric_name: float(baseline_perf)
        }
    }
    
    results_file = Path(cfg.run_dir) / 'pk_explainer_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nPK-Explainer results saved to:")
    logging.info(f"  - Excel: {excel_path}")
    logging.info(f"  - JSON: {results_file}")
    logging.info(f"  - Visualizations: {vis_dir}" if cfg.gt.pk_explainer.visualization else "")
    
    # Clean up
    model_path.unlink()
    
    return results
