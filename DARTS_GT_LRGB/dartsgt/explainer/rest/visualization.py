# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 22:11:45 2025

@author: ADMIN
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict
import torch

def visualize_head_importance(importance_scores: Dict, results: Dict, save_dir: Path):
    """Create visualizations for head importance scores"""
    
    # 1. Heatmap of head importance across layers
    num_layers = len(importance_scores)
    num_heads = len(next(iter(importance_scores.values())))
    
    importance_matrix = np.zeros((num_layers, num_heads))
    for layer_idx, head_scores in importance_scores.items():
        for head_idx, score in head_scores.items():
            importance_matrix[layer_idx, head_idx] = score
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(importance_matrix, 
                xticklabels=[f'Head {i}' for i in range(num_heads)],
                yticklabels=[f'Layer {i}' for i in range(num_layers)],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f')
    plt.title('Head Importance Scores (KL Divergence)')
    plt.tight_layout()
    plt.savefig(save_dir / 'head_importance_heatmap.png', dpi=300)
    plt.close()
    
    # 2. Bar plot comparing hypotheses
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (hypothesis, data) in enumerate(results.items()):
        ax = axes[idx]
        iac = data['iac']
        classification = data['classification']
        
        # Flatten importance scores
        all_scores = []
        labels = []
        for layer_idx, head_scores in importance_scores.items():
            for head_idx, score in head_scores.items():
                all_scores.append(score)
                labels.append(f'L{layer_idx}H{head_idx}')
        
        # Sort by importance
        if hypothesis == 'specialization':
            sorted_idx = np.argsort(all_scores)[::-1]
        else:
            sorted_idx = np.argsort(all_scores)
            
        # Plot top 10
        top_10_idx = sorted_idx[:10]
        top_scores = [all_scores[i] for i in top_10_idx]
        top_labels = [labels[i] for i in top_10_idx]
        
        ax.bar(range(10), top_scores)
        ax.set_xticks(range(10))
        ax.set_xticklabels(top_labels, rotation=45)
        ax.set_ylabel('Importance Score')
        ax.set_title(f'{hypothesis.capitalize()} Hypothesis\nIAC={iac:.3f} ({classification})')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'hypothesis_comparison.png', dpi=300)
    plt.close()
    
    # 3. Layer-wise importance distribution
    plt.figure(figsize=(10, 6))
    for layer_idx, head_scores in importance_scores.items():
        scores = list(head_scores.values())
        plt.plot(scores, marker='o', label=f'Layer {layer_idx}')
    
    plt.xlabel('Head Index')
    plt.ylabel('Importance Score')
    plt.title('Head Importance Distribution by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'layer_importance_distribution.png', dpi=300)
    plt.close()


def visualize_attention_patterns(collector, importance_scores: Dict, save_dir: Path):
    """Visualize attention patterns for most/least important heads"""
    
    # This is a placeholder - actual implementation would depend on 
    # specific visualization needs and graph structure
    # Could show attention matrices, graph attention overlays, etc.
    pass
