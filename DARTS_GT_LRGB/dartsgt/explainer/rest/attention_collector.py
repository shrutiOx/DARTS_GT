# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 22:11:00 2025

@author: ADMIN
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class AttentionCollector:
    """Simple collector for attention weights during forward passes"""
    
    def __init__(self):
        self.attention_weights = {}  # layer_idx -> list of (attention, edge_index)
        self.enabled = False
        
    def start_collecting(self):
        """Enable attention collection"""
        self.enabled = True
        self.attention_weights.clear()
        
    def stop_collecting(self):
        """Disable attention collection"""
        self.enabled = False
        
    def add_attention(self, 
                     layer_idx: int, 
                     attention: torch.Tensor,
                     edge_index: Optional[torch.Tensor] = None,
                     batch_idx: Optional[torch.Tensor] = None):
        """Add attention weights from a layer
        
        Args:
            layer_idx: Which layer this is from
            attention: For dense [B, H, N, N] or for sparse [E, H]
            edge_index: For sparse attention [2, E]
            batch_idx: Batch assignment for nodes
        """
        if not self.enabled:
            return
            
        if layer_idx not in self.attention_weights:
            self.attention_weights[layer_idx] = []
            
        self.attention_weights[layer_idx].append({
            'attention': attention.detach().cpu(),
            'edge_index': edge_index.detach().cpu() if edge_index is not None else None,
            'batch_idx': batch_idx.detach().cpu() if batch_idx is not None else None
        })
        
    def get_layer_attention(self, layer_idx: int) -> List[Dict]:
        """Get all attention weights for a specific layer"""
        return self.attention_weights.get(layer_idx, [])
    
    def clear(self):
        """Clear all collected attention weights"""
        self.attention_weights.clear()
