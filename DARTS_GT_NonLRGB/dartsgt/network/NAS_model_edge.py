# -*- coding: utf-8 -*-
"""
NAS Model - GPS Pipeline Compatible with Two-Phase Training
DARTS Phase: Uses NAS_layer with InputChoice selection
Discrete Phase: Uses DiscreteNAS_layer with fixed weights
"""

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
import torch.nn as nn
from dartsgt.layer.NAS_layer_edge import NASLayer
from dartsgt.layer.discrete_nas_layer_edge import DiscreteNASLayer
import torch_geometric.nn as pygnn
from nni.retiarii import model_wrapper
import numpy as np
import logging
from performer_pytorch import CrossAttention


# EXACT SAME FeatureEncoder as MOE
class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features - IDENTICAL to MOE
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        # Add debug info
        #print(f"Before encoder - batch.x.shape: {batch.x.shape}")
        #print(f"Before encoder - batch.x.dtype: {batch.x.dtype}")
        # Ensure correct data type for embeddings
        
        
        for module in self.children():
            
            batch = module(batch)
        return batch

#@model_wrapper
@register_network('NASModelEdge')
class NASModelEdge(torch.nn.Module):
    """
    NAS Enhanced Model with Two-Phase Training
    
    DARTS Phase: Uses NAS layers with InputChoice selection
    Discrete Phase: Uses discrete layers with fixed weights
    """

    def __init__(self, dim_in, dim_out,create_darts_layers=True):
        super().__init__()
        
        # EXACT SAME encoder as MOE
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        self.dim_out = dim_out
        # EXACT SAME pre-MP as MOE
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        # EXACT SAME dimension validation as MOE
        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        # EXACT SAME layer type parsing as MOE
        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        
        # EXACT SAME uncertainty configuration as MOE
        uncertainty_enabled = cfg.gt.uncertainty.enabled
        num_layers = cfg.gt.layers
        
        # Training mode and optimal weights storage
        self.training_mode = 'darts'  # 'darts' or 'discrete'
        self.optimal_weights = None   # Store DARTS-found optimal weights
        
        # Store layer configuration for creating discrete layers later
        self.layer_config = {
            'dim_h': cfg.gt.dim_hidden,
            'local_gnn_type': local_gnn_type,
            'global_model_type': global_model_type,
            'head_gnn_types': cfg.gt.head_gnn_types,
            'routing_mode': cfg.gt.routing_mode,
            'num_heads': cfg.gt.n_heads,
            'act': cfg.gnn.act,
            'pna_degrees': cfg.gt.pna_degrees,
            'equivstable_pe': cfg.posenc_EquivStableLapPE.enable,
            'dropout': cfg.gt.dropout,
            'attn_dropout': cfg.gt.attn_dropout,
            'layer_norm': cfg.gt.layer_norm,
            'batch_norm': cfg.gt.batch_norm,
            'bigbird_cfg': cfg.gt.bigbird,
            #'log_attn_weights': cfg.train.mode == 'log-attn-weights',
            # In layer_config dict (line ~85), ensure it has:
            'log_attn_weights': cfg.train.mode == 'log-attn-weights' or (hasattr(cfg.gt, 'pk_explainer') and cfg.gt.pk_explainer.enabled),
            'uncertainty_enabled': uncertainty_enabled,
            'gnns_type_used':cfg.gt.gnns_type_used,
            'perturbation_delta': getattr(cfg.gt.uncertainty, 'delta', 0.02),
            'perturbation_epsilon': getattr(cfg.gt.uncertainty, 'epsilon', 0.15), 
            'perturbation_max_steps': getattr(cfg.gt.uncertainty, 'max_steps', 10),
            'uncertainty_samples': getattr(cfg.gt.uncertainty, 'samples', 5),
            'epochs': getattr(cfg.gt.nas, 'darts_epochs', 50),
            
        }
        
        # Create DARTS layers (for architecture search phase)

        if create_darts_layers:
            # Parse weight_fix and calculate groups
            weight_fix = getattr(cfg.gt, 'weight_fix', 'individual')
            group_mapping = self._calculate_group_mapping(num_layers, weight_fix)
            
            self.darts_layers = nn.ModuleList()
            for layer_idx in range(num_layers):
                group_id = group_mapping[layer_idx]
                group_layers = [i for i, g in group_mapping.items() if g == group_id]
                
                layer = NASLayer(
                    nas_config=getattr(cfg.gt, 'nas', None),
                    layer_idx=layer_idx,
                    group_id=group_id,
                    group_layers=group_layers,
                    **self.layer_config
                )
                self.darts_layers.append(layer)
            self.darts_sequential = nn.Sequential(*self.darts_layers)
        else:  # ← ADD ELSE BLOCK
            self.darts_layers = None
            self.darts_sequential = None
    
        self.discrete_sequential = None
        # EXACT SAME post-MP head as MOE
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
        
        
    def _calculate_group_mapping(self, num_layers, weight_fix):
        """Calculate which group each layer belongs to"""
        if weight_fix == 'individual':
            # Each layer gets its own group
            return {i: i for i in range(num_layers)}
        
        elif weight_fix == 'all_single':
            # All layers in one group
            return {i: 0 for i in range(num_layers)}
        
        else:
            # Parse patterns like "3:3:2" which means first 3 layers in group 0, 
            # next 3 in group 1, next 2 in group 2
            try:
                # Split by colon to get group sizes
                group_sizes = [int(x) for x in weight_fix.split(':')]
                
                # Verify total matches num_layers
                if sum(group_sizes) != num_layers:
                    raise ValueError(f"Sum of group sizes {sum(group_sizes)} doesn't match num_layers {num_layers}")
                
                # Create mapping
                mapping = {}
                layer_idx = 0
                for group_id, group_size in enumerate(group_sizes):
                    for _ in range(group_size):
                        mapping[layer_idx] = group_id
                        layer_idx += 1
                
                return mapping
                
            except Exception as e:
                raise ValueError(f"Invalid weight_fix format '{weight_fix}'. Use 'individual', 'all_single', or patterns like '3:3:2'")
                
        
    def get_darts_model(self):
        """Return model configured for DARTS training"""
        self.training_mode = 'darts'
        return self
    
    def get_discrete_model(self, optimal_weights,weight_type ):
        """Create NEW discrete model with layer-specific optimal weights"""
        # Print original model parameters
        original_params = sum(p.numel() for p in self.parameters())
        print(f"=== MODEL PARAMETER COMPARISON ===")
        print(f"Original DARTS model parameters: {original_params:,}")
        #from dartsgt.network.nas_model import NASModel
        # Use original raw dimensions, not processed ones
        discrete_model = NASModelEdge(cfg.share.dim_in, self.dim_out, create_darts_layers=False)
        discrete_model.create_discrete_layers(optimal_weights,weight_type )
        discrete_model.training_mode = 'discrete'
        
        # Print parameters for comparison
        discrete_params = sum(p.numel() for p in discrete_model.parameters())
        original_params = sum(p.numel() for p in self.parameters())
        print(f"Fresh discrete model parameters: {discrete_params:,}")  
        print(f"Parameter difference: {discrete_params - original_params:,}")
        
        # Move to device
        device = torch.device(cfg.accelerator)
        discrete_model = discrete_model.to(device)
        
        print("✓ Used cfg.share.dim_in (raw input) instead of self.encoder.dim_in (processed)")
        print("✓ Fresh discrete model with correct input dimensions")
        
        return discrete_model

    def create_discrete_layers(self, layer_weights_dict, weight_type):
        """
        Create discrete layers with SINGLE BEST EXPERT per layer
        """
        discrete_layers = [] 
        num_layers = cfg.gt.layers
        
        for layer_idx in range(num_layers):
            layer_key = f'layer_{layer_idx}'
            
            # Get best expert index for this layer
            best_expert_idx = layer_weights_dict.get(layer_key, None)
            
            if best_expert_idx is None:
                raise ValueError(f"No best expert found for {layer_key}")
                
            best_expert_idx = int(best_expert_idx)
                
            # Log which expert this layer will use
            logging.info(f"Layer {layer_idx}: Using ONLY Expert {best_expert_idx} ({cfg.gt.head_gnn_types[best_expert_idx]})")
            
            is_last_layer = (layer_idx == num_layers - 1)
            
            # Create discrete layer with single expert
            discrete_layer = DiscreteNASLayer(
                is_last_layer=is_last_layer,
                optimal_weights=best_expert_idx,  # Pass the expert index directly
                layer_idx=layer_idx,
                **self.layer_config
            )
            discrete_layers.append(discrete_layer)
        
        self.discrete_sequential = nn.Sequential(*discrete_layers)
        
    def get_metrics_dict(self):
        """
        Get metrics from DARTS layers (only available in DARTS mode)
        """
        if self.training_mode != 'darts':
            raise ValueError("Metrics only available in DARTS mode")
        
        # Aggregate metrics from all DARTS layers
        all_metrics = {
            'per_epoch_metrics': {'alphas': {}},
            'cumulative_metrics': {'alphas': {}, 'selected_operations': {}}
        }
        
        for layer_idx, layer in enumerate(self.darts_layers):
            layer_metrics = layer.get_metrics_dict()
            layer_key = f'layer_{layer_idx}'
            
            # Merge per-epoch metrics
            for epoch, epoch_data in layer_metrics['per_epoch_metrics']['alphas'].items():
                if epoch not in all_metrics['per_epoch_metrics']['alphas']:
                    all_metrics['per_epoch_metrics']['alphas'][epoch] = {}
                all_metrics['per_epoch_metrics']['alphas'][epoch][layer_key] = epoch_data
            
            # Merge cumulative metrics
            all_metrics['cumulative_metrics']['alphas'][layer_key] = layer_metrics['cumulative_metrics']['alphas']
            all_metrics['cumulative_metrics']['selected_operations'][layer_key] = layer_metrics['cumulative_metrics']['selected_operations']
        # DEBUG: Print what we actually have
        #print("=== DEBUG METRICS_DICT ===")
        #print(f"all_metrics keys: {all_metrics.keys()}")
        #print(f"cumulative_metrics keys: {all_metrics['cumulative_metrics'].keys()}")
        #print(f"cumulative_metrics alphas: {all_metrics['cumulative_metrics']['alphas']}")
       # print("=== END DEBUG ===")
        return all_metrics
    


    def forward(self, batch):
        """Forward pass - same as MOE except uses NAS/discrete layers"""
        batch = self.encoder(batch)
        
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        
  
                
        if self.training_mode != 'darts':
            if self.discrete_sequential is not None:
                batch = self.discrete_sequential(batch)  # ← Single call like MOE
        else:  # discrete mode
            batch = self.darts_sequential(batch)  # ← Single call like MOE
            
        
        batch = self.post_mp(batch)
        return batch
    
    
    
