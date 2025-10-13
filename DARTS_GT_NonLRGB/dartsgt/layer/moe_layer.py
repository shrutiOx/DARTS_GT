# -*- coding: utf-8 -*-
"""
Optimized MOE Layer - Uncertainty only for last layer, only test/val
"""

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from dartsgt.layer.bigbird_layer import SingleBigBirdLayer
from dartsgt.layer.gatedgcn_layer import GatedGCNLayer
from dartsgt.layer.gine_conv_layer import GINEConvESLapPE

import torch.nn.functional as F
import random
from dartsgt.layer.graph_structural_features import compute_lean_moment_pooling
from torch_geometric.nn import global_mean_pool, global_max_pool

#TODO : Diversity Loss: Force experts to specialize on different graph types



class GraphRouter(nn.Module):
    def __init__(self, dim_h, num_experts):
        super().__init__()
        #self.router_dropout = nn.Dropout(0.2)
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        input_dim = dim_h * 2 + 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h//2),
            nn.ReLU(),
            nn.Linear(dim_h//2, num_experts)
        )
    
    def forward(self, graph_embeddings):
        #graph_embeddings = self.router_dropout(graph_embeddings)  # ADD THIS
        graph_embeddings = graph_embeddings
        expert_scores = self.mlp(graph_embeddings)
        
        return F.softmax(expert_scores / self.temperature, dim=-1)


class MOELayer(nn.Module):
    """Optimized MOE layer - uncertainty only for last layer during test/val."""

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, head_gnn_types, routing_mode, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False, uncertainty_enabled=True, is_last_layer=False,
                 perturbation_delta=None, perturbation_epsilon=None, perturbation_max_steps=None, uncertainty_samples=None):
        super().__init__()

        # Basic parameters
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.head_gnn_types = head_gnn_types
        self.routing_mode = routing_mode
        self.uncertainty_enabled = uncertainty_enabled
        self.is_last_layer = is_last_layer
        self.expert_dropout = nn.Dropout(0.3)
        
        # Add after self.is_last_layer = is_last_layer
        self.perturbation_delta = perturbation_delta if perturbation_delta is not None else 0.02
        self.perturbation_epsilon = perturbation_epsilon if perturbation_epsilon is not None else 0.15
        self.perturbation_max_steps = perturbation_max_steps if perturbation_max_steps is not None else 10
        self.uncertainty_samples = uncertainty_samples if uncertainty_samples is not None else 5
                
        # Expert configuration - number of experts equals number of GNN types
        self.num_experts = len(head_gnn_types)
        
        # Initialize modules
        self.kv_model = nn.ModuleDict()
        
        '''

        # Create local message-passing model
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == "SAGE":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.SAGEConv(self.dim_h, self.dim_h,'max')
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type'''
        
        # Create SINGLE multi-head attention module
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: {global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")
            
        # Create KV models - ONE per expert type
        for i, gnn_type in enumerate(head_gnn_types):
            self.kv_gnn_with_edge_attr = True
            if gnn_type == 'None':
                self.kv_model[str(i)] = None
            elif gnn_type == "GCN":
                self.kv_gnn_with_edge_attr = False
                self.kv_model[str(i)] = pygnn.GCNConv(dim_h, dim_h)
            elif gnn_type == 'GIN':
                self.kv_gnn_with_edge_attr = False
                gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                       self.activation(),
                                       Linear_pyg(dim_h, dim_h))
                self.kv_model[str(i)] = pygnn.GINConv(gin_nn)
            elif gnn_type == "SAGE":
                self.kv_gnn_with_edge_attr = False
                self.kv_model[str(i)] = pygnn.SAGEConv(self.dim_h, self.dim_h,'max')
            elif gnn_type == 'GENConv':
                self.kv_model[str(i)] = pygnn.GENConv(dim_h, dim_h)
            elif gnn_type == 'GINE':
                gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                       self.activation(),
                                       Linear_pyg(dim_h, dim_h))
                if self.equivstable_pe:
                    self.kv_model[str(i)] = GINEConvESLapPE(gin_nn)
                else:
                    self.kv_model[str(i)] = pygnn.GINEConv(gin_nn)
            elif gnn_type == 'GAT':
                self.kv_model[str(i)] = pygnn.GATConv(in_channels=dim_h,
                                                     out_channels=dim_h // num_heads,
                                                     heads=num_heads,
                                                     edge_dim=dim_h)
            elif gnn_type == 'GATV2':
                 self.kv_model[str(i)] = pygnn.GATv2Conv(in_channels=self.dim_h,
                                                  out_channels=self.dim_h // self.num_heads,
                                                  heads=self.num_heads,
                                                  edge_dim=self.dim_h)
            elif gnn_type == 'PNA':
                aggregators = ['mean', 'max', 'sum']
                scalers = ['identity']
                deg = torch.from_numpy(np.array(pna_degrees))
                self.kv_model[str(i)] = pygnn.PNAConv(dim_h, dim_h,
                                                     aggregators=aggregators,
                                                     scalers=scalers,
                                                     deg=deg,
                                                     edge_dim=min(128, dim_h),
                                                     towers=1,
                                                     pre_layers=1,
                                                     post_layers=1,
                                                     divide_input=False)
            elif gnn_type == 'CustomGatedGCN':
                self.kv_model[str(i)] = GatedGCNLayer(dim_h, dim_h,
                                                     dropout=dropout,
                                                     residual=True,
                                                     act=act,
                                                     equivstable_pe=equivstable_pe)
            else:
                raise ValueError(f"Unsupported KV GNN model: {gnn_type}")
        # Initialize router based on routing mode
        if self.routing_mode == 'moe':
            #self.router = GraphRouter(dim_h, self.num_experts)
            #self.router = GraphRouter(dim_h * 3 + 8, self.num_experts)  # 3 for moments, 8 for structural
            self.router = GraphRouter(dim_h, self.num_experts)
        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer', 'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )
            
        self.q_projection = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            self.activation(),
            nn.Linear(dim_h, dim_h)
        )
        
        
        # Normalization layers
        if self.layer_norm:
            #self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_kv = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            #self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_kv = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        #self.dropout_local = nn.Dropout(dropout)
        self.dropout_kv = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        
        
    def perturb_routing_weights(self, optimal_weights, delta=None, epsilon=None, max_steps=None):
        """Perturbation-based uncertainty quantification."""

        
        delta = delta or self.perturbation_delta
        epsilon = epsilon or self.perturbation_epsilon  
        max_steps = max_steps or self.perturbation_max_steps
        
        batch_size, num_experts = optimal_weights.shape
        device = optimal_weights.device
        
        perturbed_weights = optimal_weights.clone()
        current_distance = 0.0
        steps = 0
        
        while current_distance < epsilon and steps < max_steps:
            for b in range(batch_size):
                if num_experts < 2:
                    break
                i, j = random.sample(range(num_experts), 2)
                if perturbed_weights[b, i] >= delta:
                    perturbed_weights[b, i] -= delta
                    perturbed_weights[b, j] += delta
            
            current_distance = torch.norm(perturbed_weights - optimal_weights, dim=1).mean().item()
            steps += 1
        
        return perturbed_weights

    def forward(self, batch):
        h = batch.x
        h_in1 = h.clone()
        #h_out_list = []
        '''
        # 1. Local MPNN with edge attributes
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                  x=h,
                                                  edge_index=batch.edge_index,
                                                  edge_attr=batch.edge_attr,
                                                  pe_EquivStableLapPE=es_data))
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                            batch.edge_index,
                                            batch.edge_attr,
                                            batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                            batch.edge_index,
                                            batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local
            
            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)
            h_for_routing = h_local
        else:
            h_for_routing = h_in1'''
            
        moment_features = compute_lean_moment_pooling(h, batch.batch)
        structural_features = batch.graph_structural_features
        graph_embeddings = torch.cat([structural_features, moment_features], dim=-1)

        batch_size = len(torch.unique(batch.batch))
        
        if self.routing_mode == 'uniform':
            # Equal weights for all experts
            expert_routing_weights = torch.ones(batch_size, self.num_experts, device=h.device) / self.num_experts
            #nas_weights = torch.tensor([0.25117602944374084, 0.2520402669906616, 0.2501198947429657, 0.24666379392147064], 
            #                  device=h.device, dtype=torch.float32)
            #expert_routing_weights = nas_weights.unsqueeze(0).repeat(batch_size, 1)
        elif self.routing_mode == 'random':
            # Random weights for experts
            random_weights = torch.rand(batch_size, self.num_experts, device=h.device)
            expert_routing_weights = F.softmax(random_weights, dim=1)
        elif self.routing_mode == 'moe':
            # Use router to compute expert weights
            expert_routing_weights = self.router(graph_embeddings)
        else:
            raise ValueError(f"Unsupported routing mode: {self.routing_mode}")
        #expert_routing_weights = torch.round(expert_routing_weights * 100) / 100
        # Store routing weights for logging
        if hasattr(batch, 'routing_weights'):
            batch.routing_weights.append(expert_routing_weights.detach().cpu())
        else:
            batch.routing_weights = [expert_routing_weights.detach().cpu()]
        
        # 3. Process ALL experts in parallel and collect their outputs
        expert_outputs = []
        
        for expert_idx in range(self.num_experts):
            h_kv = h_in1.clone()
            gnn_type = self.head_gnn_types[expert_idx]
            
            if self.kv_model[str(expert_idx)] is not None:
                if gnn_type == 'CustomGatedGCN':
                    es_data = None
                    if self.equivstable_pe:
                        es_data = batch.pe_EquivStableLapPE
                    kv_out = self.kv_model[str(expert_idx)](Batch(batch=batch,
                                                          x=h_kv,
                                                          edge_index=batch.edge_index,
                                                          edge_attr=batch.edge_attr,
                                                          pe_EquivStableLapPE=es_data))
                    h_kv = kv_out.x
                else:
                    if self.kv_gnn_with_edge_attr:
                        if self.equivstable_pe:
                            h_kv = self.kv_model[str(expert_idx)](h_kv,
                                                batch.edge_index,
                                                batch.edge_attr,
                                                batch.pe_EquivStableLapPE)
                        else:
                            h_kv = self.kv_model[str(expert_idx)](h_kv,
                                                batch.edge_index,
                                                batch.edge_attr)
                    else:
                        h_kv = self.kv_model[str(expert_idx)](h_kv, batch.edge_index)
                    h_kv = self.dropout_kv(h_kv)
                
                if self.layer_norm:
                    h_kv = self.norm1_kv(h_kv, batch.batch)
                if self.batch_norm:
                    h_kv = self.norm1_kv(h_kv)
            
            expert_outputs.append(h_kv)
        all_expert_outputs = torch.cat(expert_outputs, dim=0)  # Combine all experts
        global_mean = all_expert_outputs.mean()
        global_std = all_expert_outputs.std()
        
        # Apply SAME normalization to all experts
        expert_outputs = [
            (exp_out - global_mean) / (global_std + 1e-8) 
            for exp_out in expert_outputs
        ]
        expert_outputs = [self.expert_dropout(exp_out) for exp_out in expert_outputs]
        
        # 4. Apply attention - with or without uncertainty computation
        if self.self_attn is not None:
            # Convert query to dense format
            q_dense, mask = to_dense_batch(h_in1, batch.batch)
            q_dense = self.q_projection(q_dense) 
            
            # Convert all expert outputs to dense format
            expert_outputs_dense = []
            for expert_out in expert_outputs:
                expert_dense, _ = to_dense_batch(expert_out, batch.batch)
                expert_outputs_dense.append(expert_dense)
            #expert_outputs_dense = expert_outputs_dense + 0.1 * h_in1  
            # Stack expert outputs: [num_experts, batch_size, seq_len, dim]
            expert_outputs_stacked = torch.stack(expert_outputs_dense, dim=0)
            #expert_outputs_normalized = F.normalize(expert_outputs_stacked, p=2, dim=-1)
            # === ROUTING VARIANCE & UNCERTAINTY COMPUTATION ===
            should_compute_variance = (self.is_last_layer and 
                                      self.uncertainty_enabled and 
                                      not self.training and
                                      hasattr(batch, 'split') and batch.split in ['val', 'test'])

            should_compute_uncertainty = (self.is_last_layer and 
                                         self.uncertainty_enabled and 
                                         not self.training and
                                         hasattr(batch, 'split') and batch.split == 'test')  # Only test

            if should_compute_variance:
                # Compute routing variance (MOE vs uniform vs random)
                expert_weights_expanded = expert_routing_weights.T.unsqueeze(2).unsqueeze(3)
                kv_combined_moe_dense = torch.sum(expert_outputs_stacked * expert_weights_expanded, dim=0)
                
               
                #kv_combined_moe_dense = torch.sum(expert_outputs_normalized * expert_weights_expanded, dim=0)
                if self.training:
                    kv_combined_moe_dense = F.dropout(kv_combined_moe_dense, p=0.2, training=True)
                
                uniform_weights = torch.ones_like(expert_routing_weights) / self.num_experts
                uniform_weights_expanded = uniform_weights.T.unsqueeze(2).unsqueeze(3)
                kv_combined_uniform_dense = torch.sum(expert_outputs_stacked * uniform_weights_expanded, dim=0)
                #kv_combined_uniform_dense = torch.sum(expert_outputs_normalized * uniform_weights_expanded, dim=0)

                random_weights = F.softmax(torch.randn_like(expert_routing_weights), dim=-1)
                random_weights_expanded = random_weights.T.unsqueeze(2).unsqueeze(3)
                kv_combined_random_dense = torch.sum(expert_outputs_stacked * random_weights_expanded, dim=0)
                #kv_combined_random_dense = torch.sum(expert_outputs_normalized * random_weights_expanded, dim=0)
                
                # Apply attention for each strategy
                if self.global_model_type == 'Transformer':
                    h_attn_moe = self.self_attn(q_dense, kv_combined_moe_dense, kv_combined_moe_dense,
                                               key_padding_mask=~mask, need_weights=False)[0][mask]
                    h_attn_uniform = self.self_attn(q_dense, kv_combined_uniform_dense, kv_combined_uniform_dense,
                                                   key_padding_mask=~mask, need_weights=False)[0][mask]
                    h_attn_random = self.self_attn(q_dense, kv_combined_random_dense, kv_combined_random_dense,
                                                  key_padding_mask=~mask, need_weights=False)[0][mask]
                elif self.global_model_type == 'Performer':
                    h_attn_moe = self.self_attn(q_dense, kv_combined_moe_dense, mask=mask)[mask]
                    h_attn_uniform = self.self_attn(q_dense, kv_combined_uniform_dense, mask=mask)[mask]
                    h_attn_random = self.self_attn(q_dense, kv_combined_random_dense, mask=mask)[mask]
                    
                elif self.global_model_type == 'BigBird':
                    h_attn_moe = self.self_attn(q_dense, kv_combined_moe_dense, attention_mask=mask)
                    h_attn_uniform = self.self_attn(q_dense, kv_combined_uniform_dense, attention_mask=mask)
                    h_attn_random = self.self_attn(q_dense, kv_combined_random_dense, attention_mask=mask)
                    
                elif self.global_model_type == 'BiasedTransformer':
                    h_attn_moe = self.self_attn(q_dense, kv_combined_moe_dense, kv_combined_moe_dense, batch.attn_bias,
                                               key_padding_mask=~mask,
                                               need_weights=False)[0][mask]
                    h_attn_uniform = self.self_attn(q_dense, kv_combined_uniform_dense, kv_combined_uniform_dense, batch.attn_bias,
                                                   key_padding_mask=~mask,
                                                   need_weights=False)[0][mask]
                    h_attn_random = self.self_attn(q_dense, kv_combined_random_dense, kv_combined_random_dense, batch.attn_bias,
                                                  key_padding_mask=~mask,
                                                  need_weights=False)[0][mask]
                else:
                    raise RuntimeError(f"Unexpected {self.global_model_type}")

                
                # Apply post-processing
                h_attn_moe = self.dropout_attn(h_attn_moe) + h_in1
                h_attn_uniform = self.dropout_attn(h_attn_uniform) + h_in1  
                h_attn_random = self.dropout_attn(h_attn_random) + h_in1
                
                if self.layer_norm:
                    h_attn_moe = self.norm1_attn(h_attn_moe, batch.batch)
                    h_attn_uniform = self.norm1_attn(h_attn_uniform, batch.batch)
                    h_attn_random = self.norm1_attn(h_attn_random, batch.batch)
                if self.batch_norm:
                    h_attn_moe = self.norm1_attn(h_attn_moe)
                    h_attn_uniform = self.norm1_attn(h_attn_uniform)
                    h_attn_random = self.norm1_attn(h_attn_random)
                
                # Compute routing variance
                outputs = torch.stack([h_attn_moe, h_attn_uniform, h_attn_random], dim=0)
                node_variance = torch.var(outputs, dim=0).mean(dim=-1)
                graph_variance = global_mean_pool(node_variance, batch.batch)
                batch.routing_variance = graph_variance
                
                h_attn = h_attn_moe
                
                # === PERTURBATION-BASED UNCERTAINTY (TEST ONLY) ===
                if should_compute_uncertainty:
                    perturbation_predictions = []
                    
                    for sample_idx in range(self.uncertainty_samples):
                        perturbed_weights = self.perturb_routing_weights(expert_routing_weights)
                        perturbed_weights_expanded = perturbed_weights.T.unsqueeze(2).unsqueeze(3)
                        kv_combined_perturbed = torch.sum(expert_outputs_stacked * perturbed_weights_expanded, dim=0)
                        if self.training:
                            kv_combined_perturbed = F.dropout(kv_combined_perturbed, p=0.2, training=True)
                        
                        if self.global_model_type == 'Transformer':
                            h_attn_perturbed = self.self_attn(q_dense, kv_combined_perturbed, kv_combined_perturbed,
                                                             key_padding_mask=~mask, need_weights=False)[0][mask]
                        # ... other model types ...
                        elif self.global_model_type == 'Performer':
                            h_attn_perturbed = self.self_attn(q_dense, kv_combined_perturbed, mask=mask)[mask]
                            
                        elif self.global_model_type == 'BigBird':
                            h_attn_perturbed = self.self_attn(q_dense, kv_combined_perturbed, attention_mask=mask)
                            
                        elif self.global_model_type == 'BiasedTransformer':
                            h_attn_perturbed = self.self_attn(q_dense, kv_combined_perturbed, kv_combined_perturbed, batch.attn_bias,
                                                   key_padding_mask=~mask,
                                                   need_weights=False)[0][mask]
                        else:
                            raise RuntimeError(f"Unexpected {self.global_model_type}")
                        
                        h_attn_perturbed = self.dropout_attn(h_attn_perturbed) + h_in1
                        if self.layer_norm:
                            h_attn_perturbed = self.norm1_attn(h_attn_perturbed, batch.batch)
                        if self.batch_norm:
                            h_attn_perturbed = self.norm1_attn(h_attn_perturbed)
                        
                        perturbation_predictions.append(h_attn_perturbed)
                    
                    # Compute uncertainty from perturbation variance
                    perturbation_stack = torch.stack(perturbation_predictions, dim=0)
                    node_uncertainty = torch.var(perturbation_stack, dim=0).mean(dim=-1)
                    graph_uncertainty = global_mean_pool(node_uncertainty, batch.batch)
                    batch.routing_uncertainty = graph_uncertainty

            else:
                # Standard MOE without variance/uncertainty computation
                expert_weights_expanded = expert_routing_weights.T.unsqueeze(2).unsqueeze(3)
                kv_combined_dense = torch.sum(expert_outputs_stacked * expert_weights_expanded, dim=0)
                if self.training:
                    kv_combined_dense = F.dropout(kv_combined_dense, p=0.2, training=True)
                
                if self.global_model_type == 'Transformer':
                    h_attn = self.self_attn(q_dense, kv_combined_dense, kv_combined_dense,
                                           key_padding_mask=~mask, need_weights=False)[0][mask]
                elif self.global_model_type == 'Performer':
                    h_attn = self.self_attn(q_dense, kv_combined_dense, mask=mask)[mask]
                    
                elif self.global_model_type == 'BigBird':
                    h_attn = self.self_attn(q_dense, kv_combined_dense, attention_mask=mask)
                    
                elif self.global_model_type == 'BiasedTransformer':
                    h_attn = self.self_attn(q_dense, kv_combined_dense, kv_combined_dense, batch.attn_bias,
                                           key_padding_mask=~mask,
                                           need_weights=False)[0][mask]
                else:
                    raise RuntimeError(f"Unexpected {self.global_model_type}")
                
                h_attn = self.dropout_attn(h_attn) + h_in1
                if self.layer_norm:
                    h_attn = self.norm1_attn(h_attn, batch.batch)
                if self.batch_norm:
                    h_attn = self.norm1_attn(h_attn)
            
            #h_out_list.append(h_attn)

        # 5. Combine local and attention outputs
        h = h_attn

        # 6. Feed Forward block
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        
        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'num_heads={self.num_heads}, ' \
            f'num_experts={self.num_experts}, ' \
            f'is_last_layer={self.is_last_layer}'
        return s
