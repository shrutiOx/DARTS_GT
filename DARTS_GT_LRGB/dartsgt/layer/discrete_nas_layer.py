# -*- coding: utf-8 -*-
"""
Created on Mon May 26 21:58:38 2025

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Discrete NAS Layer - Fixed weights version for discrete training phase
IDENTICAL architecture to NAS_layer but with fixed expert weights
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
import logging
from dartsgt.layer.bigbird_layer import SingleBigBirdLayer
from dartsgt.layer.gatedgcn_layer import GatedGCNLayer
from dartsgt.layer.gine_conv_layer import GINEConvESLapPE
from torch_geometric.graphgym.config import cfg
from torch_scatter import scatter_add
import torch.nn.functional as F
import random
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max
from performer_pytorch import CrossAttention



def pyg_softmax(src, index, num_nodes=None):
    """Sparse softmax - only normalizes over connected nodes"""
    num_nodes = maybe_num_nodes(index, num_nodes)
    
    # Subtract max for numerical stability
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    # Normalize only within each node's neighborhood
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class SparseGraphAttention(nn.Module):
    """Sparse attention that only attends to connected nodes"""
    
    def __init__(self, dim_h, num_heads, dropout=0.0):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.head_dim = dim_h // num_heads
        self.dropout = dropout
        
        # Q, K, V projections (separate for your architecture)
        self.W_k = nn.Linear(dim_h, dim_h)
        self.W_v = nn.Linear(dim_h, dim_h)
        self.W_o = nn.Linear(dim_h, dim_h)
        
    def forward(self, q, kv, edge_index,save_attn=False):
        """
        q: query features (batch.x after projection)
        kv: key-value features (GNN outputs)
        edge_index: graph connectivity
        """
        # Project K, V from kv (GNN outputs)
        K = self.W_k(kv).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(kv).view(-1, self.num_heads, self.head_dim)
        
        # Q already projected in main layer
        Q = q.view(-1, self.num_heads, self.head_dim)
        
        # Get attention scores only for connected nodes
        src_K = K[edge_index[0]]  # [E, heads, head_dim]
        dest_Q = Q[edge_index[1]]  # [E, heads, head_dim]
        
        # Compute attention scores
        scores = (src_K * dest_Q).sum(dim=-1) / np.sqrt(self.head_dim)  # [E, heads]
        
        # Sparse softmax - only over connected nodes
        attn_weights = pyg_softmax(scores, edge_index[1], num_nodes=Q.size(0))
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # In forward method after computing attn_weights
        if save_attn:
            self.sparse_attn_weights = (attn_weights.detach(), edge_index)
        
        # Apply attention to values
        src_V = V[edge_index[0]]  # [E, heads, head_dim]
        messages = src_V * attn_weights.unsqueeze(-1)  # [E, heads, head_dim]
        
        # Aggregate messages
        out = torch.zeros(Q.size(0), self.num_heads, self.head_dim, 
                         device=Q.device, dtype=Q.dtype)
        scatter(messages, edge_index[1], dim=0, out=out, reduce='add')
        
        # Reshape and project
        out = out.view(-1, self.dim_h)
        out = self.W_o(out)
        
        return out



class DiscreteNASLayer(nn.Module):
    """
    Discrete NAS Layer - Fixed weights version for final training
    IDENTICAL to NAS_layer but uses fixed expert weights instead of learnable selection
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, head_gnn_types, routing_mode, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False, uncertainty_enabled=True, is_last_layer=False,
                 perturbation_delta=None, perturbation_epsilon=None, perturbation_max_steps=None, 
                 uncertainty_samples=None, optimal_weights=None,layer_idx=None,epochs=20,gnns_type_used=None):
        super().__init__()

        # EXACT SAME basic parameters as NAS_layer
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
        self.layer_idx = layer_idx 
        # EXACT SAME uncertainty parameters
        self.perturbation_delta = perturbation_delta if perturbation_delta is not None else 0.02
        self.perturbation_epsilon = perturbation_epsilon if perturbation_epsilon is not None else 0.15
        self.perturbation_max_steps = perturbation_max_steps if perturbation_max_steps is not None else 10
        self.uncertainty_samples = uncertainty_samples if uncertainty_samples is not None else 5
        self.expert_dropout = nn.Dropout(0.3)
        self.gnns_type_used=gnns_type_used
        # Store fixed optimal weights
        self.optimal_weights = optimal_weights
        # VERIFY ORDER CONSISTENCY
        
        # Expert configuration
        self.num_experts = len(head_gnn_types)
      


        self.best_expert_idx = optimal_weights
        self.selected_gnn = head_gnn_types[self.best_expert_idx]
        logging.info(f"DiscreteNASLayer {layer_idx}: Using ONLY Expert {self.best_expert_idx} ({self.selected_gnn})")
       
        
        self.kv_model = nn.ModuleDict()
        
        self.log_attn_weights = log_attn_weights or (hasattr(cfg.gt, 'pk_explainer') and cfg.gt.pk_explainer.enabled)
        if self.log_attn_weights and global_model_type not in ['Transformer',
                                                       'BiasedTransformer', 
                                                       'SparseTransformer',
                                                       'Performer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )
         
        
        # EXACT SAME global attention model as NAS_layer
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = CrossAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout)
        elif global_model_type == 'SparseTransformer':
                self.self_attn = SparseGraphAttention(
                    dim_h, num_heads, dropout=self.attn_dropout
                )
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
            
        # EXACT SAME KV models initialization as NAS_layer
        self._initialize_kv_models(dropout, act)


        

        # EXACT SAME normalization layers as NAS_layer
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

        # EXACT SAME Feed Forward block as NAS_layer
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def _initialize_kv_models(self, dropout, act):
        """
        EXACT SAME KV models initialization as NAS_layer
        """
        gnn_type=self.head_gnn_types[self.best_expert_idx]
        self.kv_gnn_with_edge_attr = True
        if gnn_type == 'None':
            self.kv_model = None
        elif gnn_type == "GCN":
            self.kv_gnn_with_edge_attr = False
            self.kv_model = pygnn.GCNConv(self.dim_h, self.dim_h)
        elif gnn_type == "SAGE":
            self.kv_gnn_with_edge_attr = False
            self.kv_model = pygnn.SAGEConv(self.dim_h, self.dim_h,'max')
        elif gnn_type == 'GIN':
            self.kv_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(self.dim_h, self.dim_h),
                                   self.activation(),
                                   Linear_pyg(self.dim_h, self.dim_h))
            self.kv_model= pygnn.GINConv(gin_nn)
        elif gnn_type == 'GENConv':
            self.kv_model = pygnn.GENConv(self.dim_h, self.dim_h)
        elif gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(self.dim_h, self.dim_h),
                                   self.activation(),
                                   Linear_pyg(self.dim_h, self.dim_h))
            if self.equivstable_pe:
                self.kv_model = GINEConvESLapPE(gin_nn)
            else:
                self.kv_model= pygnn.GINEConv(gin_nn)
        elif gnn_type == 'GAT':
            self.kv_model = pygnn.GATConv(in_channels=self.dim_h,
                                                 out_channels=self.dim_h // self.num_heads,
                                                 heads=self.num_heads,
                                                 edge_dim=self.dim_h)
        elif gnn_type == 'GATV2':
            self.kv_model= pygnn.GATv2Conv(in_channels=self.dim_h,
                                                 out_channels=self.dim_h // self.num_heads,
                                                 heads=self.num_heads,
                                                 edge_dim=self.dim_h)
        elif gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array([1, 2, 3, 4, 5, 10, 20, 50, 100, 200]))
            self.kv_model = pygnn.PNAConv(self.dim_h, self.dim_h,
                                                 aggregators=aggregators,
                                                 scalers=scalers,
                                                 deg=deg,
                                                 edge_dim=min(128, self.dim_h),
                                                 towers=1,
                                                 pre_layers=1,
                                                 post_layers=1,
                                                 divide_input=False)
        elif gnn_type == 'CustomGatedGCN':
            self.kv_model= GatedGCNLayer(self.dim_h, self.dim_h,
                                                 dropout=dropout,
                                                 residual=True,
                                                 act=act,
                                                 equivstable_pe=self.equivstable_pe)
        else:
            raise ValueError(f"Unsupported KV GNN model: {gnn_type}")

  

    def forward(self, batch):
        """
        Forward pass - IDENTICAL to NAS_layer but with fixed expert weights
        """
        h = batch.x
        h_in1 = h.clone()
        
        batch_size = len(torch.unique(batch.batch))
        
        # CHANGED: Use only best expert if available
        if self.best_expert_idx is not None:
            # SINGLE EXPERT MODE - Use only the selected best expert
            h_kv = h_in1.clone()
            gnn_type = self.head_gnn_types[self.best_expert_idx]
            
            
            if gnn_type == 'CustomGatedGCN':
                    es_data = None
                    if self.equivstable_pe:
                        es_data = batch.pe_EquivStableLapPE
                    kv_out = self.kv_model(Batch(batch=batch,
                                                          x=h_kv,
                                                          edge_index=batch.edge_index,
                                                          edge_attr=batch.edge_attr,
                                                          pe_EquivStableLapPE=es_data))
                    h_kv = kv_out.x
            else:
                    if self.kv_gnn_with_edge_attr:
                        if self.equivstable_pe:
                            h_kv = self.kv_model(h_kv,
                                                batch.edge_index,
                                                batch.edge_attr,
                                                batch.pe_EquivStableLapPE)
                        else:
                            h_kv = self.kv_model(h_kv,
                                                batch.edge_index,
                                                batch.edge_attr)
                    else:
                        h_kv = self.kv_model(h_kv, batch.edge_index)
                    h_kv = self.dropout_kv(h_kv)
                
            if self.layer_norm:
                    h_kv = self.norm1_kv(h_kv, batch.batch)
            if self.batch_norm:
                    h_kv = self.norm1_kv(h_kv)
            
            # No mixture - kv_combined is just the single expert output
            kv_combined = h_kv
            
        
        
        # EXACT SAME transformer processing continues...
        if self.self_attn is not None:
            # Convert to dense format
            q_dense, mask = to_dense_batch(h_in1, batch.batch)

            kv_combined_dense, _ = to_dense_batch(kv_combined, batch.batch)

                
            
            # Standard attention processing
            if self.global_model_type == 'Transformer':
                h_attn, A = self.self_attn(q_dense, kv_combined_dense, kv_combined_dense,
                                       key_padding_mask=~mask, need_weights=True,
                                       average_attn_weights=False)  # ADD THIS!
                if self.log_attn_weights:
                    # Now A will be [batch_size, num_heads, target_len, source_len]
                    self.attn_weights = A.detach().cpu()
                h_attn = h_attn[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(q_dense, context=kv_combined_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(q_dense, q_dense, attention_mask=mask)
            elif self.global_model_type == 'BiasedTransformer':
                h_attn = self.self_attn(q_dense, kv_combined_dense, kv_combined_dense, batch.attn_bias,
                                       key_padding_mask=~mask, need_weights=False)[0][mask]
            #elif self.global_model_type == 'SparseTransformer':
            #        # No need for dense conversion!
            #        # Q from projection, KV from GNN
            #        h_attn = self.self_attn(
            #            h_in1,  # Q 
            #            kv_combined,               # KV from GNN
            #            batch.edge_index          # Sparse attention pattern
            #        )
            elif self.global_model_type == 'SparseTransformer':
                h_attn = self.self_attn(
                    h_in1,
                    kv_combined,
                    batch.edge_index,
                    save_attn=self.log_attn_weights  # ADD THIS PARAMETER!
                )
                
                # REPLACE the dummy matrix code with this:
                if self.log_attn_weights and hasattr(self.self_attn, 'sparse_attn_weights'):
                    # Store the sparse attention for PK-Explainer
                    self.attn_weights = self.self_attn.sparse_attn_weights
                        
            
            
            # Skip connection
            h_attn = self.dropout_attn(h_attn) + h_in1
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            
           
        h = h_attn

        # EXACT SAME Feed Forward block as NAS_layer
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        
        batch.x = h
        return batch

    def _ff_block(self, x):
        """
        EXACT SAME Feed Forward block as NAS_layer
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'num_heads={self.num_heads}, ' \
            f'num_experts={self.num_experts}, ' \
            f'routing_mode={self.routing_mode}, ' \
            f'optimal_weights={self.optimal_weights}, ' \
            f'is_last_layer={self.is_last_layer}'
        return s
    
    
