# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 04:28:13 2025

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 17 16:12:51 2025

@author: ADMIN
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


        
class UNICONDRANDLayer(nn.Module):
    """Sequential MPNN + global attention transformer layer.
    """

    def __init__(self, dim_h, global_model_type, num_heads, act='relu',head_gnn_types=[],
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False,attn_gnn='CustomGatedGCN',routing_mode=None):
        super().__init__()

        # Same initialization as GPSLayer
        self.dim_h = dim_h
        self.routing_mode=routing_mode
        self.head_gnn_types=head_gnn_types
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.expert_dropout = nn.Dropout(0.2)#0.3
        self.attn_gnn=attn_gnn
        # Expert configuration - number of experts equals number of GNN types
        self.num_experts = len(head_gnn_types)
        # Initialize modules
        self.kv_model = nn.ModuleDict()
        self.flag=0

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        
        
        self._initialize_kv_models(dropout, act)
        
        

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            #self.self_attn = QueryOnlyMultiheadAttention(dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            from performer_pytorch import CrossAttention
            self.self_attn = CrossAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")
            
        if self.self_attn is not None:
            # Q projection layer to align with expert K,V space
            self.q_projection = nn.Sequential(
                nn.Linear(dim_h, dim_h),
                self.activation(),
                nn.Linear(dim_h, dim_h)
            )
        

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_kv = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_kv = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_kv = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
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
        for i, gnn_type in enumerate(self.head_gnn_types):
            self.kv_gnn_with_edge_attr = True
            if gnn_type == 'Nonetyp1':
                self.kv_model[str(i)] = None
                self.kv_gnn_with_edge_attr = False
            elif gnn_type == 'Nonetyp2':
                self.kv_model[str(i)] = None
                self.kv_gnn_with_edge_attr = True
            elif gnn_type == "GCN":
                self.kv_gnn_with_edge_attr = False
                self.kv_model[str(i)] = pygnn.GCNConv(self.dim_h, self.dim_h)
            elif gnn_type == "SAGE":
                self.kv_gnn_with_edge_attr = False
                self.kv_model[str(i)] = pygnn.SAGEConv(self.dim_h, self.dim_h,'max')
            elif gnn_type == 'GIN':
                self.kv_gnn_with_edge_attr = False
                gin_nn = nn.Sequential(Linear_pyg(self.dim_h, self.dim_h),
                                       self.activation(),
                                       Linear_pyg(self.dim_h, self.dim_h))
                self.kv_model[str(i)] = pygnn.GINConv(gin_nn)
            elif gnn_type == 'GENConv':
                self.kv_model[str(i)] = pygnn.GENConv(self.dim_h, self.dim_h)
            elif gnn_type == 'GINE':
                gin_nn = nn.Sequential(Linear_pyg(self.dim_h, self.dim_h),
                                       self.activation(),
                                       Linear_pyg(self.dim_h, self.dim_h))
                if self.equivstable_pe:
                    self.kv_model[str(i)] = GINEConvESLapPE(gin_nn)
                else:
                    self.kv_model[str(i)] = pygnn.GINEConv(gin_nn)
            elif gnn_type == 'GAT':
                self.kv_model[str(i)] = pygnn.GATConv(in_channels=self.dim_h,
                                                     out_channels=self.dim_h // self.num_heads,
                                                     heads=self.num_heads,
                                                     edge_dim=self.dim_h)
            elif gnn_type == 'GATV2':
                self.kv_model[str(i)] = pygnn.GATv2Conv(in_channels=self.dim_h,
                                                     out_channels=self.dim_h // self.num_heads,
                                                     heads=self.num_heads,
                                                     edge_dim=self.dim_h)
            elif gnn_type == 'PNA':
                aggregators = ['mean', 'max', 'sum']
                scalers = ['identity']
                deg = torch.from_numpy(np.array([1, 2, 3, 4, 5, 10, 20, 50, 100, 200]))
                self.kv_model[str(i)] = pygnn.PNAConv(self.dim_h, self.dim_h,
                                                     aggregators=aggregators,
                                                     scalers=scalers,
                                                     deg=deg,
                                                     edge_dim=min(128, self.dim_h),
                                                     towers=1,
                                                     pre_layers=1,
                                                     post_layers=1,
                                                     divide_input=False)
            elif gnn_type == 'CustomGatedGCN':
                self.kv_model[str(i)] = GatedGCNLayer(self.dim_h, self.dim_h,
                                                     dropout=dropout,
                                                     residual=True,
                                                     act=act,
                                                     equivstable_pe=self.equivstable_pe)
            else:
                raise ValueError(f"Unsupported KV GNN model: {gnn_type}")
                
                
    def forward(self, batch):
        h = batch.x
        h_in1 = h.clone()  # Store original features for queries
        
        
        # 2. Second GNN application for attention K/V (using original features)
        h2 = h_in1.clone()  # Start with original features for K/V path
        
        
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
            
        # EXACT SAME expert normalization as NAS_layer
        all_expert_outputs = torch.cat(expert_outputs, dim=0)
        global_mean = all_expert_outputs.mean()
        global_std = all_expert_outputs.std()
        
        # Apply SAME normalization to all experts
        expert_outputs = [
            (exp_out - global_mean) / (global_std + 1e-8) 
            for exp_out in expert_outputs
        ]
        expert_outputs = [self.expert_dropout(exp_out) for exp_out in expert_outputs]
        
        if self.routing_mode == 'uniform':
            # Equal weights for all experts
            expert_routing_weights = torch.ones( self.num_experts, device=h.device) / self.num_experts
            
        elif self.routing_mode == 'random':
            # Random weights for experts
            random_weights = torch.rand(self.num_experts, device=h.device)
            expert_routing_weights = F.softmax(random_weights, dim=0)
            
        if self.flag==0:
            print('expert_routing_weights ',expert_routing_weights)    
            self.flag +=1
        kv_dense = torch.zeros_like(expert_outputs[0])
        weighted_experts = [weight * expert for weight, expert in zip(expert_routing_weights, expert_outputs)]
        kv_dense = torch.stack(weighted_experts, dim=0).sum(dim=0)
        

            
        # 3. Multi-head attention using original features as Q, second GNN output as K/V
        if self.self_attn is not None:
            # Convert to dense format
            q_dense, mask = to_dense_batch(h_in1, batch.batch)
            kv_dense = kv_dense + 0.1 * h_in1  
            kv_dense, _ = to_dense_batch(kv_dense, batch.batch)
            if self.training:
                kv_dense = F.dropout(kv_dense, p=0.2, training=True)
            q_dense = self.q_projection(q_dense) 
                
            
            # Apply attention directly (no _sa_block)
            if self.global_model_type == 'Transformer':
                if not self.log_attn_weights:
                    h_attn = self.self_attn(q_dense, kv_dense, kv_dense,
                                         key_padding_mask=~mask,
                                         need_weights=False)[0][mask]
                else:
                    h_attn, A = self.self_attn(q_dense, kv_dense, kv_dense,
                                            key_padding_mask=~mask,
                                            need_weights=True,
                                            average_attn_weights=False)
                    h_attn = h_attn[mask]
                    self.attn_weights = A.detach().cpu()
            elif self.global_model_type == 'Performer':
                # Need custom handling for Performer
                h_attn = self.self_attn(q_dense, kv_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(q_dense, kv_dense, attention_mask=mask)
            elif self.global_model_type == 'BiasedTransformer':
                if not self.log_attn_weights:
                    h_attn = self.self_attn(q_dense, kv_dense, kv_dense,batch.attn_bias,
                                         key_padding_mask=~mask,
                                         need_weights=False)[0][mask]
                else:
                    h_attn, A = self.self_attn(q_dense, kv_dense, kv_dense,batch.attn_bias,
                                            key_padding_mask=~mask,
                                            need_weights=True,
                                            average_attn_weights=False)
                    h_attn = h_attn[mask]
                    self.attn_weights = A.detach().cpu()
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")
            
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection with input
            
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn , batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn )
            #h_out_list.append(h_attn)
                
        h = h_attn #parallel connection with GNN output
        # 4. Feed Forward block (no changes)
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        
        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
