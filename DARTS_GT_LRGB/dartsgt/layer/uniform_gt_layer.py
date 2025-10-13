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
from torch_geometric.graphgym.config import cfg
from torch_scatter import scatter_add
from dartsgt.layer.bigbird_layer import SingleBigBirdLayer
from dartsgt.layer.gatedgcn_layer import GatedGCNLayer
from dartsgt.layer.gine_conv_layer import GINEConvESLapPE
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.utils.num_nodes import maybe_num_nodes
from performer_pytorch import CrossAttention
import torch.nn.functional as F
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
        
    def forward(self, q, kv, edge_index):
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
        
        
class UNIFORMGTLayer(nn.Module):
    """Sequential MPNN + global attention transformer layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False,attn_gnn='CustomGatedGCN'):
        super().__init__()

        # Same initialization as GPSLayer
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.attn_gnn=attn_gnn

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        
        # Local message-passing model.
        self.kv_gnn_with_edge_attr = True
        if attn_gnn == 'None':
            self.kv_model = None

        # MPNNs without edge attributes support.
        elif attn_gnn == "GCN":
            self.kv_gnn_with_edge_attr = False
            self.kv_model = pygnn.GCNConv(dim_h, dim_h)  # Second instance for K/V
        elif attn_gnn == "SAGE":
            self.kv_gnn_with_edge_attr = False
            self.kv_model = pygnn.SAGEConv(self.dim_h, self.dim_h,'max')
        elif attn_gnn == 'GIN':
            self.kv_gnn_with_edge_attr = False
            # Second instance for K/V path
            gin_nn_kv = nn.Sequential(Linear_pyg(dim_h, dim_h),
                            self.activation(),
                            Linear_pyg(dim_h, dim_h))
            self.kv_model = pygnn.GINConv(gin_nn_kv)

        # MPNNs supporting also edge attributes.
        elif attn_gnn == 'GENConv':
            self.kv_model = pygnn.GENConv(dim_h, dim_h)
        elif attn_gnn == 'GINE':

            # Second instance for K/V path
            gin_nn_kv = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                 self.activation(),
                                 Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:
                self.kv_model = GINEConvESLapPE(gin_nn_kv)
            else:
                self.kv_model = pygnn.GINEConv(gin_nn_kv)
            
        elif attn_gnn == 'GAT':

            self.kv_model = pygnn.GATConv(in_channels=dim_h,
                                     out_channels=dim_h // num_heads,
                                     heads=num_heads,
                                     edge_dim=dim_h)
        elif attn_gnn == 'GATV2':
            self.kv_model = pygnn.GATv2Conv(in_channels=self.dim_h,
                                                 out_channels=self.dim_h // self.num_heads,
                                                 heads=self.num_heads,
                                                 edge_dim=self.dim_h)
            
        elif attn_gnn == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            
            self.kv_model = pygnn.PNAConv(dim_h, dim_h,
                                    aggregators=aggregators,
                                    scalers=scalers,
                                    deg=deg,
                                    edge_dim=min(128, dim_h),
                                    towers=1,
                                    pre_layers=1,
                                    post_layers=1,
                                    divide_input=False)
            
        elif attn_gnn == 'CustomGatedGCN':
            
            self.kv_model = GatedGCNLayer(dim_h, dim_h,
                                    dropout=dropout,
                                    residual=True,
                                    act=act,
                                    equivstable_pe=equivstable_pe)
            
        else:
            raise ValueError(f"Unsupported kv GNN model: {attn_gnn}")
        self.attn_gnn = attn_gnn

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            #self.self_attn = QueryOnlyMultiheadAttention(dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = CrossAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif global_model_type == 'SparseTransformer':
                self.self_attn = SparseGraphAttention(
                    dim_h, num_heads, dropout=self.attn_dropout
                )
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

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

    def forward(self, batch):
        h = batch.x
        h_in1 = h.clone()  # Store original features for queries
        #h_out_list=[]
       
        
        # 2. Second GNN application for attention K/V (using original features)
        h2 = h_in1.clone()  # Start with original features for K/V path
        if self.kv_model is not None:  # Use kv_model here instead of local_model
            self.kv_model: pygnn.conv.MessagePassing  # Typing hint
            if self.attn_gnn == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                kv_out = self.kv_model(Batch(batch=batch,
                                              x=h2,
                                              edge_index=batch.edge_index,
                                              edge_attr=batch.edge_attr,
                                              pe_EquivStableLapPE=es_data))
                h2 = kv_out.x
            else:
                if self.kv_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h2 = self.kv_model(h2,
                                        batch.edge_index,
                                        batch.edge_attr,
                                        batch.pe_EquivStableLapPE)
                    else:
                        h2 = self.kv_model(h2,
                                        batch.edge_index,
                                        batch.edge_attr)
                else:
                    h2 = self.kv_model(h2, batch.edge_index)
                h2 = self.dropout_kv(h2)
            
            # Apply normalization to GNN outputs for K/V
            if self.layer_norm:
                h2 = self.norm1_kv(h2, batch.batch)
            if self.batch_norm:
                h2 = self.norm1_kv(h2)
            
        # 3. Multi-head attention using original features as Q, second GNN output as K/V
        if self.self_attn is not None:
            # Convert both original and second GNN-processed features to dense format
            q_dense, mask = to_dense_batch(h_in1, batch.batch)  # Queries from original features
            #h2 = h2 + 0.1 * h_in1  
            kv_dense, _ = to_dense_batch(h2, batch.batch)  # Keys/Values from second GNN application

            #if self.training:
            #    kv_dense = F.dropout(kv_dense, p=0.2, training=True)
            #q_dense = self.q_projection(q_dense) 
            
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
            elif self.global_model_type == 'SparseTransformer':
                    # No need for dense conversion!
                    # Q from projection, KV from GNN
                    h_attn = self.self_attn(
                        h_in1,  # Q 
                        h2,               # KV from GNN
                        batch.edge_index          # Sparse attention pattern
                    )
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(q_dense, context=kv_dense, mask=mask)[mask]
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
