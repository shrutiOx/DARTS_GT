import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max
import torch.nn.functional as F
from dartsgt.layer.bigbird_layer import SingleBigBirdLayer
from dartsgt.layer.gatedgcn_layer import GatedGCNLayer
from dartsgt.layer.gine_conv_layer import GINEConvESLapPE
from performer_pytorch import CrossAttention
from torch_geometric.graphgym.config import cfg


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

class VANLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False,layer_idx=0):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        #self.log_attn_weights = log_attn_weights
        self.log_attn_weights = log_attn_weights or (hasattr(cfg.gt, 'pk_explainer') and cfg.gt.pk_explainer.enabled)
        if self.log_attn_weights and global_model_type not in ['Transformer',
                                                       'BiasedTransformer', 
                                                       'SparseTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        

        # Global attention transformer-style model.
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
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            #self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
          
        if self.batch_norm:
            #self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        #self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            elif self.global_model_type == 'SparseTransformer':
                h_attn = self.self_attn(
                    h_in1,  # Q 
                    h_in1,  # KV 
                    batch.edge_index,  # Sparse attention pattern
                    save_attn=self.log_attn_weights  # ADD THIS!
                )
                
                if self.log_attn_weights and hasattr(self.self_attn, 'sparse_attn_weights'):
                    # Store the sparse attention for PK-Explainer
                    self.attn_weights = self.self_attn.sparse_attn_weights
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h=h_attn
            

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x, A = self.self_attn(x, x, x,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=True,
                             average_attn_weights=False)
        
        # SIMPLE ADDITION - Store attention if collector is present
        if hasattr(self, 'attention_collector') and self.attention_collector is not None:
            self.attention_collector.add_attention(
                layer_idx=self.layer_idx,
                attention=A
            )
        
        if self.log_attn_weights:
            self.attn_weights = A.detach().cpu()
        return x

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
