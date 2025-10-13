# -*- coding: utf-8 -*-
"""
NAS Layer - GPS Layer with DARTS Neural Architecture Search
ARCHITECTURE: Same as MOE except router replaced with DARTS InputChoice selection

ROUTING MODES:
- 'uniform': Equal weights [0.25, 0.25, 0.25, 0.25], 2 epochs max
- 'random': Random weights, 2 epochs max  
- 'nas': Full DARTS search with InputChoice, 50+ epochs
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
from torch_geometric.nn import global_mean_pool, global_max_pool

# DARTS integration
import nni.retiarii.nn.pytorch as nn_nni
from nni.retiarii import model_wrapper

class NASLayer(nn.Module):
    """
    NAS Layer - GPS Layer enhanced with DARTS InputChoice selection
    IDENTICAL to MOE architecture except router replaced with DARTS selection
    """

    def __init__(self, dim_h,
             local_gnn_type, global_model_type, head_gnn_types, routing_mode, num_heads, act='relu',
             pna_degrees=None, equivstable_pe=False, dropout=0.0,
             attn_dropout=0.0, layer_norm=False, batch_norm=True,
             bigbird_cfg=None, log_attn_weights=False, uncertainty_enabled=True, is_last_layer=False,
             perturbation_delta=None, perturbation_epsilon=None, perturbation_max_steps=None, 
             uncertainty_samples=None, nas_config=None,epochs=10,layer_idx=1, group_id=None, group_layers=None,gnns_type_used=None):
        super().__init__()

        # EXACT SAME basic parameters as MOE
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
        self.layer_idx=layer_idx
        self.group_id = group_id if group_id is not None else layer_idx
        self.group_layers = group_layers if group_layers is not None else [layer_idx]
        self.expert_dropout = nn.Dropout(0.3)
        self.gnns_type_used=gnns_type_used
        
        # EXACT SAME uncertainty parameters as MOE
        self.perturbation_delta = perturbation_delta if perturbation_delta is not None else 0.02
        self.perturbation_epsilon = perturbation_epsilon if perturbation_epsilon is not None else 0.15
        self.perturbation_max_steps = perturbation_max_steps if perturbation_max_steps is not None else 10
        self.uncertainty_samples = uncertainty_samples if uncertainty_samples is not None else 5
        
        # NAS configuration and tracking (like ASPECT-GT)
        self.nas_config = nas_config
        self.training_mode = 'darts'  # 'darts' or 'discrete'
        self.optimal_weights = None   # Store optimal weights from DARTS
        
        # ASPECT-GT style tracking
        self.total_epochs = epochs
        self.current_epoch = 0
        self.tracking_started = False
        self.batch_counter = 0
        self.epoch_metrics = {'alphas': {}}
        self.cumulative_metrics = {'alphas': {}, 'selected_ops': {}}
        
        # Expert configuration - same as MOE
        self.num_experts = len(head_gnn_types)
        
        # EXACT SAME local GNN initialization as MOE
        self.kv_model = nn.ModuleDict()
        
        # DARTS InputChoice for NAS mode (replaces MOE router)
        self.input_choice = None
        if self.routing_mode == 'nas':
            self.input_choice = nn_nni.InputChoice(
                n_candidates=self.num_experts,
            key=f'gnn_mixture_choice_group_{self.group_id}'
            )
        
        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer', 'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )
     
        # EXACT SAME local message-passing model as MOE
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == "SAGE":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.SAGEConv(dim_h, dim_h,'max')
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
        elif local_gnn_type == 'GATV2':
            self.local_model = pygnn.GATv2Conv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h,
                                             residual=False)
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
        self.local_gnn_type = local_gnn_type
        
        # EXACT SAME global attention model as MOE
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            from performer_pytorch import CrossAttention
            '''
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)'''
            self.self_attn = CrossAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout)
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
            
        # EXACT SAME KV models as MOE - one per expert type
        self._initialize_kv_models(dropout, act)
        
        if self.self_attn is not None:
            # Q projection layer to align with expert K,V space
            self.q_projection = nn.Sequential(
                nn.Linear(dim_h, dim_h),
                self.activation(),
                nn.Linear(dim_h, dim_h)
            )

        # EXACT SAME normalization layers as MOE
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

        # EXACT SAME Feed Forward block as MOE
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
        EXACT SAME KV models initialization as MOE
        """
        for i, gnn_type in enumerate(self.head_gnn_types):
            self.kv_gnn_with_edge_attr = True
            if gnn_type == 'None':
                self.kv_model[str(i)] = None
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
                # Same PNA setup as local model
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

  

    def update_cumulative_metrics(self):
        """
        ASPECT-GT style cumulative metrics update
        """
        if self.epoch_metrics['alphas']:
            latest_epoch = max(self.epoch_metrics['alphas'].keys())
            for layer_key in self.epoch_metrics['alphas'][latest_epoch]:
                if layer_key not in self.cumulative_metrics['alphas']:
                    self.cumulative_metrics['alphas'][layer_key] = {}
                for op_name, alpha in self.epoch_metrics['alphas'][latest_epoch][layer_key].items():
                    self.cumulative_metrics['alphas'][layer_key][op_name] = alpha

    def select_best_operations(self):
        """
        ASPECT-GT style operation selection
        """
        self.cumulative_metrics['selected_ops'] = {}
        if self.gnns_type_used==1:
            op_name_map = {
                '0': 'GIN',
                '1': 'GCN', 
                '2': 'GAT',
                '3': 'SAGE',
    
            }
        elif self.gnns_type_used==2:
            op_name_map = {
                '0': 'GINE',
                '1': 'CustomGatedGCN', 
                '2': 'GATV2',

    
            }
    
        for layer_key in self.cumulative_metrics['alphas']:
            scores = self.cumulative_metrics['alphas'][layer_key]
            all_ops = list(scores.items())
            
            ops_data = []
            for op_idx, score in all_ops:
                op_readable_name = op_name_map.get(op_idx, f"unknown-{op_idx}")
                ops_data.append((op_idx, score, op_readable_name))
                
            self.cumulative_metrics['selected_ops'][layer_key] = ops_data
            
            #print(f"{layer_key}: Operations with scores: ")
            #for op_idx, score, op_name in ops_data:
            #    print(f"  {op_idx} ({op_name}) with score {score:.4f}")

    def get_metrics_dict(self):
        """
        ASPECT-GT style metrics dictionary
        """
        self.update_cumulative_metrics()
        self.select_best_operations()
        
        return {
            'per_epoch_metrics': {
                'alphas': self.epoch_metrics['alphas'],
            },
            'cumulative_metrics': {
                'alphas': self.cumulative_metrics['alphas'],
                'selected_operations': self.cumulative_metrics['selected_ops'],
            }
        }

    

    

    def forward(self, batch):
        """
        Forward pass - SAME as MOE except routing replaced with DARTS InputChoice
        """
        h = batch.x
        h_in1 = h.clone()
        h_out_list = []
        
        # EXACT SAME local MPNN processing as MOE
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
                h_local = h_in1 + h_local  # Skip connection
            
            # EXACT SAME layer normalization as MOE
            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)


      

        batch_size = len(torch.unique(batch.batch))
        
        # ROUTING WEIGHTS - Only for DARTS phase
        if self.routing_mode == 'uniform':
            expert_weights = torch.ones(self.num_experts, device=h.device) / self.num_experts
        elif self.routing_mode == 'random':
            random_weights = torch.rand(self.num_experts, device=h.device)
            expert_weights = F.softmax(random_weights, dim=0)
        elif self.routing_mode == 'nas':
            if self.input_choice is not None:
                expert_weights = F.softmax(self.input_choice.alpha, -1)
                

        # Populate metrics for all layers in this group
        if self.current_epoch not in self.epoch_metrics['alphas']:
            self.epoch_metrics['alphas'][self.current_epoch] = {}
        
        for layer_id in self.group_layers:
            layer_key = f'layer_{layer_id}'
            if layer_key not in self.epoch_metrics['alphas'][self.current_epoch]:
                self.epoch_metrics['alphas'][self.current_epoch][layer_key] = {}
            
            for i, alpha in enumerate(expert_weights):
                self.epoch_metrics['alphas'][self.current_epoch][layer_key][str(i)] = alpha.item()
        
        # EXACT SAME expert processing as MOE
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

        # EXACT SAME expert normalization as MOE
        all_expert_outputs = torch.cat(expert_outputs, dim=0)
        global_mean = all_expert_outputs.mean()
        global_std = all_expert_outputs.std()
        
        # Apply SAME normalization to all experts
        expert_outputs = [
            (exp_out - global_mean) / (global_std + 1e-8) 
            for exp_out in expert_outputs
        ]
       
        expert_outputs = [self.expert_dropout(exp_out) for exp_out in expert_outputs]
        # KEY DIFFERENCE: Use InputChoice for DARTS, weighted sum for others
        if self.routing_mode == 'nas' and self.training_mode == 'darts' and self.input_choice is not None:
            # DARTS phase: Use InputChoice to learn optimal combination
            kv_combined = self.input_choice(expert_outputs)
        '''    
        else:
            # Non-DARTS: Use weighted combination
            kv_combined = torch.zeros_like(expert_outputs[0])
            for i, expert_out in enumerate(expert_outputs):
                weight = expert_weights[i]  # Use first batch's weights
                kv_combined += weight * expert_out
        '''
        # EXACT SAME transformer processing as MOE
        if self.self_attn is not None:
            # Convert to dense format
            q_dense, mask = to_dense_batch(h_in1, batch.batch)
            kv_combined = kv_combined + 0.1 * h_in1  
            kv_combined_dense, _ = to_dense_batch(kv_combined, batch.batch)
            # NEW: Project Q to align with expert K,V space
            q_dense = self.q_projection(q_dense)

            # Standard attention processing
            if self.global_model_type == 'Transformer':
                h_attn = self.self_attn(q_dense, kv_combined_dense, kv_combined_dense,
                                       key_padding_mask=~mask, need_weights=False)[0][mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(q_dense, context=kv_combined_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(q_dense, kv_combined_dense, attention_mask=mask)
            elif self.global_model_type == 'BiasedTransformer':
                h_attn = self.self_attn(q_dense, kv_combined_dense, kv_combined_dense, batch.attn_bias,
                                       key_padding_mask=~mask, need_weights=False)[0][mask]
            
            # Skip connection
            h_attn = self.dropout_attn(h_attn) + h_in1
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            
            h_out_list.append(h_attn)

        # EXACT SAME final processing as MOE
        h = sum(h_out_list)
        #h=h_attn
        # EXACT SAME Feed Forward block as MOE
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        
        batch.x = h
        return batch
    
    def set_training_mode(self, mode):
        """Set training mode"""
        self.training_mode = mode
   
    def _ff_block(self, x):
        """
        EXACT SAME Feed Forward block as MOE
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'num_heads={self.num_heads}, ' \
            f'num_experts={self.num_experts}, ' \
            f'routing_mode={self.routing_mode}, ' \
            f'is_last_layer={self.is_last_layer}'
        return s
