import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
import torch.nn as nn
from dartsgt.layer.moe_layer import MOELayer  # Use the optimized MOELayer
import torch_geometric.nn as pygnn
from dartsgt.layer.graph_structural_features import compute_graph_structural_features

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

@register_network('MOEModel')
class MOEModel(torch.nn.Module):
    """Optimized Sequential message passing + attention transformer model."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        
        layers = []
        uncertainty_enabled = cfg.gt.uncertainty.enabled
        num_layers = cfg.gt.layers
        is_last_layer=False
        for layer_idx in range(num_layers):
            # Only the last layer should compute uncertainty
            #is_last_layer = (layer_idx == num_layers - 1)
            if layer_idx == (num_layers-1):
                is_last_layer=True
            layers.append(MOELayer(
                    dim_h=cfg.gt.dim_hidden,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_model_type,
                    head_gnn_types=cfg.gt.head_gnn_types,
                    routing_mode=cfg.gt.routing_mode,
                    num_heads=cfg.gt.n_heads,
                    act=cfg.gnn.act,
                    pna_degrees=cfg.gt.pna_degrees,
                    equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    bigbird_cfg=cfg.gt.bigbird,
                    log_attn_weights=cfg.train.mode == 'log-attn-weights',
                    uncertainty_enabled=uncertainty_enabled,
                    is_last_layer=is_last_layer,
                    perturbation_delta=getattr(cfg.gt.uncertainty, 'delta', 0.02),
                    perturbation_epsilon=getattr(cfg.gt.uncertainty, 'epsilon', 0.15), 
                    perturbation_max_steps=getattr(cfg.gt.uncertainty, 'max_steps', 10),
                    uncertainty_samples=getattr(cfg.gt.uncertainty, 'samples', 5),
                ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    #def forward(self, batch):
    #    for module in self.children():
    #        batch = module(batch)
    #    return batch
    
    def forward(self, batch):
        # First, encode features
        batch = self.encoder(batch)
        
        # NEW: Compute structural features once for the entire model
        # These are graph-level properties that don't change between layers
        with torch.no_grad():  # No gradients needed for structural features
            batch.graph_structural_features = compute_graph_structural_features(batch)
        
        # Continue with the rest of the forward pass
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        
        batch = self.layers(batch)
        batch = self.post_mp(batch)
        
        return batch
