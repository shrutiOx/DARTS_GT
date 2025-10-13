# -*- coding: utf-8 -*-
"""
Created on Sun May 25 01:33:18 2025

@author: ADMIN
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool ,global_max_pool
from torch_geometric.utils import degree, scatter


def compute_graph_structural_features(batch):
    device = batch.x.device
    num_nodes = batch.x.size(0)
    node_graph_assignment = batch.batch
    
    # 1. Average degree
    node_degrees = degree(batch.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    avg_degree = scatter(node_degrees, node_graph_assignment, dim=0, reduce='mean')
    
    # 2. Edge density
    nodes_per_graph = scatter(torch.ones_like(node_graph_assignment, dtype=torch.float), 
                              node_graph_assignment, dim=0, reduce='sum')
    edges_per_graph = scatter(torch.ones(batch.edge_index.size(1), device=device), 
                              node_graph_assignment[batch.edge_index[0]], dim=0, reduce='sum') / 2
    max_possible_edges = nodes_per_graph * (nodes_per_graph - 1) / 2
    edge_density = edges_per_graph / (max_possible_edges + 1e-6)
    
    # 3. Degree heterogeneity 
    degree_squared = node_degrees ** 2
    avg_degree_squared = scatter(degree_squared, node_graph_assignment, dim=0, reduce='mean')
    degree_variance = avg_degree_squared - avg_degree ** 2
    degree_heterogeneity = degree_variance / (avg_degree + 1e-6)
    
    # 4. Feature smoothness
    edge_src, edge_dst = batch.edge_index
    feature_diff = torch.norm(batch.x[edge_src] - batch.x[edge_dst], p=2, dim=1)
    avg_feature_smoothness = scatter(feature_diff, node_graph_assignment[edge_src], 
                                     dim=0, reduce='mean')
    
    structural_features = torch.stack([
        avg_degree, edge_density, degree_heterogeneity, avg_feature_smoothness
    ], dim=1)
    
    return F.normalize(structural_features, p=2, dim=1)

def compute_lean_moment_pooling(node_features, batch_assignment):
    mean_pool = global_mean_pool(node_features, batch_assignment)
    max_pool = global_max_pool(node_features, batch_assignment)
    return torch.cat([mean_pool, max_pool], dim=1)


def compute_moment_pooling(node_features, batch_assignment):
    """
    Compute statistical moments of node features for each graph.
    
    Args:
        node_features: Node feature tensor [num_nodes, dim]
        batch_assignment: Batch assignment for each node [num_nodes]
        
    Returns:
        torch.Tensor: Moment features [num_graphs, dim * 3]
    """
    # Mean pooling
    mean_pool = global_mean_pool(node_features, batch_assignment)
    
    # Variance pooling
    # Var(X) = E[X^2] - E[X]^2
    squared_features = node_features ** 2
    mean_squared = global_mean_pool(squared_features, batch_assignment)
    variance_pool = mean_squared - mean_pool ** 2
    
    # Skewness pooling (third moment)
    # Skew(X) = E[(X - μ)^3] / σ^3
    # Simplified: we'll use unnormalized third moment
    expanded_mean = mean_pool[batch_assignment]
    centered = node_features - expanded_mean
    cubed_centered = centered ** 3
    skewness_pool = global_mean_pool(cubed_centered, batch_assignment)
    
    # Concatenate all moments
    moment_features = torch.cat([mean_pool, variance_pool, skewness_pool], dim=1)
    
    return moment_features


