# -*- coding: utf-8 -*-
"""
Optimized Main GPS Script
- Uncertainty only for last layer + test/val
- Router weights logging  
- JSON export of test results
"""
 
import os
import sys
import logging

# Change to your script directory


os.chdir("D:/MY_FOLDER/DARTS_GT/DARTS_GT_NonLRGB") ###Put your script directory here


print(f"Changed working directory to: {os.getcwd()}")





import dartsgt  # noqa, register custom modules
import torch
import numpy as np
import time

import datetime
import random

import dartsgt  # noqa, register custom modules
from dartsgt.agg_runs import agg_runs
from dartsgt.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from dartsgt.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from dartsgt.logger import create_logger
from yacs.config import CfgNode as CN
from dartsgt.network.gps_model import GPSModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    """Main function to train and evaluate the optimized GPS model."""
    # Get command line arguments
    args = parse_args()
    
    # Fix: Set the full path to the config file
    args.cfg_file = os.path.join(os.getcwd(), "confignas_sparse_molhiv.yaml")  # Use YOUR config file here

    print(f"Loading config from: {args.cfg_file}")
    
    # Check if file exists
    if not os.path.exists(args.cfg_file):
        raise FileNotFoundError(f"Config file not found at {args.cfg_file}")
    
    # Initialize the configuration
    set_cfg(cfg)
    
    # IMPORTANT: Register custom GPS config keys before loading config
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === ADD CUSTOM CONFIG KEYS FOR OPTIMIZED MOE ===
    from yacs.config import CfgNode as CN
    
  
    
    
    
    
    
    
    
    
    
    
        
    #if not hasattr(cfg, 'train_mode'):
    #    cfg.train_mode = 'nas_uncertainty_train'  # Default value
    
    
        
    if not hasattr(cfg.gt, 'routing_mode'):
        cfg.gt.routing_mode = 'uniform'  # Default value
        
    if not hasattr(cfg.gt, 'edge_startup'):
        cfg.gt.edge_startup = 0.1  # Default value
        
    if not hasattr(cfg.gt, 'residual_mult'):
        cfg.gt.residual_mult = 0.01  # Default value
        
    if not hasattr(cfg.gt, 'perturbation_strength'):
        cfg.gt.perturbation_strength = 10.0  # Default value
    
    if not hasattr(cfg.gt, 'head_gnn_types'):
        cfg.gt.head_gnn_types = ['GINE','CustomGatedGCN','GAT','PNA']  # Default value
    
    # ADD THIS: Uncertainty under gt section (optimized)
    if not hasattr(cfg.gt, 'uncertainty'):
        cfg.gt.uncertainty = CN()
        cfg.gt.uncertainty.enabled = True
        
    if not hasattr(cfg.gt, 'attn_gnn_uni'):
        cfg.gt.attn_gnn_uni = 'CustomGatedGCN'  # Default value
        
    if not hasattr(cfg.dataset, 'is_binary'):
        cfg.dataset.is_binary = 'binary'  # Default value   
        
    if not hasattr(cfg.gt, 'teacher_student_training'):
        cfg.gt.teacher_student_training = CN()
        cfg.gt.teacher_student_training.enabled = True
        
    if not hasattr(cfg.gt.teacher_student_training, 'teacher_instruct_epochs'):
        cfg.gt.teacher_student_training.teacher_instruct_epochs = 25  # Default value
        
    if not hasattr(cfg.gt.teacher_student_training, 'teacher_lr_multiplier_frozen'):
        cfg.gt.teacher_student_training.teacher_lr_multiplier_frozen = 0.0  # Default value
        
    if not hasattr(cfg.gt.teacher_student_training, 'teacher_lr_multiplier_finetune'):
        cfg.gt.teacher_student_training.teacher_lr_multiplier_finetune = 0.1  # Default value
        
        
    if not hasattr(cfg.gt.teacher_student_training, 'student_q_lr_multiplier'):
        cfg.gt.teacher_student_training.student_q_lr_multiplier = 5.0  # Default value
        
    if not hasattr(cfg.optim, 'swa'):
       cfg.optim.swa = CN()
       cfg.optim.swa.enabled = True
     
    if not hasattr(cfg, 'infer'):
        cfg.infer = CN()
        
    if not hasattr(cfg.optim.swa, 'start_epoch'):
        cfg.optim.swa.start_epoch = 70  # Default value
        
        
    if not hasattr(cfg.optim.swa, 'lr'):
        cfg.optim.swa.lr = 1.0e-4   # Default value
        
        
    if not hasattr(cfg.optim.swa, 'update_epochs'):
        cfg.optim.swa.update_epochs = 1   # Default value
     
        
    
    if not hasattr(cfg.gt.uncertainty, 'delta'):
        cfg.gt.uncertainty.delta = 0.02  # Default value
        
    if not hasattr(cfg.gt.uncertainty, 'epsilon'):
        cfg.gt.uncertainty.epsilon = 0.15 # Default value
        
    if not hasattr(cfg.gt.uncertainty, 'max_steps'):
        cfg.gt.uncertainty.max_steps = 10  # Default value
        
    if not hasattr(cfg.gt.uncertainty, 'samples'):
        cfg.gt.uncertainty.samples = 5  # Default value
    
    if not hasattr(cfg.gt, 'weight_fix'):
        cfg.gt.weight_fix = 'half:half'  # Default value
        
    if not hasattr(cfg.gt, 'gnns_type_used'):
        cfg.gt.gnns_type_used = 1  # Default value
        
    if not hasattr(cfg.gt, 'weight_type'):
        cfg.gt.weight_type = 'nas'  # Default value
        
    if not hasattr(cfg.gt, 'nas'):
        cfg.gt.nas = CN()
        cfg.gt.nas.enabled = True
        
    if not hasattr(cfg.gt.nas, 'darts_epochs'):
        cfg.gt.nas.darts_epochs = 50  # Default value
        
    if not hasattr(cfg.gt.nas, 'darts_split_ratio'):
        cfg.gt.nas.darts_split_ratio = 0.6  # Default value
        
    if not hasattr(cfg.gt.nas, 'stabilization_epochs'):
        cfg.gt.nas.stabilization_epochs = 0  # Default value
        
        
    if not hasattr(cfg.gt.nas, 'arc_learning_rate'):
        cfg.gt.nas.arc_learning_rate = 3.0e-4  # Default value
        
    if not hasattr(cfg.gt.nas, 'grad_clip'):
        cfg.gt.nas.grad_clip = 5.0  # Default value
        
    if not hasattr(cfg.gt.nas, 'unrolled'):
        cfg.gt.nas.unrolled = False  # Default value
        
    if not hasattr(cfg.gt.nas, 'darts_lr_schedule'):
        cfg.gt.nas.darts_lr_schedule = CN()

        
    if not hasattr(cfg.gt.nas.darts_lr_schedule, 'lr_reduce_factor'):
        cfg.gt.nas.darts_lr_schedule.lr_reduce_factor = 0.5  # Default value
        
    if not hasattr(cfg.gt.nas.darts_lr_schedule, 'lr_schedule_patience'):
        cfg.gt.nas.darts_lr_schedule.lr_schedule_patience = 10  # Default value
        
    if not hasattr(cfg.gt.nas.darts_lr_schedule, 'min_lr'):
        cfg.gt.nas.darts_lr_schedule.min_lr = 1.0e-6 # Default value
        
    if not hasattr(cfg.gt.nas.darts_lr_schedule, 'init_lr'):
        cfg.gt.nas.darts_lr_schedule.init_lr = 0.0025 # Default value
        
    if not hasattr(cfg.gt.nas.darts_lr_schedule, 'weight_decay'):
        cfg.gt.nas.darts_lr_schedule.weight_decay = 1e-5 # Default value
        
    if not hasattr(cfg.gt, 'pk_explainer'):
        cfg.gt.pk_explainer = CN()
        cfg.gt.pk_explainer.enabled = True
     
    if not hasattr(cfg.gt.pk_explainer, 'sample_ratio'):
        cfg.gt.pk_explainer.sample_ratio = 0.5  # Default value
        
    if not hasattr(cfg.gt.pk_explainer, 'graph_ids'):
        cfg.gt.pk_explainer.graph_ids = [1,2,100]  # Default value
        
    if not hasattr(cfg.gt.pk_explainer, 'k_heads'):
        cfg.gt.pk_explainer.k_heads = 1  # Default value
        
    if not hasattr(cfg.gt.pk_explainer, 'hypotheses'):
        cfg.gt.pk_explainer.hypotheses = ['specialization', 'foundation'] # Default value
        
    if not hasattr(cfg.gt.pk_explainer, 'visualization'):
        cfg.gt.pk_explainer.visualization = False  # Default value
        
    if not hasattr(cfg.gt.pk_explainer, 'save_attention'):
        cfg.gt.pk_explainer.save_attention = False  # Default value
    
    load_cfg(cfg, args)
    # ReduceOnPlateau scheduler compatibility - add to cfg.optim since that's what gets passed
    cfg.optim.train_mode = cfg.train.mode
    cfg.optim.eval_period = cfg.train.eval_period
    cfg.optim.batch_size = cfg.train.batch_size
    cfg.optim.ckpt_period = cfg.train.ckpt_period
    cfg.optim.epoch_resume = cfg.train.epoch_resume
    cfg.optim.auto_resume = cfg.train.auto_resume
    cfg.optim.ckpt_clean = cfg.train.ckpt_clean
    cfg.optim.iter_per_epoch = cfg.train.iter_per_epoch
    cfg.optim.sampler = cfg.train.sampler
    cfg.optim.neighbor_sizes = cfg.train.neighbor_sizes
    cfg.optim.walk_length = cfg.train.walk_length
    cfg.optim.node_per_graph = cfg.train.node_per_graph
    cfg.optim.radius = cfg.train.radius
    cfg.optim.sample_node = cfg.train.sample_node
    
    # Dataset attributes that might be needed
    cfg.optim.task = cfg.dataset.task
    cfg.optim.task_type = cfg.dataset.task_type
    cfg.optim.format = cfg.dataset.format
    cfg.optim.name = cfg.dataset.name
    
    # Update the accelerator to a proper value AFTER loading config
    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {cfg.accelerator}")
    
    # Set random seeds
    set_seed(cfg.seed)
    
    # Create output directory
    os.makedirs(cfg.run_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(cfg.run_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Dump configs for reference
    dump_cfg(cfg)
    
    # Log GPU info and run directory
    if torch.cuda.is_available():
        logging.info(f"GPU Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    logging.info(f"Run directory: {cfg.run_dir}")
    logging.info(f"Seed: {cfg.seed}")
    
    # Log optimized configuration
    logging.info(f"=== OPTIMIZED MOE CONFIGURATION ===")
    logging.info(f"Routing mode: {cfg.gt.routing_mode}")
    logging.info(f"Expert types: {cfg.gt.head_gnn_types}")
    logging.info(f"Number of layers: {cfg.gt.layers}")
    logging.info(f"Uncertainty enabled: {cfg.gt.uncertainty.enabled}")
    logging.info(f"Training mode: {cfg.train.mode}")
    logging.info(f"Optimization: Uncertainty only for last layer + test/val")
    logging.info(f"Additional features: Router weights logging + JSON export")
    
    # Create data loaders
    loaders = create_loader()

    # Print dataset info  
    logging.info(f"[*] Loaded dataset '{cfg.dataset.name}' from '{cfg.dataset.format}'")
    
    # Create model, optimizer, scheduler
    model = create_model()
    
    # DEBUG: Check what model was actually created
    logging.info(f"Created model type: {type(model)}")
    if hasattr(model, 'model'):
        logging.info(f"Inner model type: {type(model.model)}")
        logging.info(f"Inner model has get_darts_model: {hasattr(model.model, 'get_darts_model')}")
    else:
        logging.info(f"Model has get_darts_model: {hasattr(model, 'get_darts_model')}")
    
    cfg.params = params_count(model)
    optimizer = create_optimizer(model.parameters(), cfg.optim)
    scheduler = create_scheduler(optimizer, cfg.optim)
   

    # Print model info
    logging.info(model)
    logging.info(f"Number of parameters: {params_count(model):,}")
    
    # Start timer for training
    start_time = time.time()
    logging.info(f"Starting optimized training: {datetime.datetime.now()}")
    
    # Create data loaders
    loaders = create_loader()
    
    # Print dataset info
    logging.info(f"[*] Loaded dataset '{cfg.dataset.name}' from '{cfg.dataset.format}'")
    
    # Create loggers for tracking metrics
    loggers = create_logger()
    cfg.run_id = 0
    
    # Create training function (use optimized version)
    train_func = train_dict[cfg.train.mode]
    
    # Launch optimized training
    logging.info("=== STARTING OPTIMIZED TRAINING ===")
    train_func(loggers, loaders, model, optimizer, scheduler)
    
    # Calculate total time
    total_time = time.time() - start_time
    logging.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    
    # Aggregate results from different runs
    agg_runs(cfg.out_dir, cfg.metric_best)
    
    # Final message
    logging.info('=== OPTIMIZED TRAINING COMPLETED SUCCESSFULLY! ===')
    logging.info(f'Results saved in: {cfg.run_dir}')
    logging.info(f'Test results JSON files saved in: {cfg.run_dir}/test_results/')

if __name__ == "__main__":
    main()
