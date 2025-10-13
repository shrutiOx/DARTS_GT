# -*- coding: utf-8 -*-
"""
NAS Uncertainty Training - Two-Phase Training with DARTS + Uncertainty
Phase 1: DARTS search (uniform: 2 epochs, random: 2 epochs, nas: full search)
Phase 2: Discrete training with uncertainty (no variance)
"""

import logging
import time
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.loader import DataLoader

from dartsgt.loss.subtoken_prediction_loss import subtoken_cross_entropy
from dartsgt.utils import cfg_to_dict, flatten_dict, make_wandb_name


from dartsgt.train.dartsgraphGT import DartsTrainerGraphSparse


def split_train_dataset_for_darts(loader, split_ratio=0.6):
    """
    Split training dataset into 60/40 for DARTS (only used in 'nas' mode)
    """
    train_dataset = loader.dataset
    train_size = len(train_dataset)
    darts_train_size = int(split_ratio * train_size)
    darts_val_size = train_size - darts_train_size
    
    logging.info(f"Splitting dataset for DARTS:")
    logging.info(f"  Original train size: {train_size}")
    logging.info(f"  DARTS train size: {darts_train_size} ({split_ratio*100:.1f}%)")
    logging.info(f"  DARTS val size: {darts_val_size} ({(1-split_ratio)*100:.1f}%)")
    
    # Create reproducible split
    generator = torch.Generator().manual_seed(cfg.seed)
    darts_train_dataset, darts_val_dataset = torch.utils.data.random_split(
        train_dataset, [darts_train_size, darts_val_size], generator=generator
    )
    
    # Create new loaders with same batch size
    darts_train_loader = DataLoader(
        darts_train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers
    )
    
    darts_val_loader = DataLoader(
        darts_val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers
    )
    
    return darts_train_loader, darts_val_loader

class DartTrainer:
    """
    DARTS trainer based on user's ASPECT-GT approach
    """
    def __init__(self):
        super().__init__()
        self.saveaccuracy = []
        
    def DARTTrain(self, 
                  model_space, 
                  train_loader, 
                  val_loader, 
                  epochs=10, 
                  batches=32, 
                  arc_learning_rate=0.01,
                  grad_clip=5.0,
                  paramsDARTS=None):
        """
        Train model with DARTS to optimize architecture
        """
        # Define accuracy function for DARTS
        # Define accuracy function for DARTS
        def dart_accuracy(y_hat, y):
            if isinstance(y_hat, tuple):
                logits = y_hat[0]
            else:
                logits = y_hat
            
            if cfg.dataset.task_type == 'regression':
                mae = torch.nn.L1Loss()(logits, y)
                #self.saveaccuracy.append(mae.item())
                return {"mae": mae.item()}
            elif cfg.dataset.task_type =='classification_multilabel':
                # Handle multi-label case with 2D tensors
                preds = (torch.sigmoid(logits) > 0.5).float()
                mask = ~torch.isnan(y)
                if mask.sum() > 0:
                    acc = ((preds[mask] == y[mask]).float().sum() / mask.sum()).item()
                
                #self.saveaccuracy.append(acc)
                return {"acc": acc}
            elif cfg.dataset.task_type in ['classification_binary', 'classification_multi', 'classification']:
                if len(logits.shape) > 1 and logits.shape[1] == 1:
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                elif len(logits.shape) == 1:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                else:
                    preds = logits.argmax(dim=1).float()
                
                # Handle both 1D and 2D y tensors
                if len(y.shape) > 1:
                    y = y.squeeze()
                
                acc = (preds == y.float()).float().mean()
                #self.saveaccuracy.append(acc.item())
                return {"acc": acc.item()}
            
        if cfg.dataset.task_type == 'regression':
            criterion = torch.nn.L1Loss()
        elif cfg.dataset.task_type in ['classification_binary', 'classification_multi', 'classification','classification_multilabel']:
            if cfg.dataset.is_binary == 'binary':
                criterion = torch.nn.BCEWithLogitsLoss()
            elif cfg.dataset.is_binary == 'multi_class':
                criterion = torch.nn.CrossEntropyLoss()
            elif cfg.dataset.is_binary == 'multi_label':
                criterion = torch.nn.BCEWithLogitsLoss()
                
        optimizer = torch.optim.Adam(
            model_space.parameters(),
            lr=paramsDARTS['init_lr'] if paramsDARTS else 0.0025,
            weight_decay=paramsDARTS['weight_decay'] if paramsDARTS else 0.0
        )
        
        device = torch.device(cfg.accelerator)
        
        logging.info("Starting DARTS architecture search")
        
        # Initialize DARTS trainer (using existing dartsgraphGT)

        
        
        trainer = DartsTrainerGraphSparse(
            model=model_space.to(device),
            loss=criterion,
            metrics=lambda y_hat, y: dart_accuracy(y_hat, y),
            optimizer=optimizer,
            num_epochs=epochs,
            train_loader=train_loader,
            test_loader=val_loader,
            batch_size=batches,
            log_frequency=10,
            workers=0,
            device=device,
            arc_learning_rate=arc_learning_rate,
            grad_clip=grad_clip,
            params=paramsDARTS if paramsDARTS else {},
            is_multilabel=(cfg.dataset.is_binary == 'multi_label')
        )
        
        # Run DARTS training
        trainer.fit()
        
        # Get the trained model
        final_model = trainer.model.to(device)
        
        # Calculate average accuracy
        #DART_acc = torch.mean(torch.tensor(self.saveaccuracy)) if self.saveaccuracy else 0.0

        #logging.info(f'Mean accuracy after DARTS training: {DART_acc}')
        
        return final_model, None, trainer.nas_modules

def run_darts_phase(model, train_loader, val_loader=None):
    """
    Run DARTS architecture search phase based on routing mode
    """
    routing_mode = cfg.gt.routing_mode
    
    # ALWAYS create darts loaders regardless of mode

    darts_train_loader, darts_val_loader = split_train_dataset_for_darts(
                train_loader, 
                split_ratio=getattr(cfg.gt.nas, 'darts_split_ratio', 0.6)
                )
    
    
    # Configure model for DARTS
    
    # HANDLE GRAPHGYM MODEL WRAPPING
    if hasattr(model, 'model'):
        # GraphGym wraps the actual model in a GraphGymModule
        actual_model = model.model
        logging.info(f"Found GraphGym wrapper, using inner model: {type(actual_model)}")
    else:
        actual_model = model
        logging.info(f"Using model directly: {type(actual_model)}")
    
    # Configure model for DARTS
    if hasattr(actual_model, 'get_darts_model'):
        darts_model = actual_model.get_darts_model()
        logging.info("Successfully configured model for DARTS training")
    else:
        logging.error(f"Model {type(actual_model)} does not have get_darts_model method")
        raise AttributeError(f"Model {type(actual_model)} does not support DARTS training")
    
    
    
    
    nas_config = cfg.gt.nas 
    arc_learning_rate = nas_config.get('arc_learning_rate', 3.0e-4)
    grad_clip = nas_config.get('grad_clip', 5.0)
    # Set epochs based on mode
    if routing_mode in ['uniform', 'random']:
        epochs = 2
        logging.info(f"{routing_mode.upper()} MODE: Running 2 epochs with DARTS")
    else:  # nas mode
        epochs = nas_config.get('darts_epochs', 50)
        logging.info(f"NAS MODE: Running {epochs} epochs with DARTS")
    
   
        
    logging.info(f"DARTS Configuration:")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Architecture LR: {arc_learning_rate}")
    logging.info(f"  Grad clip: {grad_clip}")
        
    # Run DARTS training
    start_time = time.time()
        
    darts_trainer = DartTrainer()
    final_model, exported_arch, nas_modules = darts_trainer.DARTTrain(
                                                    model_space=darts_model,
                                                    train_loader=darts_train_loader,
                                                    val_loader=darts_val_loader,
                                                    epochs=epochs,
                                                    batches=cfg.train.batch_size,
                                                    arc_learning_rate=arc_learning_rate,
                                                    grad_clip=grad_clip,
                                                    paramsDARTS=nas_config.get('darts_lr_schedule', {})
                                                )
        
    darts_time = time.time() - start_time
    logging.info(f"DARTS search completed in {darts_time:.2f}s")
        
    # Get metrics DIRECTLY from NAS_model
    metrics_dict = final_model.get_metrics_dict()
    optimal_weights = extract_optimal_weights_from_metrics(metrics_dict, routing_mode)
    
    return optimal_weights
    
 

def extract_optimal_weights_from_metrics(metrics_dict, routing_mode):
    """
    Extract BEST EXPERT INDEX for EVERY layer from metrics dict
    """
    try:
        # Look in cumulative_metrics which has the final alpha values
        if 'cumulative_metrics' in metrics_dict and 'alphas' in metrics_dict['cumulative_metrics']:
            cumulative_alphas = metrics_dict['cumulative_metrics']['alphas']
            
            layer_best_experts = {}  # Changed from layer_weights
            num_experts = len(cfg.gt.head_gnn_types)
            
            for layer_key, layer_data in cumulative_alphas.items():
                # layer_data is like {'layer_0': {'0': 0.2504, '1': 0.2528, ...}}
                # Get the inner dict with actual alpha values
                layer_alphas = layer_data.get(layer_key, {})
                
                # Collect all alphas for this layer
                alphas_list = []
                for i in range(num_experts):
                    weight = layer_alphas.get(str(i), 0.0)
                    alphas_list.append(weight)
                
                # Find best expert index
                best_expert_idx = np.argmax(alphas_list)
                best_expert_name = cfg.gt.head_gnn_types[best_expert_idx]
                
                layer_best_experts[layer_key] = best_expert_idx
                
                # DETAILED LOGGING FOR VERIFICATION
                logging.info(f"\n{'='*60}")
                logging.info(f"Layer {layer_key} Expert Selection:")
                for i, (gnn_type, alpha) in enumerate(zip(cfg.gt.head_gnn_types, alphas_list)):
                    marker = " ← SELECTED" if i == best_expert_idx else ""
                    logging.info(f"  Expert {i}: {gnn_type} (α={alpha:.4f}){marker}")
                logging.info(f"Selected Expert Index: {best_expert_idx} ({best_expert_name})")
                logging.info(f"{'='*60}\n")
            
            return layer_best_experts
    
    except Exception as e:
        logging.warning(f"Failed to extract optimal weights from metrics: {e}")
    

def train_epoch_optimized(logger, loader, model, optimizer, scheduler, batch_accumulation):
    """
    Training epoch - same as MOE uncertainty train
    """
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    
    # Track router weights for the epoch
    epoch_router_weights = []
    
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        loss.backward()
        
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        
        # Collect router weights
        if hasattr(batch, 'routing_weights') and batch.routing_weights:
            last_layer_weights = batch.routing_weights[-1]
            epoch_router_weights.append(last_layer_weights.numpy())
        
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
    
    # Log router weights stats
    if epoch_router_weights:
        all_weights = np.vstack(epoch_router_weights)
        mean_weights = all_weights.mean(axis=0)
        std_weights = all_weights.std(axis=0)
        
        logging.info(f"TRAIN Router Weights (mean): {[f'{w:.3f}' for w in mean_weights]}")
        logging.info(f"TRAIN Router Weights (std):  {[f'{w:.3f}' for w in std_weights]}")
        
        most_used_expert = np.argmax(mean_weights)
        least_used_expert = np.argmin(mean_weights)
        logging.info(f"TRAIN Most used expert: {most_used_expert} ({mean_weights[most_used_expert]:.3f})")
        logging.info(f"TRAIN Least used expert: {least_used_expert} ({mean_weights[least_used_expert]:.3f})")

@torch.no_grad()
def eval_epoch_with_uncertainty(logger, loader, model, split='val'):
    """
    Evaluation with uncertainty - NO variance computation (removed)
    """
    model.eval()
    time_start = time.time()
    
    # Track only uncertainty (no variance)
    epoch_routing_uncertainties = []
    epoch_router_weights = []
    all_predictions = []
    all_true_labels = []
    all_graph_ids = []
    
    for batch_idx, batch in enumerate(loader):
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
            
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        # Collect routing uncertainty (test only)
        if hasattr(batch, 'routing_uncertainty') and batch.routing_uncertainty is not None:
            epoch_routing_uncertainties.extend(batch.routing_uncertainty.cpu().numpy())
        else:
            if hasattr(batch, 'batch'):
                num_graphs = len(torch.unique(batch.batch))
                epoch_routing_uncertainties.extend([None] * num_graphs)
        
        if hasattr(batch, 'routing_weights') and batch.routing_weights:
            last_layer_weights = batch.routing_weights[-1]
            epoch_router_weights.append(last_layer_weights.numpy())
        
        # Collect test results for JSON export
        if split == 'test':
            batch_predictions = _pred.numpy() if hasattr(_pred, 'numpy') else _pred
            batch_true = _true.numpy() if hasattr(_true, 'numpy') else _true
            
            all_predictions.extend(batch_predictions.tolist() if hasattr(batch_predictions, 'tolist') else [batch_predictions])
            all_true_labels.extend(batch_true.tolist() if hasattr(batch_true, 'tolist') else [batch_true])
            
            if hasattr(batch, 'graph_id'):
                batch_graph_ids = batch.graph_id.cpu().numpy().tolist()
            else:
                unique_batch_ids = torch.unique(batch.batch).cpu().numpy()
                batch_graph_ids = [batch_idx * len(unique_batch_ids) + i for i in range(len(unique_batch_ids))]
            
            all_graph_ids.extend(batch_graph_ids)
        
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    
    # Log routing uncertainty (test only)
    if split == 'test':
        valid_uncertainties = [u for u in epoch_routing_uncertainties if u is not None]
        if valid_uncertainties:
            mean_uncertainty = np.mean(valid_uncertainties)
            std_uncertainty = np.std(valid_uncertainties)
            logging.info(f"{split.upper()} Routing Uncertainty: {mean_uncertainty:.4f}±{std_uncertainty:.4f}")
    
    if epoch_router_weights:
        all_weights = np.vstack(epoch_router_weights)
        mean_weights = all_weights.mean(axis=0)
        std_weights = all_weights.std(axis=0)
        
        logging.info(f"{split.upper()} Router Weights (mean): {[f'{w:.3f}' for w in mean_weights]}")
        most_used_expert = np.argmax(mean_weights)
        least_used_expert = np.argmin(mean_weights)
        logging.info(f"{split.upper()} Most used expert: {most_used_expert} ({mean_weights[most_used_expert]:.3f})")
        logging.info(f"{split.upper()} Least used expert: {least_used_expert} ({mean_weights[least_used_expert]:.3f})")
    
    # Save test results (NO variance, only uncertainty)
    if split == 'test' and all_predictions:
        save_test_results_to_json(all_graph_ids, all_predictions, all_true_labels, 
                                 [], epoch_routing_uncertainties, epoch_router_weights)  # Empty variance list

def save_test_results_to_json(graph_ids, predictions, true_labels, routing_variances, routing_uncertainties, router_weights):
    """
    Save test results - NO variance data (removed)
    """
    results_dir = Path(cfg.run_dir) / 'test_results'
    results_dir.mkdir(exist_ok=True)
    
    if router_weights:
        all_router_weights = np.vstack(router_weights)
    else:
        all_router_weights = None
    
    # Save individual graph results
    for i, graph_id in enumerate(graph_ids):
        graph_result = {
            'graph_id': graph_id,
            'prediction': predictions[i] if i < len(predictions) else None,
            'true_label': true_labels[i] if i < len(true_labels) else None,
            # NO variance data
            'routing_uncertainty': float(routing_uncertainties[i]) if i < len(routing_uncertainties) and routing_uncertainties[i] is not None else None,
            'router_weights': [float(x) for x in all_router_weights[i]] if all_router_weights is not None and i < len(all_router_weights) else None,
            'expert_names': cfg.gt.head_gnn_types,
            'routing_mode': cfg.gt.routing_mode,
            'model_config': {
                'num_layers': cfg.gt.layers,
                'dim_hidden': cfg.gt.dim_hidden,
                'num_heads': cfg.gt.n_heads,
                'num_experts': len(cfg.gt.head_gnn_types),
                'uncertainty_config': {
                    'enabled': cfg.gt.uncertainty.enabled,
                    'delta': getattr(cfg.gt.uncertainty, 'delta', 0.02),
                    'epsilon': getattr(cfg.gt.uncertainty, 'epsilon', 0.15),
                    'max_steps': getattr(cfg.gt.uncertainty, 'max_steps', 10),
                    'samples': getattr(cfg.gt.uncertainty, 'samples', 5),
                }
            }
        }
        
        graph_file = results_dir / f'graph_{graph_id}_result.json'
        with open(graph_file, 'w') as f:
            json.dump(graph_result, f, indent=2)
    
    # Save summary statistics (NO variance)
    summary_stats = {
        'total_graphs': len(graph_ids),
        'prediction_stats': {
            'mean': float(np.mean(predictions)) if predictions else None,
            'std': float(np.std(predictions)) if predictions else None,
            'min': float(np.min(predictions)) if predictions else None,
            'max': float(np.max(predictions)) if predictions else None,
        },
        # NO variance stats
        'routing_uncertainty_stats': {
            'mean': float(np.mean([u for u in routing_uncertainties if u is not None])) if any(u is not None for u in routing_uncertainties) else None,
            'valid_count': len([u for u in routing_uncertainties if u is not None]),
        } if routing_uncertainties else None,
        'config': {
            'dataset': cfg.dataset.name,
            'routing_mode': cfg.gt.routing_mode,
            'expert_types': cfg.gt.head_gnn_types,
            'nas_enabled': cfg.gt.routing_mode == 'nas',
        }
    }
    
    summary_file = results_dir / 'summary_stats.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logging.info(f"Saved test results for {len(graph_ids)} graphs to {results_dir}")

@register_train('NoMixNas_uncertainty_train')
def NoMixNas_uncertainty_train(loggers, loaders, model, optimizer, scheduler):
    """
    Two-Phase NAS Training with Uncertainty (NO variance)
    Phase 1: DARTS search (uniform: 2 epochs, random: 2 epochs, nas: full search)
    Phase 2: Discrete training with uncertainty only
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
        return
    
    logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    
    logging.info("=" * 80)
    logging.info("STARTING TWO-PHASE NAS TRAINING")
    logging.info("=" * 80)
    logging.info(f"Routing mode: {cfg.gt.routing_mode}")
    logging.info(f"Expert types: {cfg.gt.head_gnn_types}")
    logging.info(f"Phase 1: Architecture search/initialization")
    logging.info(f"Phase 2: Discrete training with uncertainty (NO variance)")
    
    # Phase 1 - Architecture Search/Initialization
    logging.info("=" * 60)
    logging.info("PHASE 1: ARCHITECTURE SEARCH/INITIALIZATION")
    logging.info("=" * 60)
    
    # SAVE RNG STATE BEFORE DARTS
    '''
    import random
    import numpy as np
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    '''
    # Run DARTS phase based on routing mode
    optimal_weights = run_darts_phase(model, loaders[0])
    # HANDLE GRAPHGYM WRAPPING FOR DISCRETE MODEL
    '''
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    '''
    # ===== CLEAR ALL ACCUMULATED STATE =====
    model.zero_grad()  # Clear gradients
    for param in model.parameters():
        param.grad = None  # Explicitly clear gradients
        
    # Clear any buffer state
    for buffer in model.buffers():
        if buffer.grad is not None:
            buffer.grad = None
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ===== END CLEARING =====


    if hasattr(model, 'model'):
        # Extract actual NAS model, create discrete version, put it back
        actual_model = model.model
        discrete_model = actual_model.get_discrete_model(optimal_weights,cfg.gt.weight_type)
        # SAVE OPTIMAL WEIGHTS FOR PK-EXPLAINER
        discrete_model.optimal_weights_dict = optimal_weights  # ADD THIS!


        #discrete_model = discrete_model.to(torch.device(cfg.accelerator))

        model.model = discrete_model  # ← Simply replace the inner model
        logging.info("Replaced inner model with discrete version")
        del actual_model 
    else:
        # If no wrapper, just replace the model directly
        model = model.get_discrete_model(optimal_weights,cfg.gt.weight_type)
        model.optimal_weights_dict = optimal_weights  # CRITICAL for pk_explainer
        logging.info("Replaced model with discrete version")
    
   
    # CREATE NEW OPTIMIZER AND SCHEDULER FOR DISCRETE MODEL
    logging.info("Creating fresh optimizer/scheduler for discrete model...")
    
    # Import GraphGym's functions (should already be imported at top)
    from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
    
    # Create exactly as main code does
    optimizer = create_optimizer(discrete_model.parameters(), cfg.optim)
    scheduler = create_scheduler(optimizer, cfg.optim)
    
    # Update param count for logging
    cfg.params = sum(p.numel() for p in discrete_model.parameters())
    
    logging.info(f"Fresh optimizer created: {type(optimizer).__name__}")
    logging.info(f"Fresh scheduler created: {type(scheduler).__name__}")
    logging.info(f"Discrete model parameters: {cfg.params:,}")
    
    logging.info("=" * 60)
    logging.info("PHASE 2: DISCRETE TRAINING WITH UNCERTAINTY")
    logging.info("=" * 60)
    
    # Phase 2 - Discrete Training on Full Dataset
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        
        logging.info(f"=== Epoch {cur_epoch} ===")
        
        # Training
        train_epoch_optimized(loggers[0], loaders[0], discrete_model, optimizer, scheduler,
                             cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                # Evaluation with uncertainty (NO variance)
                eval_epoch_with_uncertainty(loggers[i], loaders[i], discrete_model,
                                          split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and is_ckpt_epoch(cur_epoch):
            save_ckpt(discrete_model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]), cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
                save_ckpt(discrete_model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()
            
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
   


    logging.info("=" * 80)
    logging.info("NAS TRAINING COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    logging.info(f"Routing mode: {cfg.gt.routing_mode}")
    logging.info(f"Final optimal weights: {optimal_weights}")
    logging.info(f"Results include routing uncertainty (test only, NO variance)")
    if cfg.gt.pk_explainer.enabled:
        from dartsgt.explainer.pk_explainer import run_pk_analysis
        pk_results = run_pk_analysis(model, loaders[0], loaders[2])  # train, test
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.finish()

    logging.info('Task done, results saved in %s', cfg.run_dir)
    
