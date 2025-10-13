# -*- coding: utf-8 -*-
"""
REWRITTEN: Uncertainty Training with Routing Variance + Perturbation Uncertainty
- Routing Variance: Last layer, val/test (strategy disagreement)
- Routing Uncertainty: Last layer, test only (perturbation-based)
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

from dartsgt.loss.subtoken_prediction_loss import subtoken_cross_entropy
from dartsgt.utils import cfg_to_dict, flatten_dict, make_wandb_name


def train_epoch_optimized(logger, loader, model, optimizer, scheduler, batch_accumulation):
    """Optimized training epoch - no variance/uncertainty tracking."""
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
        
        # === COLLECT ROUTER WEIGHTS ===
        if hasattr(batch, 'routing_weights') and batch.routing_weights:
            # Only take the last layer's routing weights (from the last MOE layer)
            last_layer_weights = batch.routing_weights[-1]  # Last layer's weights
            epoch_router_weights.append(last_layer_weights.numpy())
        
        # Regular logger update
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
    
    # === LOG ROUTER WEIGHTS STATS FOR EPOCH ===
    if epoch_router_weights:
        all_weights = np.vstack(epoch_router_weights)  # [num_batches * batch_size, num_experts]
        mean_weights = all_weights.mean(axis=0)
        std_weights = all_weights.std(axis=0)
        
        # Log router statistics
        logging.info(f"TRAIN Router Weights (mean): {[f'{w:.3f}' for w in mean_weights]}")
        logging.info(f"TRAIN Router Weights (std):  {[f'{w:.3f}' for w in std_weights]}")
        
        # Log expert usage
        most_used_expert = np.argmax(mean_weights)
        least_used_expert = np.argmin(mean_weights)
        logging.info(f"TRAIN Most used expert: {most_used_expert} ({mean_weights[most_used_expert]:.3f})")
        logging.info(f"TRAIN Least used expert: {least_used_expert} ({mean_weights[least_used_expert]:.3f})")


@torch.no_grad()
def eval_epoch_with_variance_uncertainty(logger, loader, model, split='val'):
    """
    CHANGES START: Updated evaluation with both routing variance and uncertainty tracking
    - Routing Variance: val/test (strategy disagreement)  
    - Routing Uncertainty: test only (perturbation-based)
    """
    model.eval()
    time_start = time.time()
    
    # CHANGES START: Track both variance and uncertainty separately
    epoch_routing_variances = []      # For val/test - strategy disagreement
    epoch_routing_uncertainties = []  # For test only - perturbation-based
    epoch_router_weights = []
    all_predictions = []
    all_true_labels = []
    all_graph_ids = []
    # CHANGES END
    
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
        
        # CHANGES START: Collect routing variance and uncertainty separately
        # === COLLECT ROUTING VARIANCE (val/test) ===
        if hasattr(batch, 'routing_variance') and batch.routing_variance is not None:
            epoch_routing_variances.extend(batch.routing_variance.cpu().numpy())
        else:
            # Add None placeholders to maintain index alignment
            if hasattr(batch, 'batch'):
                num_graphs = len(torch.unique(batch.batch))
                epoch_routing_variances.extend([None] * num_graphs)
        
        # === COLLECT ROUTING UNCERTAINTY (test only) ===
        if hasattr(batch, 'routing_uncertainty') and batch.routing_uncertainty is not None:
            epoch_routing_uncertainties.extend(batch.routing_uncertainty.cpu().numpy())
        else:
            # Add None placeholders to maintain index alignment  
            if hasattr(batch, 'batch'):
                num_graphs = len(torch.unique(batch.batch))
                epoch_routing_uncertainties.extend([None] * num_graphs)
        # CHANGES END
        
        if hasattr(batch, 'routing_weights') and batch.routing_weights:
            # Only take the last layer's routing weights
            last_layer_weights = batch.routing_weights[-1]
            epoch_router_weights.append(last_layer_weights.numpy())
        
        # === COLLECT TEST RESULTS FOR JSON EXPORT ===
        if split == 'test':
            # Store predictions, true labels, and graph IDs for JSON export
            batch_predictions = _pred.numpy() if hasattr(_pred, 'numpy') else _pred
            batch_true = _true.numpy() if hasattr(_true, 'numpy') else _true
            
            all_predictions.extend(batch_predictions.tolist() if hasattr(batch_predictions, 'tolist') else [batch_predictions])
            all_true_labels.extend(batch_true.tolist() if hasattr(batch_true, 'tolist') else [batch_true])
            
            # Graph IDs (if available, otherwise use batch indices)
            if hasattr(batch, 'graph_id'):
                batch_graph_ids = batch.graph_id.cpu().numpy().tolist()
            else:
                # Generate sequential IDs based on batch
                unique_batch_ids = torch.unique(batch.batch).cpu().numpy()
                batch_graph_ids = [batch_idx * len(unique_batch_ids) + i for i in range(len(unique_batch_ids))]
            
            all_graph_ids.extend(batch_graph_ids)
        
        # Regular logger update
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    
    # CHANGES START: Log routing variance and uncertainty separately
    # === LOG ROUTING VARIANCE STATS (val/test) ===
    valid_variances = [v for v in epoch_routing_variances if v is not None]
    if valid_variances:
        mean_variance = np.mean(valid_variances)
        std_variance = np.std(valid_variances)
        logging.info(f"{split.upper()} Routing Variance: {mean_variance:.4f}±{std_variance:.4f} ({len(valid_variances)}/{len(epoch_routing_variances)} valid)")
    else:
        logging.warning(f"{split.upper()} Routing Variance: NO VALID VALUES FOUND!")
    
    # === LOG ROUTING UNCERTAINTY STATS (test only) ===
    if split == 'test':
        valid_uncertainties = [u for u in epoch_routing_uncertainties if u is not None]
        if valid_uncertainties:
            mean_uncertainty = np.mean(valid_uncertainties)
            std_uncertainty = np.std(valid_uncertainties)
            logging.info(f"{split.upper()} Routing Uncertainty: {mean_uncertainty:.4f}±{std_uncertainty:.4f} ({len(valid_uncertainties)}/{len(epoch_routing_uncertainties)} valid)")
        else:
            logging.warning(f"{split.upper()} Routing Uncertainty: NO VALID VALUES FOUND!")
    # CHANGES END
    
    if epoch_router_weights:
        all_weights = np.vstack(epoch_router_weights)
        mean_weights = all_weights.mean(axis=0)
        std_weights = all_weights.std(axis=0)
        
        # Log router statistics
        logging.info(f"{split.upper()} Router Weights (mean): {[f'{w:.3f}' for w in mean_weights]}")
        logging.info(f"{split.upper()} Router Weights (std):  {[f'{w:.3f}' for w in std_weights]}")
        
        # Log expert usage
        most_used_expert = np.argmax(mean_weights)
        least_used_expert = np.argmin(mean_weights)
        logging.info(f"{split.upper()} Most used expert: {most_used_expert} ({mean_weights[most_used_expert]:.3f})")
        logging.info(f"{split.upper()} Least used expert: {least_used_expert} ({mean_weights[least_used_expert]:.3f})")
    
    # === SAVE TEST RESULTS TO JSON ===
    if split == 'test' and all_predictions:
        # CHANGES START: Pass both variance and uncertainty to JSON export
        save_test_results_to_json(all_graph_ids, all_predictions, all_true_labels, 
                                 epoch_routing_variances, epoch_routing_uncertainties, epoch_router_weights)
        # CHANGES END


def save_test_results_to_json(graph_ids, predictions, true_labels, routing_variances, routing_uncertainties, router_weights):
    """
    CHANGES START: Save test results with both routing variance and uncertainty
    """
    # Create results directory
    results_dir = Path(cfg.run_dir) / 'test_results'
    results_dir.mkdir(exist_ok=True)
    
    # Prepare router weights
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
            # CHANGES START: Include both variance and uncertainty
            'routing_variance': float(routing_variances[i]) if i < len(routing_variances) and routing_variances[i] is not None else None,
            'routing_uncertainty': float(routing_uncertainties[i]) if i < len(routing_uncertainties) and routing_uncertainties[i] is not None else None,
            # CHANGES END
            'router_weights': [float(x) for x in all_router_weights[i]] if all_router_weights is not None and i < len(all_router_weights) else None,
            'expert_names': cfg.gt.head_gnn_types,
            'routing_mode': cfg.gt.routing_mode,
            'model_config': {
                'num_layers': cfg.gt.layers,
                'dim_hidden': cfg.gt.dim_hidden,
                'num_heads': cfg.gt.n_heads,
                'num_experts': len(cfg.gt.head_gnn_types),
                # CHANGES START: Add uncertainty config
                'uncertainty_config': {
                    'enabled': cfg.gt.uncertainty.enabled,
                    'delta': getattr(cfg.gt.uncertainty, 'delta', 0.02),
                    'epsilon': getattr(cfg.gt.uncertainty, 'epsilon', 0.15),
                    'max_steps': getattr(cfg.gt.uncertainty, 'max_steps', 10),
                    'samples': getattr(cfg.gt.uncertainty, 'samples', 5),
                }
                # CHANGES END
            }
        }
        
        # Save individual graph result
        graph_file = results_dir / f'graph_{graph_id}_result.json'
        with open(graph_file, 'w') as f:
            json.dump(graph_result, f, indent=2)
    
    # CHANGES START: Update summary statistics for both metrics
    # Save summary statistics
    summary_stats = {
        'total_graphs': len(graph_ids),
        'prediction_stats': {
            'mean': float(np.mean(predictions)) if predictions else None,
            'std': float(np.std(predictions)) if predictions else None,
            'min': float(np.min(predictions)) if predictions else None,
            'max': float(np.max(predictions)) if predictions else None,
        },
        'true_label_stats': {
            'mean': float(np.mean(true_labels)) if true_labels else None,
            'std': float(np.std(true_labels)) if true_labels else None,
            'min': float(np.min(true_labels)) if true_labels else None,
            'max': float(np.max(true_labels)) if true_labels else None,
        },
        # CHANGES START: Separate stats for variance and uncertainty
        'routing_variance_stats': {
            'mean': float(np.mean([v for v in routing_variances if v is not None])) if any(v is not None for v in routing_variances) else None,
            'std': float(np.std([v for v in routing_variances if v is not None])) if any(v is not None for v in routing_variances) else None,
            'min': float(np.min([v for v in routing_variances if v is not None])) if any(v is not None for v in routing_variances) else None,
            'max': float(np.max([v for v in routing_variances if v is not None])) if any(v is not None for v in routing_variances) else None,
            'percentiles': {
                '25': float(np.percentile([v for v in routing_variances if v is not None], 25)) if any(v is not None for v in routing_variances) else None,
                '50': float(np.percentile([v for v in routing_variances if v is not None], 50)) if any(v is not None for v in routing_variances) else None,
                '75': float(np.percentile([v for v in routing_variances if v is not None], 75)) if any(v is not None for v in routing_variances) else None,
                '90': float(np.percentile([v for v in routing_variances if v is not None], 90)) if any(v is not None for v in routing_variances) else None,
                '95': float(np.percentile([v for v in routing_variances if v is not None], 95)) if any(v is not None for v in routing_variances) else None,
            },
            'valid_count': len([v for v in routing_variances if v is not None]),
            'total_count': len(routing_variances)
        } if routing_variances else None,
        'routing_uncertainty_stats': {
            'mean': float(np.mean([u for u in routing_uncertainties if u is not None])) if any(u is not None for u in routing_uncertainties) else None,
            'std': float(np.std([u for u in routing_uncertainties if u is not None])) if any(u is not None for u in routing_uncertainties) else None,
            'min': float(np.min([u for u in routing_uncertainties if u is not None])) if any(u is not None for u in routing_uncertainties) else None,
            'max': float(np.max([u for u in routing_uncertainties if u is not None])) if any(u is not None for u in routing_uncertainties) else None,
            'percentiles': {
                '25': float(np.percentile([u for u in routing_uncertainties if u is not None], 25)) if any(u is not None for u in routing_uncertainties) else None,
                '50': float(np.percentile([u for u in routing_uncertainties if u is not None], 50)) if any(u is not None for u in routing_uncertainties) else None,
                '75': float(np.percentile([u for u in routing_uncertainties if u is not None], 75)) if any(u is not None for u in routing_uncertainties) else None,
                '90': float(np.percentile([u for u in routing_uncertainties if u is not None], 90)) if any(u is not None for u in routing_uncertainties) else None,
                '95': float(np.percentile([u for u in routing_uncertainties if u is not None], 95)) if any(u is not None for u in routing_uncertainties) else None,
            },
            'valid_count': len([u for u in routing_uncertainties if u is not None]),
            'total_count': len(routing_uncertainties)
        } if routing_uncertainties else None,
        # CHANGES END
        'router_weights_stats': {
            'mean_weights': all_router_weights.mean(axis=0).tolist() if all_router_weights is not None else None,
            'std_weights': all_router_weights.std(axis=0).tolist() if all_router_weights is not None else None,
            'expert_names': cfg.gt.head_gnn_types,
            'most_used_expert': int(np.argmax(all_router_weights.mean(axis=0))) if all_router_weights is not None else None,
            'least_used_expert': int(np.argmin(all_router_weights.mean(axis=0))) if all_router_weights is not None else None,
        } if all_router_weights is not None else None,
        'config': {
            'dataset': cfg.dataset.name,
            'routing_mode': cfg.gt.routing_mode,
            'expert_types': cfg.gt.head_gnn_types,
            'num_layers': cfg.gt.layers,
            # CHANGES START: Update config info
            'routing_variance_enabled': cfg.gt.uncertainty.enabled,
            'routing_uncertainty_enabled': cfg.gt.uncertainty.enabled,
            'uncertainty_config': {
                'delta': getattr(cfg.gt.uncertainty, 'delta', 0.02),
                'epsilon': getattr(cfg.gt.uncertainty, 'epsilon', 0.15),
                'max_steps': getattr(cfg.gt.uncertainty, 'max_steps', 10),
                'samples': getattr(cfg.gt.uncertainty, 'samples', 5),
            }
            # CHANGES END
        }
    }
    # CHANGES END
    
    # Save summary
    summary_file = results_dir / 'summary_stats.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logging.info(f"Saved test results for {len(graph_ids)} graphs to {results_dir}")
    logging.info(f"Summary stats saved to {summary_file}")


@register_train('uncertainty_train')
def uncertainty_train(loggers, loaders, model, optimizer, scheduler):
    """
    CHANGES START: Updated training with routing variance + uncertainty tracking
    - Routing Variance: Last layer, val/test (strategy disagreement)
    - Routing Uncertainty: Last layer, test only (perturbation-based)
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
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
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    
    # CHANGES START: Updated logging for dual metrics
    logging.info(f"=== DUAL METRIC MOE TRAINING START ===")
    logging.info(f"Routing mode: {cfg.gt.routing_mode}")
    logging.info(f"Expert types: {cfg.gt.head_gnn_types}")
    logging.info(f"Routing variance enabled: {cfg.gt.uncertainty.enabled}")
    logging.info(f"Routing uncertainty enabled: {cfg.gt.uncertainty.enabled}")
    logging.info(f"Routing variance: last layer + val/test")
    logging.info(f"Routing uncertainty: last layer + test only")
    logging.info(f"Uncertainty config: delta={getattr(cfg.gt.uncertainty, 'delta', 0.02)}, "
                f"epsilon={getattr(cfg.gt.uncertainty, 'epsilon', 0.15)}, "
                f"max_steps={getattr(cfg.gt.uncertainty, 'max_steps', 10)}, "
                f"samples={getattr(cfg.gt.uncertainty, 'samples', 5)}")
    # CHANGES END
    
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        
        logging.info(f"=== Epoch {cur_epoch} ===")
        
        # Training (no variance/uncertainty tracking)
        train_epoch_optimized(loggers[0], loaders[0], model, optimizer, scheduler,
                             cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                # CHANGES START: Use updated evaluation function
                eval_epoch_with_variance_uncertainty(loggers[i], loaders[i], model,
                                                   split=split_names[i - 1])
                # CHANGES END
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
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
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
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
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
    
    # CHANGES START: Updated completion logging
    logging.info(f"=== DUAL METRIC TRAINING COMPLETED ===")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    logging.info(f"Results include both routing variance (val/test) and uncertainty (test only)")
    # CHANGES END
    
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)
    # CHANGES END
