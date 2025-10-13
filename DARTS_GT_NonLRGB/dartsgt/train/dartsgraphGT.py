# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 18:09:01 2025

@author: SSC
"""


######################################################################################################################################################################SPARSE HANDLING###################################################################################################################################################################################################################################################

import copy
import logging
import warnings
from collections import OrderedDict
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.retiarii.oneshot import interface
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device
from torch_geometric.loader import DataLoader
from . import dartsgraphbase

import time
from datetime import timedelta
_logger = logging.getLogger(__name__)

# Modified DARTS LayerChoice for storing alpha values
class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
        self.op_names = layer_choice.names  # Store operation names

    def forward(self, *args, **kwargs):
        # Check if we're handling edge attributes
        if len(args) >= 3:  # H, edge_index, edge_attr
            H, edge_index, edge_attr = args[0], args[1], args[2]
            op_results = []
            
            # Process each operation with appropriate arguments
            for name, op in self.op_choices.items():
                # Operations that USE edge attributes
                if hasattr(op, '__class__') and op.__class__.__name__ in [
                    'GENConv', 'GINEConv', 'GATConv', 'PNAConv', 
                    'GatedGCNLayer', 'GINEConvESLapPE'
                ]:
                    result = op(H, edge_index, edge_attr)
                # Operations that DON'T use edge attributes  
                elif hasattr(op, '__class__') and op.__class__.__name__ in [
                    'GCNConv', 'GINConv'
                ]:
                    result = op(H, edge_index)
                else:
                    # Fallback: try with edge attributes first, then without
                    try:
                        result = op(H, edge_index, edge_attr)
                    except:
                        result = op(H, edge_index)
                        
                op_results.append(result)
        else:
            # Standard case without edge attributes
            op_results = [op(*args, **kwargs) for op in self.op_choices.values()]
            
        op_results = torch.stack(op_results)
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]
    
    def get_op_attentions(self, *args, **kwargs):
        """Get output for each individual operation"""
        return {name: op(*args, **kwargs) for name, op in self.op_choices.items()}

class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]
    
    
# Modified DARTS trainer for sparse graph data
class DartsTrainerGraphSparse(dartsgraphbase.DartsTrainerGraph):
    """
    Modified DARTS trainer to handle sparse graph data from PyTorch Geometric
    """
    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, train_loader, test_loader, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False,params=None,timing_callback=None,is_multilabel=False):
        
        warnings.warn('DartsTrainer is deprecated. Please use strategy.DARTS instead.', DeprecationWarning)
        self.initial_arc_lr = arc_learning_rate
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency
        self.model.to(self.device)
        self.timing_callback = timing_callback
        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)
        
        self.model_optim = optimizer
        # use the same architecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.grad_clip = grad_clip
        self.is_multilabel = is_multilabel
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                    self.model_optim, mode='min',
                                                    factor=params['lr_reduce_factor'] if params else 0.5,
                                                    patience=params['lr_schedule_patience'] if params else 50,

                                                )
        
        self.min_lr = params['min_lr'] if params else 0.00001
        
    def _get_task_type(self):
        """Detect task type from loss function"""
        if isinstance(self.loss, (torch.nn.L1Loss, torch.nn.MSELoss)):
            return 'regression'
        elif isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            if self.is_multilabel:
                return 'classification_multilabel'
            else:
                return 'classification_binary'
        elif isinstance(self.loss, torch.nn.CrossEntropyLoss):
            return 'classification_multi'
        else:
            return 'unknown'

    def _train_one_epoch(self, epoch):
        # Call timing callback at epoch start
        epoch_start_time = time.time()
    
        if hasattr(self, 'timing_callback') and self.timing_callback is not None:
            self.timing_callback.on_epoch_start()
        self.model.train()
        meters = AverageMeterGroup()
        
        # Create zip iterator for train and validation loaders
        train_val_iter = zip(self.train_loader, self.test_loader)
        
        for step, (trn_batch, val_batch) in enumerate(train_val_iter):
            # Move data to device
            trn_batch = trn_batch.to(self.device)
            val_batch = val_batch.to(self.device)
            
            # Phase 1: Architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_batch, val_batch)
            else:
                self._backward(val_batch)
            self.ctrl_optim.step()
            
            # Phase 2: Child network step
            self.model_optim.zero_grad()
            output = self.model(trn_batch)
    
            # Handle tuple output
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            # Handle multi-label targets differently
            if self.is_multilabel:
                
                targets = trn_batch.y  # Keep 2D shape for multi-label
                #mask = ~torch.isnan(targets)
                targets_for_metrics = trn_batch.y  # Also keep 2D for metrics
            else:
                targets = trn_batch.y.view(-1)  # Flatten for other tasks
                targets_for_metrics = trn_batch.y.view(-1)
                
            task_type = self._get_task_type()
            
            # SHAPE HANDLING BASED ON TASK TYPE  
            if task_type == 'regression':
                if len(logits.shape) > 1 and logits.shape[1] == 1:
                    logits = logits.squeeze(-1)
                loss = self.loss(logits, targets.float())
            elif task_type in [ 'classification_binary','classification_multilabel']:
                if self.is_multilabel:

                    
                    # Create mask for valid targets
                    mask = ~torch.isnan(targets)
                    
                    if mask.sum() > 0:
                        # Option 1: Use pos_weight to ignore NaN positions
                        # Set pos_weight to 0 for NaN positions
                        pos_weight = mask.float()
                        
                        # Replace NaN with 0 (they'll be ignored due to pos_weight=0)
                        targets_clean = targets.clone()
                        targets_clean[~mask] = 0
                        
                        # Create loss with pos_weight
                        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        loss = loss_fn(logits, targets_clean)
                else:
                    # Single-label binary: squeeze if needed
                    if len(logits.shape) > 1 and logits.shape[1] == 1:
                        logits = logits.squeeze(-1)
                    loss = self.loss(logits, targets.float())
            else:  # multi-class
                loss = self.loss(logits, targets.long())
                
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.model_optim.step()
            
            # Compute metrics with correct target shape
            metrics = self.metrics(logits, targets_for_metrics)
            metrics['loss'] = loss.item()
            
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.warning('Epoch [%s/%s] Step [%s/%s]  %s', epoch + 1,
                             self.num_epochs, step + 1, len(self.test_loader), meters)
                if torch.cuda.is_available():
                    print('GPU memory consumption ', self._get_gpu_memory_summary())
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {timedelta(seconds=epoch_time)}")
            
            
         
            
                    
        
            

        
    def _validatelr(self):
        """Compute validation loss for scheduler"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for val_batch in self.test_loader:
                val_batch = val_batch.to(self.device)
                output = self.model(val_batch)
    
                # FIX: Handle tuple output
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                if self.is_multilabel:
                    targets = val_batch.y  # Keep 2D shape for multi-label
                    #targets_for_metrics = val_batch.y  # Also keep 2D for metrics
                else:
                    targets = val_batch.y.view(-1)  # Flatten for other tasks
                    #targets_for_metrics = val_batch.y.view(-1)
                #targets = val_batch.y.view(-1)
                task_type = self._get_task_type()
                # SAME SHAPE HANDLING LOGIC
                if task_type == 'regression':
                    if len(logits.shape) > 1 and logits.shape[1] == 1:
                        logits = logits.squeeze(-1)
                    loss = self.loss(logits, targets.float())
                elif task_type in [ 'classification_binary','classification_multilabel']:
                    if self.is_multilabel:

                        
                        # Create mask for valid targets
                        mask = ~torch.isnan(targets)
                        
                        if mask.sum() > 0:
                            # Option 1: Use pos_weight to ignore NaN positions
                            # Set pos_weight to 0 for NaN positions
                            pos_weight = mask.float()
                            
                            # Replace NaN with 0 (they'll be ignored due to pos_weight=0)
                            targets_clean = targets.clone()
                            targets_clean[~mask] = 0
                            
                            # Create loss with pos_weight
                            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            loss = loss_fn(logits, targets_clean)
                    else:
                        # Single-label binary: squeeze if needed
                        if len(logits.shape) > 1 and logits.shape[1] == 1:
                            logits = logits.squeeze(-1)
                        loss = self.loss(logits, targets.float())
                else:  # multi-class
                    loss = self.loss(logits, targets.long())
                

                total_loss += loss.item() * val_batch.num_graphs
                total_samples += val_batch.num_graphs
                #print('val_batch.num_graphs ',val_batch.num_graphs)
                #print('total_samples ',total_samples)
        
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def _get_gpu_memory_summary(self):
        """Returns a summary of GPU memory usage in MB."""
        allocated = round(torch.cuda.memory_allocated() / 1024**2, 1)
        reserved = round(torch.cuda.memory_reserved() / 1024**2, 1)
        return f"GPU Memory: Allocated: {allocated} MB, Reserved: {reserved} MB"

    def _backward(self, val_batch):
        """
        Simple backward with gradient descent for sparse graph data
        """
        output = self.model(val_batch)
    
        # FIX: Handle tuple output from GraphGym models
        if isinstance(output, tuple):
            logits = output[0]  # Take only predictions, ignore true labels
        else:
            logits = output
        # Handle multi-label targets differently
        if self.is_multilabel:
            targets = val_batch.y  # Keep 2D shape for multi-label
        else:
            targets = val_batch.y.view(-1)  # Flatten for other tasks
        
        task_type = self._get_task_type()
        
        # SHAPE HANDLING BASED ON TASK TYPE
        if task_type == 'regression':
            # For regression: ensure same shape
            if len(logits.shape) > 1 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)  # [batch, 1] -> [batch]
            loss = self.loss(logits, targets.float())
        elif task_type in [ 'classification_binary','classification_multilabel']:
            if self.is_multilabel:

                
                # Create mask for valid targets
                mask = ~torch.isnan(targets)
                
                if mask.sum() > 0:
                    # Option 1: Use pos_weight to ignore NaN positions
                    # Set pos_weight to 0 for NaN positions
                    pos_weight = mask.float()
                    
                    # Replace NaN with 0 (they'll be ignored due to pos_weight=0)
                    targets_clean = targets.clone()
                    targets_clean[~mask] = 0
                    
                    # Create loss with pos_weight
                    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fn(logits, targets_clean)
            else:
                # Single-label binary: squeeze if needed
                if len(logits.shape) > 1 and logits.shape[1] == 1:
                    logits = logits.squeeze(-1)
                loss = self.loss(logits, targets.float())
        else:  # multi-class
            loss = self.loss(logits, targets.long())
        loss.backward()

    def _unrolled_backward(self, trn_batch, val_batch):
        """
        Compute unrolled loss and backward its gradients for sparse data
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # Virtual step on training data
        lr = self.model_optim.param_groups[0]["lr"]
        momentum = self.model_optim.param_groups[0].get("momentum", 0)
        weight_decay = self.model_optim.param_groups[0]["weight_decay"]
        
        # Compute virtual model (don't need zero_grad, using autograd)
        logits = self.model(trn_batch)
        loss = self.loss(logits, trn_batch.y.view(-1))
        gradients = torch.autograd.grad(loss, self.model.parameters())
        
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.model_optim.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

        # Calculate unrolled loss on validation data
        logits = self.model(val_batch)
        if self.is_multilabel:
            # Multi-label: keep 2D shapes
            loss = self.loss(logits, val_batch.y.float())
        else:
            loss = self.loss(logits, val_batch.y.view(-1))
        
        #loss = self.loss(logits, val_batch.y.view(-1))
        
        w_model, w_ctrl = tuple(self.model.parameters()), tuple([c.alpha for _, c in self.nas_modules])
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # Compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_batch)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # Restore weights
        self._restore_weights(backup_params)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, data):
        """
        Compute hessian for sparse data
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            _logger.warning('In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d
                    
            logits = self.model(data)
            loss = self.loss(logits, data.y.view(-1))
            dalphas.append(torch.autograd.grad(loss, [c.alpha for _, c in self.nas_modules]))

        dalpha_pos, dalpha_neg = dalphas
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
    
    
    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)
            # Compute validation loss and step scheduler
            val_loss = self._validatelr()
            self.scheduler.step(val_loss)
            
            # Check for early stopping
            #print('lr' , self.model_optim.param_groups[0]['lr'])
            #print('self.min_lr ' ,self.min_lr)
            
