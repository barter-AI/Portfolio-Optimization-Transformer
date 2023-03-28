import torch
import sys
import os
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from data import PortfolioDataset
from losses import SharpeLoss
from layers import SineActivation
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import pandas as pd
from absl import app, flags
import argparse
import wandb
import gc
import numpy as np
device = torch.device('cuda')

def train_one_epoch(model, constrain_head, optimizer, lr_scheduler, loss_fn, dataloader, progress_bar, time2vec, eval = False):
    if not eval:
        model.train()
    else:
        model.eval()
    sharpe_list = []
    with torch.set_grad_enabled(mode = not eval):
        for batch in dataloader:
            past_values = batch[0] # BS, SEQ+1, 5 
            past_mask = batch[1] # BS, SEQ+1 
            past_time = batch[2] # BS, SEQ+1, 3
            future_values = batch[3] # BS, PRED+1, 5 
            future_mask = batch[4] # BS, PRED+1 
            future_time = batch[5] # BS, PRED+1, 3

            # static_categorical_features = batch[6][:,:,0] # BS, 1

            # past_time_features = time2vec(past_time)
            # future_time_features = time2vec(future_time)
            past_time_features = past_time
            future_time_features = future_time

            prices = batch[6].to(device)
            future_sz = batch[7].to(device)
            outputs = model(
                past_values = past_values.to(device), 
                past_observed_mask = past_mask.to(device),
                past_time_features = past_time_features.to(device),
                # static_categorical_features = torch.zeros(past_values.shape[0],1).to(device).long(),
                # static_real_features = torch.zeros(past_values.shape[0],4).to(device),
                future_values = future_values.to(device),
                future_time_features = future_time_features.to(device),
                return_dict = False
            )
            weights = constrain_head(outputs[0])
            loss, sharpe, expected_return, variance_return = loss_fn(weights, prices, future_sz)
            
            # train_sharpe.append(-loss.cpu())
            # print(-loss)
            if not eval:
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            sharpe = -sharpe.detach().cpu().numpy()
            sharpe_list.append(sharpe)
            progress_bar.update(1)
            progress_bar.set_postfix(sharpe=sharpe)
            if not eval:
                wandb.log({"Sharpe_train": -loss,
                            "Learning Rate": lr_scheduler.get_last_lr()[0], 
                            "Expected Return": expected_return, 
                            "Variance Return": variance_return})
            else:
                wandb.log({"Sharpe_eval": -loss,
                            "Expected Return eval": expected_return, 
                            "Variance Return eval": variance_return})
            gc.collect()
    return np.mean(sharpe), weights
        

def eval_one_epoch(model, eval_dataloader, time2vec, constrain_head, loss_fn):
    model.eval()
    assert len(eval_dataloader)==1
    for batch in eval_dataloader:
        with torch.no_grad():
            past_values = batch[0] # BS, SEQ+1, 5 
            past_mask = batch[1] # BS, SEQ+1 
            past_time = batch[2] # BS, SEQ+1, 3
            future_values = batch[3] # BS, PRED+1, 5 
            future_mask = batch[4] # BS, PRED+1 
            future_time = batch[5] # BS, PRED+1, 3

            # static_categorical_features = batch[6][:,:,0] # BS, 1

            past_time_features = time2vec(past_time)
            future_time_features = time2vec(future_time)

            prices = batch[6].to(device)
            future_sz = batch[7].to(device)
            outputs = model(
                past_values = past_values.to(device), 
                past_observed_mask = past_mask.to(device),
                past_time_features = past_time_features.to(device),
                static_categorical_features = torch.zeros(past_values.shape[0],1).to(device).long(),
                static_real_features = torch.zeros(past_values.shape[0],4).to(device),
                future_values = future_values.to(device),
                future_time_features = future_time_features.to(device),
                return_dict = False
            )
            weights = constrain_head(outputs[0])
            loss_eval, expected_rp, variance_rp = loss_fn(weights, prices, future_sz)
            print(f"Sharpe_eval: {-loss_eval}")
            wandb.log({"Sharpe_eval": -loss_eval,
                       "Expected Return eval": expected_rp,
                       "Variance Return eval": variance_rp,
                       })
            return -loss_eval