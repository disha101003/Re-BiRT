#!/usr/bin/env python3
from tqdm.notebook import tqdm, trange
import dagshub
import mlflow
from mlflow import MlflowClient
from src import const
from src.dataset import balanced_fine_tune_data
from torchsummary import summary
import torch.nn.functional as F
import time
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from src import const
from src.dataset import test_data
from src.models.arch import get_models, D_buffer

def mlflow_log_params():
    mlflow.log_param('Training', 'fine-tuning on balanced dataset')
    mlflow.log_param("DATASET", const.DATASET)
    mlflow.log_param("SUBSET", const.SUBSET)
    mlflow.log_param("NUM_SUBSET_IMAGES", const.NUM_SUBSET_IMAGES)
    mlflow.log_param("tasks", const.tasks)
    mlflow.log_param("epochs", const.fine_tune_epoch)
    mlflow.log_param("base_lr", const.base_lr)
    mlflow.log_param("weight_decay", const.weight_decay)
    mlflow.log_param("batch_size", const.batch_size)
    # logging in all hyperparameters used for BiRT specific training using mlflow
    for key, value in const.config.items():
        mlflow.log_param(key, value)
    mlflow.log_param('optimizer','Adam')

    return

# Balanced train loop
def train_balanced(criterion,optimizer,scheduler, balanced_loader, model_g, model_f_w):
    c = 0
    for epoch in trange(const.fine_tune_epoch, desc="Training"):
        start = time.time()
        train_loss = 0.0
        for batch in tqdm(balanced_loader, desc="Training"):
            x, y = batch
            x, y = x.to(const.device), y.to(const.device)
            y_hat_temp,_ = model_g(x,c)
            y_hat_temp_copy = y_hat_temp.detach().clone()
            y_hat_temp_copy = y_hat_temp.to(const.device)
            y_hat,_ = model_f_w(y_hat_temp_copy,c)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.detach().cpu().item() / len(balanced_loader)

        end = time.time()
        epochtime = end/60 - start/60
        scheduler.step()

        print(f"Epoch {epoch + 1}/{const.fine_tune_epoch} loss: {train_loss:.2f}")
        mlflow.log_metric("Epoch_loss",  train_loss, step = epoch)
        mlflow.log_metric("epoch_time", epochtime, step = epoch)




if __name__ == '__main__':
    dagshub.init("hackathonF23-artix", "ML-Purdue" )
    mlflow.set_tracking_uri("https://dagshub.com/ML-Purdue/hackathonF23-artix.mlflow")
   
    balanced_loader = balanced_fine_tune_data()
    criterion = nn.CrossEntropyLoss()
    
    model_g, model_f_w, model_f_s = get_models()
    optimizer = optim.Adam(list(model_g.parameters()) + list(model_f_w.parameters()), lr=const.base_lr, weight_decay=const.weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(optimizer)
    mlflow.end_run()
    mlflow.start_run()
    mlflow_log_params()
    model_g.load_state_dict(torch.load(const.MODEL_DIR/'model_g'))
    model_f_w.load_state_dict(torch.load(const.MODEL_DIR/'model_f_w'))
    model_f_s.load_state_dict(torch.load(const.MODEL_DIR/'model_f_s'))
    train_balanced(criterion,optimizer,scheduler, balanced_loader, model_g, model_f_w)
    torch.save(model_g.state_dict(),const.MODEL_DIR/'model_g' )
    torch.save(model_f_w.state_dict(),const.MODEL_DIR/'model_f_w' )
    torch.save(model_f_s.state_dict(),const.MODEL_DIR/'model_f_s' )
    mlflow.pytorch.log_model(model_g, "g()", registered_model_name="g()")
    mlflow.pytorch.log_model(model_f_s, "f_s()", registered_model_name="f_s()")
    mlflow.pytorch.log_model(model_f_w, "f_w()", registered_model_name= "f_w()")
    mlflow.end_run()


