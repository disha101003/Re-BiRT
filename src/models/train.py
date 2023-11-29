#!/usr/bin/env python3
from tqdm.notebook import tqdm, trange
import dagshub
import mlflow
from mlflow import MlflowClient
from src import const
from src.dataset import train_data
import torch.nn.functional as F
import time
import torch
from torch import nn
import numpy as np
import torch.optim as optim


from src.models.arch import get_models, D_buffer
def mlflow_log_params():
   
    mlflow.log_param("DATASET", const.DATASET)
    mlflow.log_param("SUBSET", const.SUBSET)
    mlflow.log_param("NUM_SUBSET_IMAGES", const.NUM_SUBSET_IMAGES)

    mlflow.log_param("tasks", const.tasks)
    mlflow.log_param("epochs", const.epochs)
    mlflow.log_param("base_lr", const.base_lr)
    mlflow.log_param("weight_decay", const.weight_decay)
    mlflow.log_param("batch_size", const.batch_size)
    # logging in all hyperparameters used for BiRT specific training using mlflow
    for key, value in const.config.items():
        mlflow.log_param(key, value)
    mlflow.log_param('optimizer','Adam')

    return

def train(task_list, sem_mem, model_g, model_f_w, model_f_s, criterion, optimizer):
 
    c = 0
    # training loop of BiRT
    for task_index, trainloader in enumerate(tqdm((task_list), desc=f"task in training", leave=False)):
        start = int(time.time()/60) # task training start time
        for epoch in trange(const.epochs, desc="Training_Epochs"):
            train_loss = 0.0
            task_sem_mem_list = []
            for batch_idx, batch in enumerate(tqdm((trainloader), desc=f"Epoch {epoch + 1} in training", leave=False)):
                x, y = batch
                x, y = x.to(const.device), y.to(const.device)
                y_hat_temp,_ = model_g(x,c)                      # get output from g()
                y_hat_temp_copy = y_hat_temp.detach().clone()  # create a detached copy to be processed further

                # store output from g() along with labels in episodic memory

                task_sem_mem_list.append((y_hat_temp_copy, y))

                # contols sampled from a normal distribution to control different noises introduced in BiRT
                alpha_t_comp, alpha_a_comp, alpha_s_comp, alpha_e_comp = np.random.uniform(0,1,4)

                # sampling mini batch from episodic memory
                if (sem_mem.is_empty() == False):
                    r, r_y = sem_mem.get_batch()
                    r = r.to((const.device))
                    r_y = r_y.to(const.device)

                    # implementing label noise
                    if(alpha_t_comp < const.alpha_t):
                        num_change = int(const.percentage_change / 100 * const.batch_size)
                        indices_change_r_y = torch.randperm(len(y))[:num_change].to(const.device)
                        r_y_changed = torch.randint(0,const.num_classes,(num_change,)).to(const.device)
                        r_y[indices_change_r_y] = r_y_changed

                    # implementing attention noise
                    if(alpha_a_comp < const.alpha_a):
                        c = 1

                    r_y_working,_ = model_f_w(r,c)
                    r_y_semantic,_ = model_f_s(r,c)

                    # adding noise to logits of semantic/episodic memory
                    if(alpha_s_comp < const.alpha_s):
                        r_y_semantic =  r_y_semantic + torch.rand(r_y_semantic.size()).to(const.device)*const.std + const.mean

                y_hat_temp = y_hat_temp_copy.to(const.device)
                y_working,_ = model_f_w(y_hat_temp_copy,c)
                y_semantic,_ = model_f_s(y_hat_temp_copy, c)


                # computing loss
                if(task_index == 0):
                    loss_representation = criterion(y_working,y)   # representation loss
                else:
                    loss_representation = criterion(y_working,y) + const.alpha_loss_rep*criterion(model_f_w(r, c)[0], r_y)  # loss

                if(task_index == 0):
                    loss_consistency_reg = const.beta_1_loss * torch.norm(y_working - y_semantic, p=2)**2
                else:
                    loss_consistency_reg = const.beta_1_loss * torch.norm(y_working - y_semantic, p=2)**2  + const.beta_2_loss*torch.norm( r_y_working-r_y_semantic, p = 2)**2  # consistency regulation noise

                loss = loss_representation + const.rho_loss_cr * loss_consistency_reg   # total loss
                loss = loss/const.accum_iter
                loss.backward(retain_graph = True)

                print(loss)
                if ((batch_idx + 1) % const.accum_iter == 0) or (batch_idx + 1 == len(trainloader)):
                    optimizer.step()
                    optimizer.zero_grad()

                # interpolating parameters of epiodic memory at intervals
                if(alpha_e_comp < const.alpha_e and task_index > 0):
                    for params1, params2 in zip(model_f_s.parameters(), model_f_w.parameters()):
                        interpolated_params = const._gamma * params1.data + (1 - const._gamma) * params2.data
                        params1.data.copy_(interpolated_params)

                train_loss += loss.detach().cpu().item() /len(trainloader)

                # Printing average loss per epoch
            print(f"Epoch {epoch + 1}/{const.epochs} loss: {train_loss:.2f}")
            mlflow.log_metric(f"task{task_index}_epoch_loss", train_loss, step = epoch)

        # copying f_w() paramerters to f_s() for first task
        if task_index == 0:
            for params1, params2 in zip(model_f_s.parameters(), model_f_w.parameters()):
                interpolated_params = params2.data
                params1.data.copy_(interpolated_params)

        end = int(time.time()/60) # task training end time
        task_train_time = end - start
        mlflow.log_metric(f"Time taken to complete training task {task_index}",task_train_time )
        print(f"Task {task_index} done in {task_train_time} mins")
        start = int(time.time()/60)

        sem_mem.update(task_sem_mem_list, task_index)
      
        end = int(time.time()/60) # task training end time
        mem_update_time = end - start
        print(f"Memory {task_index} updated in {mem_update_time} mins")
        torch.save(model_g.state_dict(),const.MODEL_DIR/'model_g' )
        torch.save(model_f_w.state_dict(),const.MODEL_DIR/'model_f_w' )
        torch.save(model_f_s.state_dict(),const.MODEL_DIR/'model_f_s' )
        mlflow.log_metric(f"Time taken to update memory after task {task_index}",mem_update_time )

if __name__ == '__main__':
    dagshub.init("hackathonF23-artix", "ML-Purdue" )
    mlflow.set_tracking_uri("https://dagshub.com/ML-Purdue/hackathonF23-artix.mlflow")
    model_g, model_f_w, model_f_s = get_models()
    # model_g.load_state_dict(torch.load(const.MODEL_DIR/'model_g'))
    # model_f_w.load_state_dict(torch.load(const.MODEL_DIR/'model_f_w'))
    # model_f_s.load_state_dict(torch.load(const.MODEL_DIR/'model_f_s'))
    sem_mem = D_buffer(const.sem_mem_length,const.batch_size,const.num_classes, const.tasks )   
    task_list = train_data()
    optimizer = optim.Adam(list(model_g.parameters()) + list(model_f_w.parameters()), lr=const.base_lr, weight_decay=const.weight_decay)
    criterion = nn.CrossEntropyLoss()
    while mlflow.active_run() is not None:
        mlflow.end_run()
    if mlflow.active_run() is None:
        mlflow.start_run() 
  
    mlflow_log_params()
    train(task_list, sem_mem, model_g, model_f_w, model_f_s,criterion, optimizer )
    torch.save(model_g.state_dict(),const.MODEL_DIR/'model_g' )
    torch.save(model_f_w.state_dict(),const.MODEL_DIR/'model_f_w' )
    torch.save(model_f_s.state_dict(),const.MODEL_DIR/'model_f_s' )
    mlflow.pytorch.log_model(model_g, "g()", registered_model_name="g()")
    mlflow.pytorch.log_model(model_f_s, "f_s()", registered_model_name="f_s()")
    mlflow.pytorch.log_model(model_f_w, "f_w()", registered_model_name= "f_w()")
    mlflow.end_run()