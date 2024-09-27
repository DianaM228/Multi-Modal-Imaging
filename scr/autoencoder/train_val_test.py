# -*- coding:utf-8 -*-
import os
import time
import torch
import wandb
import torch.nn.functional as F
from utils_AE import LRScheduler
import random
import numpy as np
from scripts.AE.AE import frange_cycle_linear

def process_data_batch(
        data,
        phase,
        model,
        optimizer,
        loss_fn,
        beta_weight,
        g_clip,
        normalize,
        model_type,
        device,
        reduction,
        b_scheduler,
        epoch=None
        ):        
    with torch.set_grad_enabled(phase == "train"):    
        batchsize,ch,h,w = data.shape

        # FORWARD AND BACK PROP        

        if model_type =="VAE":
            decoded, z_mean, z_log_var, z = model(data)
            # total loss = reconstruction loss + KL divergence
            kl_div = -0.5 * torch.sum(
                1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
            )  # sum over latent dimension                    
            
            kl_div = kl_div.mean()  # average over batch dimension
            # Norm kl_loss with z size
            if normalize:
                div_kdiv = z_mean.size(1) # z size
                kl_div=kl_div/div_kdiv

            if b_scheduler is not None:
                beta_weight = b_scheduler[epoch]
                #print(beta_weight)

            kl_div=beta_weight * kl_div
        else:
            _,decoded= model(data)
            kl_div=torch.tensor([0]).to(device) ### No KL_loss when using AE

        if isinstance(loss_fn, torch.nn.modules.loss.BCELoss):            
            pixelwise = loss_fn(decoded.view(batchsize, -1),data.view(batchsize, -1))
            pixelwise = pixelwise*h*w*ch            

        else:
            if reduction == "sum":
                pixelwise = loss_fn(decoded, data, reduction="none")       
                pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) 
                pixelwise = pixelwise.mean()
            elif reduction == "mean":
                pixelwise = loss_fn(decoded, data)
                
        
        if normalize:
            pixelwise = pixelwise/(h*w*ch)     
                   
        loss =  pixelwise + kl_div          

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            if g_clip:
                max_grad_norm = 1.0  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
                
    return pixelwise.item(),kl_div.item(), loss.item()

def train_vae(
    num_epochs,
    model,
    optimizer,
    device,
    train_loader,
    val_loader,       
    beta_weight=1,
    model_type="VAE",
    loss_fn=None, 
    save_model=None,
    g_clip=None,
    Parameters=None,
    normalize=None,
    lr_schedule=None,
    dropout=None,
    rcrop = None,
    reduction="sum",
    b_scheduler=None
):   

    log_dict = {
        "train_combined_loss_per_batch": [],
        "val_combined_loss_per_batch": [],
        "train_reconstruction_loss_per_batch": [],
        "train_kl_loss_per_batch": [],
    }

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "BCE":
        #loss_fn = F.cross_entropy
        loss_fn = torch.nn.BCELoss()

    if lr_schedule:
        lr_scheduler = LRScheduler(optimizer)
    
    if b_scheduler:
       b_schedule = frange_cycle_linear(num_epochs)
    else:
       b_schedule = b_scheduler



    start_time = time.time()
    wandb.watch(model, optimizer, log="all", log_freq=20)
    wandb.watch(model, log="gradients")

    def print_header():
        sub_header = " Epoch      Loss        Met               Loss           Met"
        print("-" * (len(sub_header) + 6))
        print("              Training                       Validation")
        print("           ------------------               ------------------")
        print(sub_header)
        print("-" * (len(sub_header) + 6))

    print()

    print_header()

    best_loss = 100000000    
    for epoch in range(num_epochs):
        for phase in ["train","val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

        
            running_rec_losses = []
            running_kl_losses = []
            running_all_losses = []
            for batch_idx, data in enumerate(loader): 
                if len(data) == 2:                          
                    images, names = data["image"], data["name"]
                else:
                    images, names,roi_name = data["image"], data["name"], data["roi"]

                if rcrop:        
                    p_size = 200                    
                    bz,_,w,h = images.shape

                    coo = [
                            [
                                random.randint(0, w - p_size),
                                random.randint(0, h - p_size),
                            ]
                            for i in range(rcrop)
                        ]
                    
                    coo = np.array(coo)

                    new_names = []
                    new_batch = []
                    for bi in range(bz):
                        example = images[bi].squeeze()
                        new_p_batch = np.stack(
                                    [
                                        example[
                                            :,
                                            i[1] : (i[1] + p_size),
                                            i[0] : (i[0] + p_size),
                                        ]
                                        for i in coo
                                    ]
                                )
                        
                        new_batch.append(new_p_batch)
                        ## replicate names
                        new_names = new_names + [names[bi]] * rcrop

                    new_batch = np.concatenate(new_batch, axis=0)
                    
                    data = torch.from_numpy(new_batch).to(device)
                else:
                    data = images.to(device)
                
                rec_loss, kl_loss, comb_loss = process_data_batch(data,
                                                                  phase,
                                                                  model,
                                                                  optimizer,
                                                                  loss_fn,
                                                                  beta_weight,
                                                                  g_clip,
                                                                  normalize,
                                                                  model_type,
                                                                  device,
                                                                  reduction, 
                                                                  b_schedule,
                                                                  epoch                                                                 
                                                                  )

                running_rec_losses.append(rec_loss)
                running_kl_losses.append(kl_loss)
                running_all_losses.append(comb_loss)

                # LOGGING
                if phase == "train":
                    log_dict["train_combined_loss_per_batch"].append(comb_loss)
                    log_dict["train_reconstruction_loss_per_batch"].append(comb_loss)
                    log_dict["train_kl_loss_per_batch"].append(kl_loss)
                elif phase == "val":
                    log_dict["val_combined_loss_per_batch"].append(comb_loss)


            epoch_rec = torch.mean(torch.tensor(running_rec_losses))
            epoch_kl = torch.mean(torch.tensor(running_kl_losses).to(torch.float))
            epoch_all = torch.mean(torch.tensor(running_all_losses))

      
            if phase == "train":
                wandb.log({"train_loss_rec": epoch_rec}, step=epoch + 1)
                wandb.log({"train_loss_kl": epoch_kl}, step=epoch + 1)
                wandb.log({"train_loss_combined": epoch_all}, step=epoch + 1)
                
                message = f" {epoch}/{num_epochs}"
            space = 8 if phase == "train" else 40
            message += " " * (space - len(message))
            message += f"{epoch_all:.2f}"
            space = 22 if phase == "train" else 56
            message += " " * (space - len(message))
            message += f"{epoch_rec:.2f}"
            if phase == "val":     
                if lr_schedule:
                    lr_scheduler(epoch_all)   
                            
                print(message)

                wandb.log({"val_loss_rec": epoch_rec}, step=epoch + 1)
                wandb.log({"val_loss_kl": epoch_kl}, step=epoch + 1)
                wandb.log({"val_loss_combined": epoch_all}, step=epoch + 1)
            
                if epoch_all < best_loss:
                    checkpoint = {'Parameters': Parameters,# Experiment parameters
                                      'state_dict': model.state_dict(),# Model layers and weights
                                      'optimizer': optimizer.state_dict(),# Optimizer                                      
                                      'loss_val': epoch_all,# Average loss history in validation set each epoch                                                                           
                                      'epoch':epoch,
                                      
                                      }
                    torch.save(checkpoint,os.path.join(os.path.dirname(save_model),'BestcheckPoint.pth.tar'))

                    torch.save(
                        model.state_dict(),
                        os.path.join(os.path.dirname(save_model), "Best_model_AE.pt"),
                    )

        if epoch == num_epochs-1:
            torch.save(model.state_dict(), save_model)
    print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))
    return log_dict

