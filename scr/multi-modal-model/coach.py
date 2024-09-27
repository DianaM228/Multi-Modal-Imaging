"""Abstract model class."""


import os
import sys
import warnings
import time
import copy
from itertools import combinations
import math
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import torch
from torcheval.metrics import MulticlassF1Score
from torcheval.metrics import MulticlassAccuracy
from torchmetrics import F1Score
from torchmetrics import Accuracy
import wandb
import utils_baseline
import random
import numpy as np


class ModelCoach:
    """Model fitting functionality."""

    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        criterion,
        auxiliary_criterion,
        device=None,
        classes=None,
        wloss=None,
        tile_size=None,
        n_crops=None,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.aux_criterion = (
            auxiliary_criterion.to(device)
            if auxiliary_criterion is not None
            else auxiliary_criterion
        )
        self.classes = classes
        self.wloss = wloss
        self.tile_size = tile_size
        self.n_crops = n_crops
        # Save 3 best model weights
        self.best_perf = {"epoch a": 0.0, "epoch c": 0.0, "epoch b": 0.0}
        self.best_wts = {"epoch a": None, "epoch c": None, "epoch b": None}
        self.current_perf = {"epoch a": 0}
        self.device = device
        self.best_metric = 0

    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device) for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def _compute_auxiliary_loss(self, features):
        "Embedding vector distance loss."
        losses = []

        y = torch.ones(1).to(self.device)
        for x1, x2 in combinations(features, 2):
            losses.append(self.aux_criterion(x1, x2, y))

        loss = torch.tensor(losses).mean()

        return loss.to(self.device)

    def _compute_metric(
        self, labels, predictions, metric="acc", rcrop=None, n_crops=None
    ):

        if metric == "acc":
            if self.classes == 2:
                predictions = torch.sigmoid(predictions)
                threshold = 0.5  # Umbral para clasificación binaria
                binary_predictions = (predictions >= threshold).int()
                if rcrop:
                    n_pred = []
                    n_lab = []
                    for bp in range(int(len(binary_predictions) / n_crops)):
                        ## majority voting
                        n_pred.append(
                            torch.mode(
                                binary_predictions[
                                    bp * n_crops : (bp + 1) * n_crops
                                ].squeeze()
                            ).values.item()
                        )
                        """# If at least 1 crop predicted as 1, the gloabl patient labels is 1
                        n_pred.append(
                            int(
                                torch.any(
                                    binary_predictions[
                                        bp * n_crops : (bp + 1) * n_crops
                                    ].squeeze()
                                    == 1
                                )
                            )
                        )"""
                        n_lab.append(
                            labels[bp * n_crops : (bp + 1) * n_crops][0].item()
                        )

                    labels = torch.tensor(n_lab)
                    binary_predictions = torch.tensor(n_pred)

                correct = (
                    (binary_predictions.squeeze().cpu() == labels.cpu()).sum().item()
                )
                accuracy = correct / len(labels)

            elif self.classes > 2:
                _, multiclass_predictions = torch.max(predictions, dim=1)
                correct = (multiclass_predictions == labels).sum().item()
                accuracy = correct / len(labels)

        else:
            sys.exit("evaluation metric not implemented.")

        return torch.tensor(accuracy).to(self.device)

    def _compute_loss(self, pred, labels, modality_features):
        loss = self.criterion(
            pred=pred,
            target=labels,
            classes=self.classes,
            wloss=self.wloss,
            device=self.device,
        )

        is_multimodal = len(self.model.data_modalities) > 1

        if not is_multimodal and self.aux_criterion is not None:
            warnings.warn(
                "Input data is unimodal: auxiliary" + " loss is not applicable."
            )

        if is_multimodal and self.aux_criterion is not None:
            # Embedding vector distance loss
            auxiliary_loss = self._compute_auxiliary_loss(modality_features)
            loss = (1.0 * auxiliary_loss) + (0.05 * loss)

        return loss

    def _log_info(self, phase, logger, epoch, epoch_loss, epoch_met):
        info = {phase + "_loss": epoch_loss, phase + "_metric": epoch_met}

        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)
            wandb.log({tag: value}, step=epoch)

    def _process_data_batch(
        self, data, phase, retain_graph=None, rcrop=None, n_crops=None
    ):
        if len(data) == 2:
            (
                data,
                labels,
            ) = data  
        elif len(data) == 3:  
            data, labels, patient_id = data
        data = self._data_to_device(data)
        labels = labels.to(self.device)

        with torch.set_grad_enabled(phase == "train"):
            feature_representations, pred = self.model(data)
            modality_features = feature_representations["modalities"]
            loss = self._compute_loss(
                pred, labels, modality_features
            )  # modality_features is needed for auxiliary loss

            if phase == "val" and rcrop == True: 
                met = self._compute_metric(
                    labels, pred, metric="acc", rcrop=True, n_crops=n_crops
                )
            else:
                met = self._compute_metric(
                    labels, pred, metric="acc", rcrop=False, n_crops=n_crops
                )

            # print("D loss", loss)
            # print("D Metric", met)
            if phase == "train":
                # Zero out parameter gradients
                self.optimizer.zero_grad()
                if retain_graph is not None:
                    loss.backward(retain_graph=retain_graph)
                else:
                    loss.backward()
                self.optimizer.step()
                

        return loss, met, pred 

    def _run_training_loop(self, num_epochs, scheduler, info_freq, log_dir, rcrop):
        logger = SummaryWriter(log_dir)
        log_info = True

        if info_freq is not None:

            def print_header():
                sub_header = " Epoch     Loss     Met     Loss     Met"
                print("-" * (len(sub_header) + 2))
                print("             Training        Validation")
                print("           ------------     ------------")
                print(sub_header)
                print("-" * (len(sub_header) + 2))

            print()

            print_header()

        for epoch in range(1, num_epochs + 1):
            # print("EPOCH:",epoch)
            if info_freq is None:
                print_info = False
            else:
                print_info = epoch == 1 or epoch % info_freq == 0

            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_losses = []
                running_met = []

                ################ CHECK PARAMS CHANGE
                # params = list(self.model.HE_submodel.parameters())
                # mean_param_b = torch.mean(torch.cat([param.view(-1) for param in params]))
                ####################################
                self.dataloaders["train"].dataset.epoch = epoch
                self.dataloaders["val"].dataset.epoch = epoch
                self.dataloaders["test"].dataset.epoch = num_epochs

                self.dataloaders["train"].dataset.phase = phase
                self.dataloaders["val"].dataset.phase = phase
                self.dataloaders["test"].dataset.phase = phase

                # Iterate over data
                for data in self.dataloaders[
                    phase
                ]:  # Take a batch patients for train or val                    
                    if rcrop:
                        ### crop n patches randomly
                        p_size = self.tile_size
                        num_patches = self.n_crops
                        modalities = list(data[0].keys())
                        modalities_data = data[0]
                        labels_ = data[1]
                        names = data[2]
                        batch_size, _, _, w, h = modalities_data[modalities[0]].shape

                       
                        ### Exclude border 
                        max_x = w - p_size - 25
                        max_y = h - p_size - 50
                        
                        coo = [
                            [
                                np.random.randint(25, max_x + 1),
                                np.random.randint(50, max_y + 1),
                            ]
                            for i in range(num_patches)
                        ]


                        coo = np.array(coo)

                        new_labels = []
                        new_names = []
                        for ind_m, modality in enumerate(modalities):
                            tensor = modalities_data[modality]
                            new_batch = []
                            for ind_b, bi in enumerate(range(batch_size)):
                                example = tensor[bi].squeeze()
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
                                new_p_batch = np.expand_dims(new_p_batch, axis=1)
                                new_batch.append(new_p_batch)
                                ## replicate labels only once
                                if ind_m == 0:
                                    new_labels = (
                                        new_labels
                                        + [labels_[ind_b].item()] * num_patches
                                    )
                                    new_names = new_names + [names[ind_b]] * num_patches

                            new_batch = np.concatenate(new_batch, axis=0)
                            # Debug
                            """import matplotlib.image
                            for ind,i in enumerate(new_batch):
                            matplotlib.image.imsave("/runs/Pitzalis_Data_Fusion-375804/scripts/p"+str(ind)+".png",i[0,0,:,:])
                            """
                            data[0][modality] = torch.from_numpy(new_batch)
                        data[1] = torch.tensor(new_labels)
                        data[2] = tuple(new_names)

                    batch_result = self._process_data_batch(
                        data, phase, rcrop=rcrop, n_crops=self.n_crops
                    )
                    loss, met, pred = batch_result

                    # Stats
                    running_losses.append(loss.item())
                    running_met.append(met.item())

                    ################ CHECK PARAMS CHANGE
                    # params = list(self.model.HE_submodel.parameters())
                    # mean_param_a = torch.mean(torch.cat([param.view(-1) for param in params]))
                    # if mean_param_b != mean_param_a:
                    #    print("the parameters of HE backbone changed")
                    ####################################

                epoch_loss = torch.mean(torch.tensor(running_losses))
                epoch_met = torch.mean(torch.tensor(running_met))

                if print_info:
                    if phase == "train":
                        message = f" {epoch}/{num_epochs}"
                    space = 10 if phase == "train" else 27
                    message += " " * (space - len(message))
                    message += f"{epoch_loss:.4f}"
                    space = 19 if phase == "train" else 36
                    message += " " * (space - len(message))
                    message += f"{epoch_met:.3f}"
                    if phase == "val":
                        print(message)

                if log_info:
                    self._log_info(
                        phase=phase,
                        logger=logger,
                        epoch=epoch,
                        epoch_loss=epoch_loss,
                        epoch_met=epoch_met,
                    )

                if phase == "val":
                    if scheduler:
                        act = self.optimizer.param_groups[0]["lr"]
                        scheduler.step(
                            epoch_loss
                        )  ###### drecrese lr if criterion is met
                        new = self.optimizer.param_groups[0]["lr"]
                        if act != new:
                            print("Epoch {}, lr {}".format(epoch, new))                   

                    # Record current performance
                    k = list(self.current_perf.keys())[0]
                    self.current_perf["epoch" + str(epoch)] = self.current_perf.pop(k)
                    self.current_perf["epoch" + str(epoch)] = epoch_met
                    # Deep copy the model
                    for k, v in self.best_perf.items():
                        if epoch_met >= v:
                            self.best_perf["epoch" + str(epoch)] = self.best_perf.pop(k)
                            self.best_perf["epoch" + str(epoch)] = epoch_met
                            self.best_wts["epoch" + str(epoch)] = self.best_wts.pop(k)
                            self.best_wts["epoch" + str(epoch)] = copy.deepcopy(
                                self.model.state_dict()
                            )
                            break

    def train(self, num_epochs, scheduler, info_freq, log_dir, rcrop=None):
        """Train multimodal PyTorch model."""
        start_time = time.time()

        # Handle keyboard interrupt
        try:
            self._run_training_loop(num_epochs, scheduler, info_freq, log_dir, rcrop)

            hrs, mins, secs = utils_baseline.elapsed_time(start_time)
            print()
            message = ">>>>> Training completed in"
            if hrs > 0:
                message += f" {hrs}h"
            if mins > 0:
                message += f" {mins}m"
            print(message + f" {secs}s")
            print(">>>>> Best validation metrics:")
            for k, v in self.best_perf.items():
                print(f"     {v} ({k})")
        except KeyboardInterrupt:
            hrs, mins, secs = utils_baseline.elapsed_time(start_time)
            print()
            print(">>> Keyboard interrupt! <<<")
            print(f"(trained for {hrs}h {mins}m {secs}s)")
            print()
            print("Best validation concordance values:")
            for k, v in self.best_perf.items():
                print(f"     {round(v, 4)} ({k})")
