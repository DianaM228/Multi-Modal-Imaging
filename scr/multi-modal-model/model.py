"""Abstract model class."""


import os
import wandb
import numpy as np
import torch
from torch.optim import Adam
from multimodal import MultiModal
from lr_range_test import LRRangeTest
from coach import ModelCoach
from predictor import Predictor
from sub_models import freeze_layers
from loss import Loss
from collections import OrderedDict
from utils_baseline import check_parameters_between_two_models, count_parameters


class _BaseModelWithData:
    """Abstract model with input data."""

    def __init__(
        self,
        dataloaders,
        classes,
        wloss,
        dropout,
        k_init,
        fusion_method=None,
        unimodal_state_files=None,
        freeze_up_to=None,
        device=None,
    ):
        self.fusion_method = fusion_method
        self.dataloaders = dataloaders
        self.classes = classes
        self.wloss = wloss
        self.dropout = dropout
        self.k_init = k_init
        self.unimodal_state_files = unimodal_state_files
        self.freeze_up_to = freeze_up_to
        self.device = device
        eg_dataloader = list(dataloaders.values())[0]  ###
        data_dirs = eg_dataloader.dataset.data_dirs  #####
        self.data_modalities = [
            modality for modality in data_dirs if data_dirs[modality] is not None
        ]  # list of modalities names
        self._instantiate_model()
        self.model_blocks = [name for name, _ in self.model.named_children()]

    def _instantiate_model(self, move_to_device=True):
        print("Instantiating MultiModal model...")
        self.model = MultiModal(
            data_modalities=self.data_modalities,
            classes=self.classes,
            fusion_method=self.fusion_method,
            device=self.device,
            freeze_up_to=self.freeze_up_to,
            dropout=self.dropout,
            k_init=self.k_init
        )

        if self.unimodal_state_files is not None:
            self.pretrained_weights = self._get_pretrained_unimodal_weights()
            print("(loading pretrained unimodal model weights...)")
            self.model.load_state_dict(self.pretrained_weights)

        if move_to_device:
            self.model = self.model.to(self.device)

    def _get_pretrained_unimodal_weights(self):
        for modality in self.data_modalities:
            # Load and collect saved weights
            pretrained_dict = torch.load(self.unimodal_state_files[modality])
            # Filter out unnecessary keys
            model_weight_dict = self.model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in self.model.state_dict()
            }
            # Overwrite entries in the existing state dict
            model_weight_dict.update(pretrained_dict)

        return model_weight_dict


class Model(_BaseModelWithData):
    """Top abstract model class."""

    "unimodal_state_files = state_dict available from trained unimodal models."

    def __init__(
        self,
        dataloaders,
        classes,
        wloss,
        dropout,
        k_init,
        fusion_method="max",
        auxiliary_criterion=None,
        unimodal_state_files=None,
        freeze_up_to=None,
        device=None,
    ):
        super().__init__(
            dataloaders,
            classes,
            wloss,
            dropout,
            k_init,
            fusion_method,
            unimodal_state_files,
            freeze_up_to,
            device,
        )
        self.optimizer = Adam
        self.loss = Loss()
        self.aux_loss = auxiliary_criterion
        self.dropout = dropout
        self.k_init = k_init

    def test_lr_range(self):
        self._instantiate_model()

        self.lr_test = LRRangeTest(
            dataloader=self.dataloaders["train"],            
            optimizer=self.optimizer(
                self.model.parameters(), lr=1e-4, weight_decay=0.001
            ),
            criterion=self.loss,
            auxiliary_criterion=self.aux_loss,
            model=self.model,
            device=self.device,
        )
        self.lr_test.run(init_value=1e-6, final_value=10.0, beta=0.98)

    def plot_lr_range(self, trim=4):
        try:
            self.lr_test.plot(trim)
        except AttributeError as error:
            print(f"Error: {error}.")
            print(f'       Please run {".test_lr_range"} first.')

    def fit(
        self,
        lr,
        num_epochs,
        info_freq,
        log_dir,
        lr_factor=0.1,
        scheduler_patience=5,
        lr_scheduler=None,
        rcrop=None,
        crop=None,
        tile_size=None,
        Orig_size=None,
        n_crops=None,
    ):
        
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        wandb.watch(
            self.model, optimizer, log="all", log_freq=20
        )  # model's weights and biases

        if lr_scheduler is not None:
            print("LR scheduler ACTIVATED ")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=lr_factor,
                patience=scheduler_patience,
                verbose=True,
                threshold=1e-3,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = None

        model_coach = ModelCoach(
            model=self.model,
            dataloaders=self.dataloaders,
            optimizer=optimizer,
            criterion=self.loss,
            auxiliary_criterion=self.aux_loss,
            device=self.device,
            classes=self.classes,
            wloss=self.wloss,
            tile_size=tile_size,
            n_crops=n_crops,
        )

        model_coach.train(
            num_epochs, scheduler, info_freq, log_dir, rcrop
        )  ## train model

        self.model = model_coach.model
        self.best_model_weights = model_coach.best_wts
        self.best_metric_values = model_coach.best_perf
        self.current_metric = model_coach.current_perf

    def save_weights(self, saved_epoch, prefix, weight_dir):
        valid_keys = self.best_model_weights.keys()
        assert saved_epoch in list(valid_keys) + ["current"], (
            f'Valid "saved_epoch" options: {list(valid_keys)}'
            f'\n(use "current" to save current state)'
        )

        print("Saving model weights to file:")
        if saved_epoch == "current":
            epoch = list(self.current_metric.keys())[0]
            value = self.current_metric[epoch]
            file_name = os.path.join(
                weight_dir, f"{prefix}_{epoch}_metric{value:.2f}.pth"
            )
        else:
            file_name = os.path.join(
                weight_dir,
                f"{prefix}_{saved_epoch}_"
                + f"metric{self.best_metric_values[saved_epoch]:.2f}.pth",
            )
            self.model.load_state_dict(self.best_model_weights[saved_epoch])

        torch.save(self.model.state_dict(), file_name)
        print("   ", file_name)

    def load_weights(self, path):
        print("Load model weights:")
        print(path)        
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)

    def predict(self, input_data, rcrop=None):
        predictor = Predictor(self.model, self.device)
        # Use midpoints of MultiSurv output intervals
        prediction = predictor.predict(input_data, rcrop=rcrop)
        feature_representations, label = prediction

        return feature_representations, label

    def predict_dataset(self, dataset, verbose=True):
        predictor = Predictor(self.model, self.device)

        return predictor.predict_dataset(dataset, verbose)
