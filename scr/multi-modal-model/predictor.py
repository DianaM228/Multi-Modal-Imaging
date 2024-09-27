"""Model predictions."""


import warnings

import torch
import numpy as np


class Predictor:
    """Add prediction functionality."""

    def __init__(self, model, device):
        self.model = model
        self.device = device        

    def _check_dropout(self, dataset):
        dropout = dataset.dropout
        if dropout > 0:
            warnings.warn(f"Data dropout set to {dropout} in input dataset")

    def _data_to_device(self, data):
        for modality in data:
            data[modality] = data[modality].to(self.device)
        return data

    def _clone(self, data):
        data_clone = {}

        for modality in data:
            data_clone[modality] = data[modality].clone()

        return data_clone

    def predict(self, patient_data,rcrop=None):
        """Predict patient pathotype"""
        data = self._clone(patient_data)

        # Model expects batch dimension
        if not rcrop:
            for modality in data:
                data[modality] = data[modality].unsqueeze(0)

        data = self._data_to_device(data)
        
        self.model.eval()
        
        with torch.set_grad_enabled(False):
            feature_representations, probabilities = self.model(data)

        return feature_representations, probabilities

    def predict_dataset(self, dataset, verbose=True):
        """Predict pathotype for provided set of patients."""
        self._check_dropout(dataset)

        if verbose:
            print("Analyzing patients")
            n = len(dataset)

        pids = []
        result = {}
        real_labels = []
        probabilities = []

        # Get all patient data and predictions
        for i, patient in enumerate(dataset):
            if verbose:
                print("\r" + f"{str((i + 1))}/{n}", end="")
            
            data, label, patient_id = patient
            pids.append(patient_id)
            real_labels.append(label)

            data = self._data_to_device(data)
            _, predictions = self.predict(data)

            probabilities.append(predictions)

        result["patient_data"] = {}

        for i, pid in enumerate(pids):
            result["patient_data"][pid] = probabilities[i]

        return result
