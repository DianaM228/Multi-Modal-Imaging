"""Performance evaluation."""

import warnings
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.utils import resample
from sklearn.metrics import f1_score
from torcheval.metrics import MulticlassF1Score
from torcheval.metrics import MulticlassAccuracy
from torchmetrics import F1Score
from torchmetrics import Accuracy
from torcheval.metrics.functional import multiclass_precision
import json
import utils_baseline
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Evaluation:
    def __init__(
        self, model, dataset, device=None, rcrop=None, tile_size=None, n_crops=None
    ):
        self.model = model
        self.device = device
        self.data = dataset
        self.patient_predictions = None
        self.rcrop = rcrop
        self.tile_size = (tile_size,)
        self.n_crops = n_crops

    def _collect_patient_ids(self):
        return self.data.patient_ids

    def _data_to_device(self, data):
        for modality in data:
            data[modality] = data[modality].to(self.device)
        return data

    def _predict(self, patient):
        data, label = self.data.get_patient_data(patient)
        if self.rcrop:
            p_size = self.tile_size
            num_patches = self.n_crops
            modalities = list(data.keys())
            modalities_data = data
            _, _, w, h = modalities_data[modalities[0]].shape

            random.seed(0)
            coo = [
                [
                    random.randint(0, w - p_size[0]),
                    random.randint(0, h - p_size[0]),
                ]
                for i in range(num_patches)
            ]
            coo = np.array(coo)

            for ind_m, modality in enumerate(modalities):
                tensor = modalities_data[modality]
                example = tensor.squeeze()
                new_p_batch = np.stack(
                    [
                        example[
                            :,
                            i[1] : (i[1] + p_size[0]),
                            i[0] : (i[0] + p_size[0]),
                        ]
                        for i in coo
                    ]
                )
                new_p_batch = np.expand_dims(new_p_batch, axis=1)

                data[modality] = torch.from_numpy(new_p_batch)

        data = self._data_to_device(data)
        features, preds = self.model.predict(data, rcrop=self.rcrop)
        

        return label, preds, features

    def _collect_patient_predictions(self):
        # Get all patient labels and predictions
        patient_data = {}
        pids = self._collect_patient_ids()
        for i, patient in enumerate(pids):
            print(
                f"\r"
                + f"Collect patient predictions:"
                + f" {str((i + 1))}/{len(pids)}",
                end="",
            )

            label, pred, features = self._predict(patient)
            patient_data[patient] = {
                "label": label,
                "probabilities": pred,
                "features": features,
            }

        print()
        print()

        return patient_data

    def _unpack_data(self, data):
        labels = [data[patient]["label"] for patient in data]
        predictions = [data[patient]["probabilities"] for patient in data]
        pred = torch.stack(predictions, dim=0)
        pred = torch.squeeze(pred, dim=1)
        lab = torch.Tensor(labels)

        return lab, pred

    def compute_metrics(self, path_save, classes, set_data="test"):
        """Calculate evaluation metrics."""
        if self.patient_predictions is None:
            # Get all patient labels and predictions
            self.patient_predictions = self._collect_patient_predictions()
            labels, predictions = self._unpack_data(self.patient_predictions)
            torch.save(
                self.patient_predictions,
                os.path.join(path_save, "predictions_emd_" + set_data + ".pth"),
            )

        #######
        if classes == 2:
            predictions = torch.sigmoid(predictions)
            threshold = 0.5  # Umbral para clasificación binaria
            binary_predictions = (predictions >= threshold).int()

            if self.rcrop:
                ## majority voting
                binary_predictions = torch.tensor(
                    [
                        torch.mode(bp.squeeze()).values.item()
                        for bp in binary_predictions
                    ]
                )
                
            correct = (
                (
                    (binary_predictions.squeeze().to(self.device))
                    == labels.to(self.device)
                )
                .sum()
                .item()
            )
            accuracy = correct / len(labels)
            print("Acc: ", accuracy)
            print("Confusion Matrix")
            cm = confusion_matrix(labels.cpu(), binary_predictions.squeeze().cpu())
            print(cm)
            print("Recall/Sens-TP: ", cm[1, 1] / (cm[1, 1] + cm[1, 0]))
            print("Specificity-TN : ", cm[0, 0] / (cm[0, 0] + cm[0, 1]))
            print(
                "f1: ", f1_score(binary_predictions.cpu().numpy(), labels.cpu().numpy())
            )
            print("precision", cm[1, 1] / (cm[1, 1] + cm[0, 1]))

            #### Save ROC
            if not self.rcrop:
                fpr, tpr, thresholds = roc_curve(
                    labels.cpu().numpy(), predictions.cpu().numpy()
                )
                roc_auc = auc(fpr, tpr)
                print("AUC: ", roc_auc)

                plt.figure()
                plt.plot(
                    fpr, tpr, color="darkorange", label="ROC (area = %0.2f)" % roc_auc
                )
                plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positives rate")
                plt.ylabel("True Positives rate")
                plt.title("ROC curve")
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(path_save, set_data + "_ROC.png"))

        elif classes > 2:
            _, multiclass_predictions = torch.max(predictions, dim=1)
            correct = (multiclass_predictions == labels.to(self.device)).sum().item()
            accuracy = correct / len(labels)
            print("Acc: ", accuracy)
            print("Confusion Matrix")
            cm = confusion_matrix(labels, multiclass_predictions.cpu())
            print(cm)
            y_true_onehot = F.one_hot(labels, classes).numpy()
            y_pred_onehot = F.one_hot(multiclass_predictions, classes).numpy()
            f1_multiclass = f1_score(y_true_onehot, y_pred_onehot, average="weighted")
            print("f1: ", f1_multiclass)

            class_counts = np.bincount(labels.numpy(), minlength=classes)
            class_frequency = class_counts / np.sum(class_counts)
            class_recall = np.diag(cm) / np.sum(cm, axis=1)
            weighted_recall = np.average(class_recall, weights=class_frequency)
            print("weighted Recall: ", weighted_recall)
            class_specificity = []
            for i in range(classes):
                mask = np.ones(classes, dtype=bool)
                mask[i] = False
                non_class_samples = np.sum(cm[mask, :][:, mask])
                true_negatives = non_class_samples - np.sum(cm[mask, i])
                class_specificity.append(true_negatives / non_class_samples)

            weighted_specificity = np.average(
                class_specificity, weights=class_frequency
            )
            print("Specificity : ", weighted_specificity)
        return accuracy
