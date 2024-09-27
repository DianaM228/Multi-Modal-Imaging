#%%  Prediction with best multi-modal model saved


import torch
from model import Model
import utils_baseline
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

DATA = "/root/workdir/IMMUCan/IMC_TMA_650x650_norm"


weights= "/root/workdir/IMMUCan/BestParamPathOversBatch80cv5Shuffletvo/CD3_SMA_DNA2_init_lr0.0013970419945925315/fold3/CD3_SMA_DNA2_init_lr0.0013970419945925315_epoch190_metric0.79.pth"
fold = "fold3"
task = "histology" # "stage"   "histology"

data_modalities = ["CD3","DNA2","SMA"]#["CD3","DNA2","SMA"]

if task == "histology":
    labels_file_path = (
        "/root/workdir/IMMUCan/Files_TMA/Files_DL/Subset_TMA_DL_PathAdenVsSqu2.xlsx"
    )
elif task == "stage":
    labels_file_path = (
        "/root/workdir/IMMUCan/Files_TMA/Files_DL/Subset_TMA_DL_Stage2classes.xlsx"
    )

save_path = os.path.join(os.path.dirname(weights),"Predictions_feat")

batch = 80
freez = bool(0)
classes = 2
fusion_method = "sum"
k_init =None
wloss = bool(0)


utils_baseline.seed_everything()

## Function to split data into train-validation using oversampling of minoritary class
data_dir = utils_baseline.train_val_data(
                    labels_file_path, validation="cv", oversampling=True
                )


data_tvt = {
    "train": data_dir["data"][fold]["train"],
    "val": data_dir["data"][fold]["val"],
    "test": data_dir["test"],
}


dataloaders = utils_baseline.get_dataloaders(
    data_location=DATA,
    modalities=data_modalities,
    data=data_tvt,
    wsi_patch_size=650,
    n_wsi_patches=1,
    transform=False,
    exclude_patients=None,
    batch_size=batch,
    return_patient_id=True,
)

multimodal = Model(
    dataloaders=dataloaders,
    classes=classes,
    fusion_method=fusion_method,
    device="cpu",
    freeze_up_to=freez,    
    dropout= None,
    k_init =k_init,
    wloss = wloss
)

multimodal.load_weights(weights)

if not os.path.isdir(save_path):
   os.makedirs(save_path)

performance = utils_baseline.Evaluation(
    model=multimodal, dataset=dataloaders["val"].dataset, device="cpu"
)
performance.compute_metrics(
    path_save=save_path,classes=2
)


### load predictions 
results = torch.load(os.path.join(save_path,"predictions_emd_test.pth"))


#### justo to check because the results are given by performance.compute_metrics
labels = []
pred = []
for key,val in results.items():
    pred.append(val['probabilities'].item())
    labels.append(val['label'])

labels=torch.tensor(labels)
predictions = torch.sigmoid(torch.tensor(pred))
threshold = 0.5  # Umbral para clasificación binaria
binary_predictions = (predictions >= threshold).int()
correct = (binary_predictions.squeeze() == labels).sum().item()
accuracy = correct / len(labels)
print("Acc:" ,accuracy)

cm = confusion_matrix(labels, binary_predictions.squeeze().cpu())
print(cm)
print("Recall/Sens-TP: ", cm[1, 1] / (cm[1, 1] + cm[1, 0]))
print("Specificity-TN : ", cm[0, 0] / (cm[0, 0] + cm[0, 1]))
print(
    "f1: ", f1_score(binary_predictions.cpu().numpy(), labels.cpu().numpy())
)
print("precision", cm[1, 1] / (cm[1, 1] + cm[0, 1]))

#### Save ROC
fpr, tpr, thresholds = roc_curve(labels.numpy(),predictions.numpy())
roc_auc = auc(fpr, tpr)            
print("AUC: ",roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positives rate')
plt.ylabel('True Positives rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_path,"ROC.png"))
                         


# %%
