#%% 
"""
Extract pre-trained features from Multi-modal model for Immucan 
"""

from model import Model
import utils_baseline
import pandas as pd
import os
import numpy as np
import glob 

DATA = "/root/workdir/IMMUCan/IMC_Immucan_650x650_CD3_SMA_DNA"


weights= "./CD3_SMA_DNA2_init_lr0.0013970419945925315_epoch483_metric0.90.pth"

data_modalities = ["CD3","DNA2","SMA"]

save_path = os.path.join(os.path.dirname(weights),"Predictions_Immucan")

batch = 80
freez = bool(0)
classes = 2
fusion_method = "sum"
k_init =None
wloss = bool(0)


utils_baseline.seed_everything()
path = "/root/workdir/IMMUCan/IMC_Immucan_650x650_CD3_SMA_DNA/CD3"
patients = os.listdir(path)

pat_and_rois = glob.glob(path + os.sep + "**" + os.sep + "*.png", recursive=True)

patients = [os.path.basename(i).split(".")[0] for i in pat_and_rois]

train_data = pd.DataFrame({"name": patients, "label": list(np.zeros(len(patients),dtype=int))})
val_data = pd.DataFrame({"name": patients, "label": list(np.zeros(len(patients),dtype=int))})
test_data = pd.DataFrame({"name": patients, "label": list(np.zeros(len(patients),dtype=int))})

# To follow multi-modal model dic structure
data_dir = {
            "data": {"fold0": {"train": train_data, "val": val_data}},
        }
data_dir["test"] = test_data

data_tvt = {
    "train": data_dir["data"]["fold0"]["train"],
    "val": data_dir["data"]["fold0"]["val"],
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
    database = "Immucan"
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
# it saves a file with the features (fused and from classifier)
performance.compute_metrics(
    path_save=save_path,classes=2
)

