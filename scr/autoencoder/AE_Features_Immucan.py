#%% 
"""
Extract pre-trained features from AE model for Immucan 
"""

from AE import ConvAE
import glob
from natsort import natsorted
import torch
from utils_AE import *
import pickle


MultiCh_Immucan = "/root/workdir/IMMUCan/IMC_Immucan_MultiChROIS_650x650_CD3_SMA_DNA"


device = "cpu"
state_dict ="/root/workdir/IMMUCan/Files_TMA/Features_AE_raw/WeightsBest_model_VAE_10_10_64.pt"

list_images = natsorted(
    glob.glob(os.path.join(MultiCh_Immucan, "**/*.png"), recursive=True)
)


model = ConvAE(channels=3)
model.load_state_dict(torch.load(state_dict, map_location=torch.device("cpu")))
model.to(device)

### images to dataloader 
dataset = ImcDataset(list_images,dataset="Immu")
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=30, shuffle=True, num_workers=0
    )

features = {}
for inddt,data in enumerate(data_loader):    
    if len(data) == 2:                          
        images, names = data["image"], data["name"]
    else:
        images, names,roi_names = data["image"], data["name"], data["roi"]


    data = images.to(device)
    z,decoded_train= model(data)

    for ind,b in enumerate(z):
        p_name = names[ind]
        roi = roi_names[ind]
        p_feat = (z[ind,:,:,:].detach().numpy()).flatten()
        
        try:
            features[p_name][roi] = p_feat
        except:
            features[p_name] = {roi:p_feat}


with open(
    "/root/workdir/IMMUCan/IMC/ImmucanPretrainFeaturesAE/ImmucanPretrainFeatAE_239Pat_CD3_DNA_SMA.pickle",
    "wb",
) as archivo:
    pickle.dump(features, archivo)

