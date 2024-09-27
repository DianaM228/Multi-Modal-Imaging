# Normalize and scale TMA dataset """

#%%  Compute quantile per channel to normalize images
import pandas as pd

file = "./cells_meanIntensity.csv"
single_cell = pd.read_csv(file, sep=";")


quantiles_2_5 = single_cell.quantile(q=0.025).to_frame()
quantiles_97_5 = single_cell.quantile(q=0.975).to_frame()

quantiles_means_CSI = {}
c = 0
for ind, row in quantiles_2_5.iterrows():
    quantiles_means_CSI[row.name] = {
        "q1": round(row.values[0], 2),
        "q2": round(quantiles_97_5.iloc[c].values[0], 2),
    }
    c += 1

quantiles_means_CSI = {
    "SMA": {"q1": 0.15, "q2": 14.99},
    "Vimentin": {"q1": 0.09, "q2": 21.88},
    "CD14": {"q1": 0.6, "q2": 8.0},
    "CD16": {"q1": 0.1, "q2": 2.55},
    "CD163": {"q1": 0.02, "q2": 1.78},
    "Cytok": {"q1": 0.14, "q2": 57.68},
    "CCR4": {"q1": 0.09, "q2": 2.19},
    "CD63": {"q1": 0.32, "q2": 5.74},
    "CD31": {"q1": 0.05, "q2": 1.83},
    "CD45": {"q1": 0.06, "q2": 6.47},
    "FoxP3": {"q1": 0.08, "q2": 1.15},
    "CD4": {"q1": 0.05, "q2": 1.3},
    "CD68": {"q1": 0.15, "q2": 13.09},
    "C1Qa": {"q1": 0.07, "q2": 1.15},
    "CD20": {"q1": 0.0, "q2": 0.21},
    "CD8": {"q1": 0.03, "q2": 2.49},
    "Arginase1": {"q1": 0.08, "q2": 1.5},
    "S100A8": {"q1": 0.26, "q2": 186.62},
    "ki67": {"q1": 0.07, "q2": 10.93},
    "Coll_I": {"q1": 0.06, "q2": 10.46},
    "CD3": {"q1": 0.11, "q2": 2.51},
    "CD66a": {"q1": 0.06, "q2": 73.63},
    "HLA-DR": {"q1": 0.21, "q2": 22.24},
}


#%% Normalize, rescale (650x650) and save images for deep learning (Multi-modal model)
import pandas as pd
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from skimage.transform import rescale,resize
import imageio
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


quantiles_means_CSI = {
    "SMA": {"q1": 0.15, "q2": 14.99},
    "Vimentin": {"q1": 0.09, "q2": 21.88},
    "CD14": {"q1": 0.6, "q2": 8.0},
    "CD16": {"q1": 0.1, "q2": 2.55},
    "CD163": {"q1": 0.02, "q2": 1.78},
    "Cytok": {"q1": 0.14, "q2": 57.68},
    "CCR4": {"q1": 0.09, "q2": 2.19},
    "CD63": {"q1": 0.32, "q2": 5.74},
    "CD31": {"q1": 0.05, "q2": 1.83},
    "CD45": {"q1": 0.06, "q2": 6.47},
    "FoxP3": {"q1": 0.08, "q2": 1.15},
    "CD4": {"q1": 0.05, "q2": 1.3},
    "CD68": {"q1": 0.15, "q2": 13.09},
    "C1Qa": {"q1": 0.07, "q2": 1.15},
    "CD20": {"q1": 0.0, "q2": 0.21},
    "CD8": {"q1": 0.03, "q2": 2.49},
    "Arginase1": {"q1": 0.08, "q2": 1.5},
    "S100A8": {"q1": 0.26, "q2": 186.62},
    "ki67": {"q1": 0.07, "q2": 10.93},
    "Coll_I": {"q1": 0.06, "q2": 10.46},
    "CD3": {"q1": 0.11, "q2": 2.51},
    "CD66a": {"q1": 0.06, "q2": 73.63},
    "HLA-DR": {"q1": 0.21, "q2": 22.24},
    "DNA2": {"q1": 4.45, "q2": 29.32},
}


path_images = "/root/workdir/IMMUCan/DATA_TMA"
file = "/root/workdir/IMMUCan/Files_TMA/SubsetInformation/Subset_TMA_DL.xlsx"
df = pd.read_excel(file)
output = "/root/workdir/IMMUCan/IMC_TMA_650x650_norm"

markersCh = {"SMA": 5, "DNA2": 33, "CD3": 29}

for ind, sample in df.iterrows():
    im_name = sample["Image"]
    path_im = os.path.join(path_images, im_name + ".tiff")
    with tifffile.TiffFile(path_im) as tif:
        pyramid = list(reversed(sorted(tif.series, key=lambda p: p.size)))
        size = pyramid[0].size
        pyramid = [p for p in pyramid if size % p.size == 0]
        pyramid = [p.asarray() for p in pyramid]
    image = pyramid[0]

    ## Normalize the image
    for m_name, m_ch in markersCh.items():
        # extract channel
        one_ch_list = image[m_ch, :, :]
        # normalize image
        q1all = quantiles_means_CSI[m_name]["q1"]
        q2all = quantiles_means_CSI[m_name]["q2"]
        norm_one_ch = (one_ch_list - q1all) / (q2all - q1all)
        norm_one_ch = np.clip(norm_one_ch, 0, 1)
                
        r_image = rescale(norm_one_ch, (0.5, 0.5))
        
        # Save image and json file
        out = os.path.join(output, m_name, sample.PATID)
        if not os.path.isdir(out):
            os.makedirs(out)

        image_name = os.path.join(out, sample.PATID + ".png")
        group = np.repeat(r_image[:, :, np.newaxis], 3, axis=2)
        imageio.imwrite(image_name, (group * 255).astype(np.uint8))

        filename = os.path.join(out, sample.PATID + ".json")
        # save json file
        pat_dic = {
            "name": sample.PATID,
            "tiles": [image_name],
        }
        a_file = open(filename, "w")
        a_file = json.dump(pat_dic, a_file, cls=NpEncoder)


