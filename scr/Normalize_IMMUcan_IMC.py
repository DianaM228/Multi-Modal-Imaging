#%% 
import os
import glob
import pandas as pd
import numpy as np

path = "/root/workdir/IMMUCan/IMC_MeanIntensity"

csv_paths = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

df_all_imgs = pd.DataFrame()

## Information from subset of files to compute quantiles
for i in csv_paths:    
    df = pd.read_csv(i, sep=";")
    if df.shape[1]==1:
        df = pd.read_csv(i)

    df_all_imgs = pd.concat([df_all_imgs, df])

del df_all_imgs['Unnamed: 0'] 
quantiles_2_5 = df_all_imgs.quantile(q=0.025).to_frame()
quantiles_97_5 = df_all_imgs.quantile(q=0.975).to_frame()

## Generate dictionary with quantiles for each marker
quantiles_means_CSI = {}
c = 0
for ind,row in quantiles_2_5.iterrows():
    quantiles_means_CSI[row.name] = {
        "q1":round(row.values[0],2),
        "q2":round(quantiles_97_5.iloc[c].values[0],2)}
    c+=1

## Result for some channels 
"""quantiles_means = {'MPO': {'q1': 0.02, 'q2': 1.9},
                    'HistoneH3': {'q1': 0.76, 'q2': 26.56},
                    'SMA': {'q1': 0.0, 'q2': 3.44},
                    'CD16': {'q1': 0.07, 'q2': 5.12},
                    'CD38': {'q1': 0.03, 'q2': 3.22},
                    'HLADR': {'q1': 0.19, 'q2': 43.8},
                    'CD27': {'q1': 0.12, 'q2': 4.07},
                    'CD15': {'q1': 0.01, 'q2': 49.43},
                    'CD45RA': {'q1': 0.1, 'q2': 12.08},
                    'CD163': {'q1': 0.04, 'q2': 8.18},
                    'B2M': {'q1': 0.53, 'q2': 11.46},
                    'CD20': {'q1': 0.14, 'q2': 30.55},
                    'CD68': {'q1': 0.41, 'q2': 28.33},
                    'Ido1': {'q1': 0.31, 'q2': 5.78},
                    'CD3': {'q1': 0.2, 'q2': 9.04},
                    'LAG3': {'q1': 0.04, 'q2': 0.69},
                    'CD11c': {'q1': 0.17, 'q2': 13.7},
                    'PD1': {'q1': 0.1, 'q2': 1.58},
                    'PDGFRb': {'q1': 0.22, 'q2': 5.46},
                    'CD7': {'q1': 0.16, 'q2': 12.05},
                    'GrzB': {'q1': 0.41, 'q2': 6.23},
                    'PDL1': {'q1': 0.14, 'q2': 4.31},
                    'TCF7': {'q1': 0.19, 'q2': 5.9},
                    'CD45RO': {'q1': 0.82, 'q2': 24.39},
                    'FOXP3': {'q1': 0.07, 'q2': 1.9},
                    'CD303': {'q1': 0.01, 'q2': 3.44},
                    'CD206': {'q1': 0.13, 'q2': 15.13},
                    'cleavedPARP': {'q1': 0.0, 'q2': 0.51},
                    'DNA1': {'q1': 6.41, 'q2': 94.71},
                    'DNA2': {'q1': 11.24, 'q2': 166.53}
                    }

"""

#%% INDEPENDENT QUANTILE CALCULATION

import os
import glob
import pandas as pd
import numpy as np

path = "/root/workdir/IMMUCan/IMC_MeanIntensity"

for cohort in ["Cohort1", "Cohort2"]:
    c_path = os.path.join(path,cohort)
    csv_paths = glob.glob(os.path.join(c_path, "**/*.csv"), recursive=True)

    df_all_imgs = pd.DataFrame()

    ## get a subset of files to compute quantiles
    for i in csv_paths:    
        df = pd.read_csv(i, sep=";")
        if df.shape[1]==1:
            df = pd.read_csv(i)

        df_all_imgs = pd.concat([df_all_imgs, df])

    del df_all_imgs['Unnamed: 0'] 
    quantiles_2_5 = df_all_imgs.quantile(q=0.025).to_frame()
    quantiles_97_5 = df_all_imgs.quantile(q=0.975).to_frame()

    ## Generate dictionary with quantiles for each marker
    quantiles_means_CSI = {}
    c = 0
    for ind,row in quantiles_2_5.iterrows():
        quantiles_means_CSI[row.name] = {
            "q1":round(row.values[0],2),
            "q2":round(quantiles_97_5.iloc[c].values[0],2)}
        c+=1
    
    print(cohort)
    print(quantiles_means_CSI)

"""Cohort1
{'HistoneH3': {'q1': 1.69, 'q2': 50.96}, 
'SMA': {'q1': 0.0, 'q2': 5.33}, 
'CD16': {'q1': 0.07, 'q2': 4.0}, 
'CD38': {'q1': 0.03, 'q2': 3.26}, 
'HLADR': {'q1': 0.22, 'q2': 48.11}, 
'CD27': {'q1': 0.11, 'q2': 5.05}, 
'CD15': {'q1': 0.03, 'q2': 79.87}, 
'CD45RA': {'q1': 0.1, 'q2': 11.19}, 
'CD163': {'q1': 0.04, 'q2': 8.6}, 
'B2M': {'q1': 0.4, 'q2': 10.02}, 
'CD20': {'q1': 0.11, 'q2': 26.17}, 
'CD68': {'q1': 0.37, 'q2': 22.94}, 
'Ido1': {'q1': 0.25, 'q2': 5.6}, 
'CD3': {'q1': 0.18, 'q2': 9.72}, 
'LAG3': {'q1': 0.05, 'q2': 0.79}, 
'CD11c': {'q1': 0.17, 'q2': 13.68}, 
'PD1': {'q1': 0.11, 'q2': 1.66}, 
'PDGFRb': {'q1': 0.2, 'q2': 5.97}, 
'CD7': {'q1': 0.15, 'q2': 16.11}, 
'GrzB': {'q1': 0.37, 'q2': 5.58}, 
'PDL1': {'q1': 0.16, 'q2': 4.17}, 
'TCF7': {'q1': 0.16, 'q2': 6.44}, 
'CD45RO': {'q1': 1.29, 'q2': 22.62}, 
'FOXP3': {'q1': 0.05, 'q2': 2.47}, 
'ICOS': {'q1': 0.1, 'q2': 2.56}, 
'CD8a': {'q1': 0.12, 'q2': 20.56}, 
'CarbonicAnhydrase': {'q1': 0.35, 'q2': 9.59}, 
'CD33': {'q1': 0.25, 'q2': 9.26}, 
'Ki67': {'q1': 0.01, 'q2': 35.83}, 
'VISTA': {'q1': 0.09, 'q2': 4.95}, 
'CD40': {'q1': 0.08, 'q2': 7.88}, 
'CD4': {'q1': 0.47, 'q2': 11.84}, 
'CD14': {'q1': 1.1, 'q2': 22.03}, 
'Ecad': {'q1': 0.33, 'q2': 37.47}, 
'CD303': {'q1': 0.0, 'q2': 3.85}, 
'CD206': {'q1': 0.1, 'q2': 9.4}, 
'cleavedPARP': {'q1': 0.0, 'q2': 0.36}, 
'DNA1': {'q1': 10.8, 'q2': 141.22}, 
'DNA2': {'q1': 19.0, 'q2': 246.86}, 
'MPO': {'q1': 0.08, 'q2': 3.94}}

Cohort2

{'MPO': {'q1': 0.01, 'q2': 0.48}, 
'HistoneH3': {'q1': 0.63, 'q2': 7.99}, 
'SMA': {'q1': 0.0, 'q2': 2.2}, 
'CD16': {'q1': 0.07, 'q2': 5.79}, 
'CD38': {'q1': 0.03, 'q2': 3.19}, 
'HLADR': {'q1': 0.18, 'q2': 40.12}, 
'CD27': {'q1': 0.14, 'q2': 3.31}, 
'CD15': {'q1': 0.01, 'q2': 26.63}, 
'CD45RA': {'q1': 0.1, 'q2': 12.73}, 
'CD163': {'q1': 0.03, 'q2': 7.92}, 
'B2M': {'q1': 0.75, 'q2': 12.15}, 
'CD20': {'q1': 0.17, 'q2': 33.83}, 
'CD68': {'q1': 0.43, 'q2': 32.83}, 
'Ido1': {'q1': 0.39, 'q2': 5.99}, 
'CD3': {'q1': 0.21, 'q2': 8.44}, 
'LAG3': {'q1': 0.04, 'q2': 0.55}, 
'CD11c': {'q1': 0.18, 'q2': 13.71}, 
'PD1': {'q1': 0.1, 'q2': 1.55}, 
'PDGFRb': {'q1': 0.22, 'q2': 5.16}, 
'CD7': {'q1': 0.17, 'q2': 8.76}, 
'GrzB': {'q1': 0.46, 'q2': 6.73}, 
'PDL1': {'q1': 0.13, 'q2': 4.38}, 
'TCF7': {'q1': 0.2, 'q2': 5.42}, 
'CD45RO': {'q1': 0.72, 'q2': 25.48}, 
'FOXP3': {'q1': 0.1, 'q2': 1.59}, 
'ICOS': {'q1': 0.16, 'q2': 2.79}, 
'CD8a': {'q1': 0.14, 'q2': 19.28}, 
'CarbonicAnhydrase': {'q1': 0.26, 'q2': 10.46}, 
'CD33': {'q1': 0.63, 'q2': 13.44}, 
'Ki67': {'q1': 0.0, 'q2': 16.34}, 
'VISTA': {'q1': 0.14, 'q2': 4.66}, 
'CD40': {'q1': 0.16, 'q2': 7.38}, 
'CD4': {'q1': 0.62, 'q2': 9.72}, 
'CD14': {'q1': 1.51, 'q2': 22.02}, 
'Ecad': {'q1': 0.47, 'q2': 39.03}, 
'CD303': {'q1': 0.31, 'q2': 3.18}, 
'CD206': {'q1': 0.18, 'q2': 20.09}, 
'cleavedPARP': {'q1': 0.0, 'q2': 0.61}, 
'DNA1': {'q1': 4.7, 'q2': 72.92}, 
'DNA2': {'q1': 8.25, 'q2': 128.53}}

"""
