#%%
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from utils_AE import *
from AE import ConvAE
from torchvision.utils import save_image
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import MiniBatchSparsePCA
import utils_AE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cross_decomposition import PLSRegression


utils_AE.seed_everything()  ## needed for reproducibility

task = "histology" # histology or stage

device = "cpu"
model_ = "ConvAE"
rcrop = None 
batch = 30
z_size = None  
p_size = 650 
im_size = 650 
channels = 3
n_components = 10
augmentation = "smote" 
reduction = "PLS" 
validation = None 
f = "fold0" 
features = False  
dataset = "TMA" 


if task == "stage" and dataset == "TMA":
    print("TASK = STAGE","\n")
    file = "/root/workdir/IMMUCan/Files_TMA/Files_DL/Subset_TMA_DL_Stage2classes.xlsx"
elif task == "histology" and dataset == "TMA":
    print("TASK = Histology","\n")
    file = "/root/workdir/IMMUCan/Files_TMA/Files_DL/Subset_TMA_DL_PathAdenVsSqu2.xlsx"
else:
    file = "/root/workdir/IMMUCan/IMC/IMC_Subset_For_Deep_LearningOK.xlsx"
    
Labels = pd.read_excel(file)

if dataset == "TMA" and channels == 3:
    input_path = "/root/workdir/IMMUCan/IMC_TMA_Multi-channel650x650_CD3_SMA_DNA"
if dataset == "TMA" and channels == 4:
    input_path = "/root/workdir/IMMUCan/IMC_TMA_Multi-channel650x650_CD3_SMA_DNA_CD20"

if dataset == "Immu" and channels == 3 and rcrop:
    input_path = "/root/workdir/IMMUCan/IMC_Immucan_MultiChannelROIS_ToCROP_CD3_SMA_DNA"
elif dataset == "Immu" and channels == 4 and rcrop:
    input_path = (
        "/root/workdir/IMMUCan/IMC_Immucan_MultiChannelROIS_ToCROP_CD3_SMA_DNA_CD20"
    )
elif dataset == "Immu" and channels == 3 and not rcrop:
    
    input_path = "/root/workdir/IMMUCan/IMC_Immucan_MultiChROIS_650x650_CD3_SMA_DNA"


state_dict ="/root/workdir/IMMUCan/AEdeep_Milan_3ch_CD3-DNA-SMA_z15-2000e/fold0/Best_model_VAE.pt"

out = "/root/workdir/IMMUCan/AEdeep_Milan_3ch_CD3-DNA-SMA_z15-2000e/fold0/TestBranch"


if not os.path.isdir(out):
       os.makedirs(out)


if dataset == "TMA":
    Path_images = natsorted(
        glob.glob(os.path.join(input_path, "**/*.png"), recursive=True)
    )
elif dataset == "Immu":
    Path_images = os.listdir(input_path)
    Path_images = [os.path.join(input_path,p) for p in Path_images]


if validation:
    data_dir = train_val_data(Path_images, validation=validation, file=file,data_inp=input_path,dataset=dataset)  
else:
    data_dir = train_val_data(Path_images,dataset=dataset) 


print("\n", f, "\n")
train_list = data_dir["data"][f]["train"]
val_list = data_dir["data"][f]["val"]
print("Examples for training: ", len(train_list))
print("Examples for evaluation: ", len(val_list))


###
train_dataset = ImcDataset(train_list,dataset=dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch, shuffle=True, num_workers=0
)

val_dataset = ImcDataset(val_list,dataset=dataset)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch, shuffle=False, num_workers=0
)

##
model = ConvAE(channels=channels)


model.load_state_dict(torch.load(state_dict, map_location=torch.device("cpu")))
model.to(device)

with torch.no_grad():
    train_Z = []
    train_LABELS = []
    train_NAMES = []
    train_rois = []
    train_missing = []
    train_save_coo = []
    for inddt,data in enumerate(train_loader):        
        if len(data) == 2:                          
            images, names = data["image"], data["name"]
        else:
            images, names,roi_names = data["image"], data["name"], data["roi"]

        if rcrop:        
            p_size = p_size                    
            bz,_,w,h = images.shape
            coo = [
                    [
                        random.randint(0, w - p_size),
                        random.randint(0, h - p_size),
                    ]
                    for i in range(rcrop)
                ]
            
            coo = np.array(coo)
            #print(coo)
            new_names = []
            new_batch = []      
            new_rois = []      
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
                train_save_coo.append([np.array([i[1],i[0]]) for i in coo])                
                ## replicate names
                new_names = new_names + [names[bi]] * rcrop
                if dataset == "Immu":
                    new_rois = new_rois + [roi_names[bi]] * rcrop
            new_batch = np.concatenate(new_batch, axis=0)
            
            data = torch.from_numpy(new_batch).to(device)
        else:
            data = images.to(device)

        z,decoded_train= model(data)
                         

        train_Z.append(np.array(z))        
        if rcrop:
            names=new_names            
            if dataset == "Immu":
                roi_names= new_rois
        train_NAMES.append(names)

        if dataset == "Immu":
            train_rois.append(roi_names)

                
        ## Colect labels for each of the patients
        labels = []
        for idn,name in enumerate(names):
            filter = Labels[Labels["PATID"]==name]                
            
            try: # if the patient exist for task
                lab = filter.Label.item()                
                labels.append(lab)
            except:
                train_missing.append(name)
                continue

        train_LABELS.append(labels)
        

    ### VALIDATION
    val_Z = []
    val_LABELS = []
    val_NAMES = []
    val_rois = []
    val_missing = []
    val_save_coo = []
    for indv,data in enumerate(val_loader):
        if len(data) == 2:                          
            images, namesval  = data["image"], data["name"]
        else:
            images, namesval ,roi_names = data["image"], data["name"], data["roi"]
        if rcrop:        
            p_size = p_size                    
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
            new_rois = []
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
                val_save_coo.append([np.array([i[1],i[0]]) for i in coo])
                ## replicate names (one per crop)
                new_names = new_names + [namesval[bi]] * rcrop
                if dataset == "Immu":
                    new_rois = new_rois + [roi_names[bi]] * rcrop
            new_batch = np.concatenate(new_batch, axis=0)
            
            data = torch.from_numpy(new_batch).to(device)
        else:
            data = images.to(device)

        z,decoded_val= model(data)                        

        val_Z.append(np.array(z))
        if rcrop:
            namesval = new_names
            if dataset == "Immu":
               roi_names = new_rois
        val_NAMES.append(namesval)
        if dataset == "Immu":
            val_rois.append(roi_names)

                    
        labels = []
        for idn, name in enumerate(namesval):
            filter = Labels[Labels["PATID"]==name]
                                           
            try: # if the patient exist for task
                lab = filter.Label.item()
                labels.append(lab)
            except:
                val_missing.append(name)
                continue

        val_LABELS.append(labels)                                   

#### PREDICT WITH latent space as features 

### Organize train data 
train_new_Z = np.vstack(train_Z)
if len(train_new_Z.shape) >2:
    train_new_Z = train_new_Z.reshape(train_new_Z.shape[0],-1)

train_y=[]
train_names =[]
train_coo = []
for sublista in train_LABELS:
    train_y += sublista

for sublista in train_NAMES:
    train_names += sublista

for sublista in train_save_coo:
    train_coo += sublista

if dataset=="Immu":
    train_ROI=[]
    for sublista in train_rois:
        train_ROI += sublista

## Buid dictionary with features
if features and rcrop and dataset=="TMA": # TMA
    featCrops = {"train":{}}    
    for i,(nam,ft) in enumerate(zip(train_names,train_new_Z)): 
        try:
            c_ind = len(featCrops["train"][nam].keys())            
            featCrops["train"][nam]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}
        except:
            c_ind = 0
            featCrops["train"][nam]={}
            featCrops["train"][nam]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}



if features and dataset=="Immu" and rcrop:
    featCrops = {"immucan":{}}    
    for i,(nam,ft) in enumerate(zip(train_names,train_new_Z)): 
        try:
            c_ind = len(featCrops["immucan"][nam+"_"+train_ROI[i]].keys())            
            featCrops["immucan"][nam+"_"+train_ROI[i]]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}
        except:
            c_ind = 0
            featCrops["immucan"][nam+"_"+train_ROI[i]]={}
            featCrops["immucan"][nam+"_"+train_ROI[i]]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}

## check if there are missing labels for patients in training set
indx = [ind for ind, tr in enumerate(train_names) if tr in train_missing]

if len(indx)>0:
    print(len(indx), "   training examples without label for this task: ",task)
    # exclude examples without label
    train_new_Z = np.delete(train_new_Z,indx,axis=0)    
    train_names_miss = [i for i in train_names if i not in train_missing]
else:
    train_names_miss = train_names

orig_train_new_Z=train_new_Z
orig_train_y = train_y


### Organize val data 
new_Z = np.vstack(val_Z)

if len(new_Z.shape) >2:
    new_Z = new_Z.reshape(new_Z.shape[0],-1)

y = []
names = []
val_coo = []
for sublista in val_LABELS:
    y += sublista

for sublista in val_NAMES:
    names += sublista

for sublista in val_save_coo:
    val_coo += sublista

if dataset=="Immu":
    val_ROI=[]
    for sublista in val_rois:
        val_ROI += sublista

### Build dictionary with features

if features and rcrop and dataset=="TMA": # TMA
    featCrops["val"] = {}    
    for i,(nam,ft) in enumerate(zip(names,new_Z)): 

        try:
            c_ind = len(featCrops["val"][nam].keys())            
            featCrops["val"][nam]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}
        except:
            c_ind = 0
            featCrops["val"][nam]={}
            featCrops["val"][nam]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}

    """with open('/root/workdir/IMMUCan/Files_TMA/Features_AE_raw/Feat_AE_TMA_60cropsSize200_split80_20.pickle', 'wb') as handle:
        pickle.dump(featCrops, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

if features and dataset=="Immu" and rcrop:       
    for i,(nam,ft) in enumerate(zip(names,new_Z)): 
        try:
            c_ind = len(featCrops["immucan"][nam+"_"+val_ROI[i]].keys())            
            featCrops["immucan"][nam+"_"+val_ROI[i]]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}
        except:
            c_ind = 0
            featCrops["immucan"][nam+"_"+val_ROI[i]]={}
            featCrops["immucan"][nam+"_"+val_ROI[i]]["c"+str(c_ind)] = {"features":ft,"coo":train_coo[i]}
    
    """ with open('/root/workdir/IMMUCan/IMC/ImmucanPretrainFeaturesAE/PretrainFeat_AE_Immucan_60cropsSize200_split80_20.pickle', 'wb') as handle:
        pickle.dump(featCrops, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

### discard examples without label for the task
indx = [ind for ind, val in enumerate(names) if val in val_missing]

if len(indx)>0:
    print(len(indx), "   validation examples without label for this task: ",task)
    new_Z = np.delete(new_Z,indx,axis=0)      
    val_names_miss = [i for i in names if i not in val_missing]
else:
    val_names_miss = names

orig_new_Z = new_Z


####### Data augmentation
if augmentation:
    sm = BorderlineSMOTE(sampling_strategy='all',k_neighbors=5,random_state = 42)
    train_new_Z, train_y = sm.fit_resample(train_new_Z, train_y)

###  Classification
print('classification full features')
kernel = 1.0 * RBF(1.0)
clf = make_pipeline(StandardScaler(), SVC(gamma=1e1, C=1e1, kernel=kernel))
clf.fit(train_new_Z, train_y)
print("test score = " + str(clf.score(new_Z, y)))
cm = confusion_matrix(y,clf.predict(new_Z))
print(cm)
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print("Sensitivity : ", sensitivity)

specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print("Specificity : ", specificity)

if reduction =="PCA":
    print("reduction with PCA")    
    pca = PCA(n_components=n_components)
    pca.fit(train_new_Z)
    components_train = pca.fit_transform(train_new_Z)
    components_val = pca.fit_transform(new_Z)
    plt.figure()
    plt.scatter(components_val[:, 0], components_val[:, 1],c=y)
    plt.title("PCA")
    plt.show()
elif reduction =="MiniBatchPCA":
    print("reduction with MiniBatchSparsePCA")
    mb_spca = MiniBatchSparsePCA(n_components=n_components, batch_size=10)
    mb_spca.fit(train_new_Z)
    components_train = mb_spca.transform(train_new_Z)
    components_val = mb_spca.fit_transform(new_Z)
    plt.figure()
    plt.scatter(components_val[:, 0], components_val[:, 1],c=y)
    plt.title("MiniBatchSparsePCA")
    plt.show()
elif reduction == "LDA":
    print("reduction with LDA")
    n_components = 1
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    components_train = lda.fit_transform(train_new_Z, train_y)
    components_val = lda.transform(new_Z)
elif reduction == "tsne":
    print("reduction with tsne")
    tsne = TSNE(n_components=n_components, random_state=0,perplexity=3)
    components_train = tsne.fit_transform(train_new_Z)    
    components_val =  tsne.fit_transform(new_Z)
elif reduction == "PLS":
    print("reduction with PLS")
    pls = PLSRegression(n_components=n_components)
    pls.fit(train_new_Z, train_y)    
    components_train = pls.transform(train_new_Z)
    components_val =  pls.transform(new_Z)
    

kernel = 1.0 * RBF(1.0)
clf = make_pipeline(StandardScaler(), SVC(gamma=1e1, C=1e1, kernel=kernel))
clf.fit(components_train, train_y)
print("Prediction with ",n_components," components")
print("test score = " + str(clf.score(components_val, y)))
cm = confusion_matrix(y,clf.predict(components_val))
print(cm)

sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print("Sensitivity : ", sensitivity)

specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print("Specificity : ", specificity)

f1 = f1_score(y, clf.predict(components_val))
print("f1: ",f1)

fpr, tpr, thresholds = roc_curve(y,clf.predict(components_val))
roc_auc = auc(fpr, tpr)            
print("AUC: ",roc_auc)

## save set of features in a file 
dic ={"val":{"val_names":val_names_miss,"val_y":y,"val_features":orig_new_Z},
    "train":{"train_names":train_names_miss,"train_y":orig_train_y,"train_features":orig_train_new_Z}}
print(len(orig_train_y))
print(orig_train_new_Z.shape)

"""with open('/root/workdir/IMMUCan/Files_TMA/Features_AE_raw/Milan_Hist_rawFeaturesConvAE10_10_64_split80-20.pickle', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)"""


# %%
