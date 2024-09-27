import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import glob
import sys
import torch
from skimage import io
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import random

def seed_everything(seed=228):
    """Function used to manage same seed and therefore reproducibility"""
    print(f"=>[REPLICABLE] True, with seed {seed}")
    print("    =>WARNING: SLOWER THIS WAY")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ToTensor:
    """Convert patch ndarray to Tensor."""

    def __init__(self, output_dict=True):
        """
        Parameters
        ----------
        output_dict:bool
            Whether to ouptut dict of patch and label batches. If false outputs
            tuple.
        """
        self.output_dict = output_dict

    def __call__(self, image):        
        image = image.transpose((2, 0, 1)) / 255
        image = torch.from_numpy(image).float()

        return image


def train_val_data(patients_list, folds=5, dataset="TMA", validation=None, oversampling=True,file=None,data_inp = None):

    if file:
        df = pd.read_excel(file)
        df = df.sample(frac=1, random_state=228)
        print("file name: ", file)

        if validation == None:
            print("Dividing data into 80 for training and 20 validation")
            x_train, x_val, y_train, y_val = train_test_split(
            df["PATID"],
            df["Label"],
            test_size=0.2,
            stratify=df["Label"],
            random_state=42,
        )
            print(x_val)

            if oversampling:
                print()
                X = np.array(list(x_train)).reshape(-1, 1)
                y1 = list(y_train)
                print("Original dataset shape %s" % Counter(y1))

                ros = RandomOverSampler(random_state=42)
                x_train, y_train = ros.fit_resample(X, y1)
                x_train = x_train.squeeze()
                print("Resampled dataset shape %s" % Counter(y_train))

            train_paths = [os.path.join(data_inp,i,i+".png") for i in x_train]            
            val_paths = [os.path.join(data_inp,i,i+".png") for i in x_val]

            ### Dictionary with dataframe per split
            data_dir = {
                "data": {"fold0": {"train": train_paths, "val": val_paths}},
            }
            

        else:
            #######################
            if validation == "cv":
                x = df["PATID"]
                y = df["Label"]
                fold = folds
                print("cross-validation strategy")
            elif validation == "loo":
                x = df["PATID"]
                y = df["Label"]
                fold = len(x)
                print("leave-one-patient-out validation strategy")
            else:
                sys.exit("validation strategy not accepted")

            skf = StratifiedKFold(n_splits=fold)
            skf.get_n_splits(x, y)

            data_dir = {"data": {}}
            for i, (train_index, val_index) in enumerate(skf.split(x, y)):

                Xtrain_indx = x.iloc[train_index]
                Ytrain_indx = y.iloc[train_index]
                Xval_indx = x.iloc[val_index]
                Yval_indx = y.iloc[val_index]
                print("fold"+str(i),"\n",Xval_indx)
                
                if oversampling:
                    X = np.array(list(Xtrain_indx)).reshape(-1, 1)
                    y1 = list(Ytrain_indx)
                    print("Original dataset shape %s" % Counter(y1))

                    ros = RandomOverSampler(random_state=42)
                    Xtrain_indx, Ytrain_indx = ros.fit_resample(X, y1)
                    Xtrain_indx = Xtrain_indx.squeeze()
                    print("Resampled dataset shape %s" % Counter(Ytrain_indx))

                train_paths = [os.path.join(data_inp,i,i+".png") for i in Xtrain_indx]

                ### test including patient without labesl for histology
                if file == "/root/workdir/IMMUCan/Files_TMA/Files_DL/Subset_TMA_DL_PathAdenVsSqu2.xlsx":
                    l = ["Rln120050_61", "Rln120024_53",
                        "Rln100069_51", "Rln120050_60",
                        "Rln120029_47", "Rln120029_46",
                        "Rln120110_59", "Rln100131_54", 
                        "Rln120024_52", "Rln120110_58", 
                        "Rln100131_55", "Rln100069_50"]
                    extra_im = [os.path.join(data_inp,i,i+".png") for i in l]
                    train_paths = train_paths + extra_im


                val_paths = [os.path.join(data_inp,i,i+".png") for i in Xval_indx]
                
                data_dir["data"]["fold" + str(i)] = {"train": train_paths, "val": val_paths}

    else:
        if validation == None:
            print("Dividing data into 80 for training and 20 validation")
            train_paths, test_paths = train_test_split(
                patients_list, test_size=0.2, random_state=42
            )

            if dataset=="Immu":
                all_ROIS = []
                for p in train_paths:
                    all_ROIS+=glob.glob(os.path.join(p, "**/*.png"), recursive=True)
                train_paths = all_ROIS

                all_ROIS = []
                for p in test_paths:
                    all_ROIS+=glob.glob(os.path.join(p, "**/*.png"), recursive=True)
                test_paths = all_ROIS

            print(test_paths)

            data_dir = {
                "data": {"fold0": {"train": train_paths, "val": test_paths}},
            }

            print("\n",test_paths,"\n")
        else:
            #######################
            if validation == "cv":
                fold = folds
                print("cross-validation strategy")
            elif validation == "loo":
                fold = len(patients_list)
                print("leave-one-patient-out validation strategy")
            else:
                sys.exit("validation strategy not accepted")

            kf = KFold(n_splits=fold, shuffle=True, random_state=42)
            data_dir = {"data": {}}
            for fold, (train_indices, test_indices) in enumerate(kf.split(patients_list)):
                train_paths = [patients_list[i] for i in train_indices]
                test_paths = [patients_list[i] for i in test_indices]

                if dataset=="Immu":
                    all_ROIS = []
                    for p in train_paths:
                        all_ROIS+=glob.glob(os.path.join(p, "**/*.png"), recursive=True)
                    train_paths = all_ROIS

                    all_ROIS = []
                    for p in test_paths:
                        all_ROIS+=glob.glob(os.path.join(p, "**/*.png"), recursive=True)
                    test_paths = all_ROIS

                print("fold"+str(fold),test_paths)

                data_dir["data"]["fold" + str(fold)] = {
                    "train": train_paths,
                    "val": test_paths,
                }

                print("\n",test_paths,"\n")

    return data_dir


class ImcDataset(Dataset):
    """CMRI dataset for aorta segmentation on 3D"""

    def __init__(self, images_paths, transform=None,dataset = "TMA"):

        self.images_paths = images_paths
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img_path = self.images_paths[index]
        if self.dataset == "TMA":
            pat_Name = os.path.basename(img_path).split(".")[0]
        elif self.dataset == "Immu":
            pat_Name = os.path.basename(img_path).split(".")[0]
            pat_Name,roi_name = pat_Name.split("_")[0],pat_Name.split("_")[1]
             


        image = io.imread(img_path)
        tensorIm = ToTensor()
        image = tensorIm(image)        

        if self.transform:
            image = self.transform(image)            
        
        if self.dataset == "TMA":
            sample = {"name": pat_Name, "image": image}
        else:
            sample = {"name": pat_Name, "image": image,"roi":roi_name}
        return sample


def plot_training_loss(
    minibatch_losses, num_epochs, averaging_iterations=100, custom_label="", save=None
):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        range(len(minibatch_losses)),
        (minibatch_losses),
        label=f"Minibatch Loss{custom_label}",
    )
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    try:
        ax1.set_ylim([0, np.max(minibatch_losses[num_losses:]) * 1.5])
    except:
        pass

    ax1.plot(
        np.convolve(
            minibatch_losses,
            np.ones(
                averaging_iterations,
            )
            / averaging_iterations,
            mode="valid",
        ),
        label=f"Running Average{custom_label}",
    )
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


def plot_generated_images(
    data_loader,
    model,
    device,
    unnormalizer=None,
    figsize=(20, 2.5),
    n_images=15,
    modeltype="autoencoder",
    save=None,
):

    fig, axes = plt.subplots(
        nrows=2, ncols=n_images, sharex=True, sharey=True, figsize=figsize
    )

    for data in data_loader:
        images, names = data["image"], data["name"]
        data = images.to(device)

        color_channels = data.shape[1]
        image_height = data.shape[2]
        image_width = data.shape[3]

        with torch.no_grad():
            if modeltype == "autoencoder":
                decoded_images = model(data)[:n_images]
            elif modeltype == "VAE":
                encoded, z_mean, z_log_var, decoded_images = model(data)[:n_images]
            else:
                raise ValueError("`modeltype` not supported")

        orig_images = data[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device("cpu"))
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap="binary")


def plot_latent_space_with_labels(num_classes, data_loader, encoding_fn, device):
    d = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            embedding = encoding_fn(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to("cpu").numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(d[i][:, 0], d[i][:, 1], color=colors[i][1], label=f"{i}", alpha=0.5)

    plt.legend()

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(
        self, optimizer, patience=15, min_lr=1e-6, factor=0.1):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)