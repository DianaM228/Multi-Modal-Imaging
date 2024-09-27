import time
import os
import torchvision
import transforms as patch_transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import sys
import dataset
import torch
from evaluation import Evaluation
import json
from PIL import Image
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


def count_parameters(model, show=False):
    "Helper function to count the number of trainable and nontrainable parameters in a model"
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    if show:
        print(f"Number of trainable parameters: {trainable_params}")
    if show:
        print(f"Number of non_trainable parameters: {non_trainable_params}")
    return trainable_params, non_trainable_params


def check_parameters_between_two_models(model1, model2):
    "Helper function to check if two models are equal including their parameters"
    parameters1 = torch.tensor([p.T.mean() for p in model1.parameters()])
    parameters2 = torch.tensor([p.T.mean() for p in model2.parameters()])
    return torch.all(parameters1 == parameters2)


def seed_worker(worker_id):
    # Set a fixed random seed for each worker. For reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def elapsed_time(start):
    """Compute time since provided start time.
    Parameters
    ----------
    start: float
        Output of time.time().
    Returns
    -------
    Elapsed hours, minutes and seconds (as tuple of int).
    """
    time_elapsed = time.time() - start
    hrs = time_elapsed // 3600
    secs = time_elapsed % 3600
    mins = secs // 60
    secs = secs % 60

    return int(hrs), int(mins), int(secs)


def train_val_test_TMA(data_file, folds=5, validation=None):

    df = pd.read_excel(data_file)
    df = df.sample(frac=1, random_state=228)
    df_unique = df.drop_duplicates(subset=["TissueID"])

    train_val_patients, test_patients = train_test_split(
        df_unique, test_size=0.1, random_state=42, stratify=df_unique["Label"]
    )

    test_set = df[df["TissueID"].isin(test_patients["TissueID"])]

    # Further split train-test set into train and validation sets
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=0.1,
        random_state=42,
        stratify=train_val_patients["Label"],
    )

    train_set = df[df["TissueID"].isin(train_patients["TissueID"])]
    val_set = df[df["TissueID"].isin(val_patients["TissueID"])]

    train_data = dict(zip(list(train_set["PATID"]), list(train_set["Label"])))
    val_data = dict(zip(list(val_set["PATID"]), list(val_set["Label"])))
    test_data = dict(zip(list(test_set["PATID"]), list(test_set["Label"])))

    ### Dictionary with dataframe per split
    data_dir = {
        "data": {"fold0": {"train": train_data, "val": val_data}},
        "test": test_data,
    }

    return data_dir


def train_val_test_data(data_file, folds=5, validation=None, oversampling=False):
    """Make dictionary of patient labels for train validation and test after applying
        stratified cross-validation
    Parameters
    data_file: file with a list of patients and the respective label for the clasifications task
    validation: validation strategy (options: "loo" and "cv")
    ----------
    Returns:
    -------
    Dframe with PATID and label
    """
    df = pd.read_excel(data_file)
    df = df.sample(frac=1, random_state=228)
    print("file name: ", data_file)

    if validation == None:
        ## split 80% for train, 10% for validation and 10% test
        train_ratio = 0.8
        validation_ratio = 0.1
        test_ratio = 0.1
        x_train, x_test, y_train, y_test = train_test_split(
            df["PATID"],
            df["Label"],
            test_size=1 - train_ratio,
            stratify=df["Label"],
            random_state=42,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_test,
            y_test,
            test_size=test_ratio / (test_ratio + validation_ratio),
            stratify=y_test,
            random_state=42,
        )

        if oversampling:
            print()
            X = np.array(list(x_train)).reshape(-1, 1)
            y1 = list(y_train)
            print("Original dataset shape %s" % Counter(y1))

            ros = RandomOverSampler(random_state=42)
            x_train, y_train = ros.fit_resample(X, y1)
            x_train = x_train.squeeze()
            print("Resampled dataset shape %s" % Counter(y_train))

        train_data = pd.DataFrame({"name": list(x_train), "label": list(y_train)})
        val_data = pd.DataFrame({"name": list(x_val), "label": list(y_val)})
        test_data = pd.DataFrame({"name": list(x_test), "label": list(y_test)})

        ### Dictionary with dataframe per split
        data_dir = {
            "data": {"fold0": {"train": train_data, "val": val_data}},
            "test": test_data,
        }

    elif validation:
        ## extract test data (10%)
        x_train, x_test, y_train, y_test = train_test_split(
            df["PATID"],
            df["Label"],
            test_size=0.1,
            stratify=df["Label"],
            random_state=42,
        )

        ## k-folds cv
        if validation == "cv":
            fold = folds
            print("cross-validation strategy")
        elif validation == "loo":
            fold = len(x_train)
            print("leave-one-patient-out validation strategy")
        else:
            sys.exit("validation strategy not accepted")

        skf = StratifiedKFold(n_splits=fold)
        skf.get_n_splits(x_train, y_train)

        data_dir = {"data": {}}
        for i, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):

            Xtrain_indx = x_train.iloc[train_index]
            Ytrain_indx = y_train.iloc[train_index]
            Xval_indx = x_train.iloc[val_index]
            Yval_indx = y_train.iloc[val_index]

            if oversampling:
                X = np.array(list(Xtrain_indx)).reshape(-1, 1)
                y1 = list(Ytrain_indx)
                print("Original dataset shape %s" % Counter(y1))

                ros = RandomOverSampler(random_state=42)
                Xtrain_indx, Ytrain_indx = ros.fit_resample(X, y1)
                Xtrain_indx = Xtrain_indx.squeeze()
                print("Resampled dataset shape %s" % Counter(Ytrain_indx))

            train_data = pd.DataFrame(
                {"name": list(Xtrain_indx), "label": list(Ytrain_indx)}
            )
            val_data = pd.DataFrame({"name": list(Xval_indx), "label": list(Yval_indx)})

            data_dir["data"]["fold" + str(i)] = {"train": train_data, "val": val_data}

        test_data = pd.DataFrame({"name": list(x_test), "label": list(y_test)})
        data_dir["test"] = test_data

    return data_dir


def train_val_test_data_balanced_Under(
    data_file, folds=5, percentage=0, validation=None
):

    df = pd.read_excel(data_file)
    df = df.sample(frac=1, random_state=228)
    print("\n", "file name: ", data_file, "\n")

    count_by_class = df["Label"].value_counts()
    clase_menor = count_by_class.idxmin()
    clase_mayor = count_by_class.idxmax()

    ## always take out 10% od the minor class for test
    num_datos_menor = int(0.9 * count_by_class[clase_menor])
    num_datos_mayor = int(num_datos_menor + (num_datos_menor * percentage))

    if num_datos_mayor > count_by_class[clase_mayor]:
        print(
            "Error: The number of data selected exceeds the number of samples available"
        )
        exit()

    lis_classes = []
    for l in pd.unique(df["Label"]):
        if l != clase_mayor:
            lis_classes.append(df[df["Label"] == l].sample(n=num_datos_menor))
        else:
            lis_classes.append(df[df["Label"] == l].sample(n=num_datos_mayor))

    sel = pd.concat(lis_classes)  ## Selected data for train val
    nosel = df[~df.isin(sel)].dropna()  # No selected (test)

    print("\n", "Selected data (train-val):  ", "\n", sel["Label"].value_counts())
    print("\n", "No selected data (test):  ", "\n", nosel["Label"].value_counts())

    if validation == None:
        ## split 90% for train, 10% for validation
        x_train, x_val, y_train, y_val = train_test_split(
            sel["PATID"],
            sel["Label"],
            test_size=0.1,
            random_state=42,
            stratify=sel["Label"],
        )

        x_test = nosel["PATID"]
        y_test = nosel["Label"]


        train_data = dict(
            zip(list(x_train.to_frame()["PATID"]), list(y_train.to_frame()["Label"]))
        )
        val_data = dict(
            zip(list(x_val.to_frame()["PATID"]), list(y_val.to_frame()["Label"]))
        )
        test_data = dict(
            zip(list(x_test.to_frame()["PATID"]), list(y_test.to_frame()["Label"]))
        )

        ### Dictionary with dataframe per split
        data_dir = {
            "data": {"fold0": {"train": train_data, "val": val_data}},
            "test": test_data,
        }

    elif validation:
        ## k-folds cv
        if validation == "cv":
            fold = folds
            print("cross-validation strategy")
        elif validation == "loo":
            fold = len(sel)
            print("leave-one-patient-out validation strategy")
        else:
            sys.exit("validation strategy not accepted")

        skf = StratifiedKFold(n_splits=fold)
        skf.get_n_splits(sel["PATID"], sel["Label"])

        x_test = nosel["PATID"]
        y_test = nosel["Label"].astype(int)
        test_data = dict(
            zip(list(x_test.to_frame()["PATID"]), list(y_test.to_frame()["Label"]))
        )

        data_dir = {"data": {}}
        for i, (train_index, val_index) in enumerate(
            skf.split(sel["PATID"], sel["Label"])
        ):
            Xtrain_indx = sel.iloc[train_index]["PATID"]
            Ytrain_indx = sel.iloc[train_index]["Label"]
            Xval_indx = sel.iloc[val_index]["PATID"]
            Yval_indx = sel.iloc[val_index]["Label"]

            train_data = dict(
                zip(
                    list(Xtrain_indx.to_frame()["PATID"]),
                    list(Ytrain_indx.to_frame()["Label"]),
                )
            )
            val_data = dict(
                zip(
                    list(Xval_indx.to_frame()["PATID"]),
                    list(Yval_indx.to_frame()["Label"]),
                )
            )

            data_dir["data"]["fold" + str(i)] = {"train": train_data, "val": val_data}
        data_dir["test"] = test_data

    return data_dir


def train_val_test_data_balanced_Over(data_file, folds=5, validation=None):

    df = pd.read_excel(data_file)
    df = df.sample(frac=1, random_state=228)
    print("\n", "file name: ", data_file, "\n")

    X = np.array(df["PATID"]).reshape(-1, 1)
    y = list(df["Label"])

    unique = np.unique(y)
    target_0 = unique[0]
    target_1 = unique[1]
    A = Counter(y)

    if len(unique) == 2:
        class_0 = A.get(target_0)
        class_1 = A.get(target_1)
        n0 = int(class_0 / 10)
        n1 = int(class_1 / 10)
        baseline_dist = max(class_0, class_1) / (
            max(class_0, class_1) + min(class_0, class_1)
        )
        ratio = max(class_0, class_1) / min(class_0, class_1)
    elif len(unique) > 2:
        ratio = 2
    print("Original dataset shape %s" % Counter(y))

    if ratio >= 2:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)
        print("Resampled dataset shape %s" % Counter(y))

    copyY = y
    yy = np.array(y)

    if validation == None:
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            yy,
            test_size=0.2,
            random_state=42,
            stratify=yy,
        )

        ## Delet from val replicated patients on train
        toBeDeleted = []
        for ind, i in enumerate(x_test):
            query = i[0]
            j = 0
            query_found = 0
            while j < x_train.shape[0] and query_found == 0:
                match = x_train[j][0]

                if query == match:
                    query_found = 1
                    toBeDeleted.append(ind)
                j = j + 1

        if x_test.shape[0] - len(toBeDeleted) >= 5:  
            x_test = np.delete(x_test, np.s_[toBeDeleted], axis=0)
            y_test = np.delete(y_test, np.s_[toBeDeleted], axis=0)

        train_data = dict(
            zip(list(x_train.to_frame()["PATID"]), list(y_train.to_frame()["Label"]))
        )
        val_data = dict(
            zip(list(x_test.to_frame()["PATID"]), list(y_test.to_frame()["Label"]))
        )
        test_data = val_data

        ### Dictionary with dataframe per split
        data_dir = {
            "data": {"fold0": {"train": train_data, "val": val_data}},
            "test": test_data,
        }

    elif validation:
        ## k-folds cv
        if validation == "cv":
            fold = folds
            print("cross-validation strategy")
        elif validation == "loo":
            fold = len(np.size(X))
            print("leave-one-patient-out validation strategy")
        else:
            sys.exit("validation strategy not accepted")

        
        kf = KFold(n_splits=fold, random_state=10, shuffle=True)
        kf.get_n_splits(X)

        data_dir = {"data": {}}
        for i, (train_index, val_index) in enumerate(kf.split(X, yy)):
            Xtrain_indx = X[train_index]
            Ytrain_indx = yy[train_index]
            Xval_indx = X[val_index]
            Yval_indx = yy[val_index]

            #### Delet from val replicated patients on train
            toBeDeleted = []
            for ind, i in enumerate(Xval_indx):
                query = i[0]
                j = 0
                query_found = 0
                while j < Xtrain_indx.shape[0] and query_found == 0:
                    match = Xtrain_indx[j][0]

                    if query == match:
                        query_found = 1
                        toBeDeleted.append(ind)
                    j = j + 1

            if Xval_indx.shape[0] - len(toBeDeleted) >= 5:  
                print("Borrando repetidos")
                x_val = np.delete(Xval_indx, np.s_[toBeDeleted], axis=0)
                y_val = np.delete(Yval_indx, np.s_[toBeDeleted], axis=0)

            ##
            train_data = dict(
                zip(
                    list(np.array([i[0] for i in Xtrain_indx])),
                    list(Ytrain_indx),
                )
            )

            val_data = dict(
                zip(
                    list(np.array([i[0] for i in x_val])),
                    list(y_val),
                )
            )
            print(val_data)

            test_data = val_data

            data_dir["data"]["fold" + str(i)] = {"train": train_data, "val": val_data}
        data_dir["test"] = test_data

    return data_dir


def train_val_data(data_file, folds=5, validation=None, oversampling=False):
    """
    When there are few examples we use train and validation only, but
    in order to not crash the remaining code a "test" set is cloned from the validation set.
    """
        
    df = pd.read_excel(data_file)
    df = df.sample(frac=1, random_state=228)
    print("file name: ", data_file)

    if validation == None:
        ## split 80% for train, 20% for validation

        x_train, x_val, y_train, y_val = train_test_split(
            df["PATID"],
            df["Label"],
            test_size=0.2,
            stratify=df["Label"],
            random_state=42,
        )

        if oversampling:
            print()
            X = np.array(list(x_train)).reshape(-1, 1)
            y1 = list(y_train)
            print("Original dataset shape %s" % Counter(y1))

            ros = RandomOverSampler(random_state=42)
            x_train, y_train = ros.fit_resample(X, y1)
            x_train = x_train.squeeze()
            print("Resampled dataset shape %s" % Counter(y_train))

        train_data = pd.DataFrame({"name": list(x_train), "label": list(y_train)})
        val_data = pd.DataFrame({"name": list(x_val), "label": list(y_val)})
        test_data = pd.DataFrame({"name": list(x_val), "label": list(y_val)})

        ### Dictionary with dataframe per split
        data_dir = {
            "data": {"fold0": {"train": train_data, "val": val_data}},
        }
        data_dir["test"] = test_data

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

            if oversampling:
                X = np.array(list(Xtrain_indx)).reshape(-1, 1)
                y1 = list(Ytrain_indx)
                print("Original dataset shape %s" % Counter(y1))

                ros = RandomOverSampler(random_state=42)
                Xtrain_indx, Ytrain_indx = ros.fit_resample(X, y1)
                Xtrain_indx = Xtrain_indx.squeeze()
                print("Resampled dataset shape %s" % Counter(Ytrain_indx))

            train_data = pd.DataFrame(
                {"name": list(Xtrain_indx), "label": list(Ytrain_indx)}
            )
            val_data = pd.DataFrame({"name": list(Xval_indx), "label": list(Yval_indx)})

            data_dir["data"]["fold" + str(i)] = {"train": train_data, "val": val_data}
        data_dir["test"] = val_data

    return data_dir


def custom_norm(x):    

    img_norm = Image.fromarray(np.array(x) / 255)

    return img_norm


def custom_collate(batch):
    
    return torch.stack([torch.stack(images) for images in batch], dim=0)


def get_dataloaders(
    data_location,
    modalities,
    data,
    wsi_patch_size=None,
    n_wsi_patches=None,
    exclude_patients=None,
    return_patient_id=False,
    transform=False,
    batch_size=None,
    path_save=None,
    crop=None,
    tile_size=None,
    Orig_size=None,
    n_crops=None,
    database = None
):
    """Instantiate PyTorch DataLoaders.
    Parameters
        data_location: str
                  path to folder with dataset (one subfolder per modality )
        exclude_patients:  list
                  patients you want to exclude for the run
    ----------
    Returns
    -------
    Dict of Pytorch Dataloaders.
    """

    data_dirs = {
        "HE": os.path.join(data_location, "HE"),
        "CD3": os.path.join(data_location, "CD3"),
        "CD4": os.path.join(data_location, "CD4"),
        "CD7": os.path.join(data_location, "CD7"),
        "CD8": os.path.join(data_location, "CD8"),
        "CD16": os.path.join(data_location, "CD16"),
        "CD20": os.path.join(data_location, "CD20"),
        "CD21": os.path.join(data_location, "CD21"),
        "CD68": os.path.join(data_location, "CD68"),
        "CD138": os.path.join(data_location, "CD138"),
        "SMA": os.path.join(data_location, "SMA"),
        "DNA1": os.path.join(data_location, "DNA1"),
        "DNA2": os.path.join(data_location, "DNA2"),
        "PDGFRb": os.path.join(data_location, "PDGFRb"),
        "ICOS": os.path.join(data_location, "ICOS"),
        "Cytok": os.path.join(data_location, "Cytok"),
        "CellType": os.path.join(data_location, "CellType"),
    }

    data_dirs = {mod: data_dirs[mod] for mod in modalities}
    if batch_size is None:
        if n_wsi_patches > 1:
            batch_size = 2**5
        else:
            batch_size = 2**7

    
    patient_labels = {
        "train": data["train"],
        "val": data["val"],
        "test": data["test"],
    }

    
    if transform == "DA":
        transforms = {
            "train": torchvision.transforms.Compose(
                [
                    patch_transforms.ToPIL(),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ColorJitter(
                        contrast=0.5, brightness=0.5, saturation=0.5, hue=0.5
                    ),
                    torchvision.transforms.RandomApply(
                        [torchvision.transforms.GaussianBlur(kernel_size=3)], p=0.5
                    ),
                    patch_transforms.ToNumpy(),
                    patch_transforms.ToTensor(),
                ]
            ),
            # No data augmentation for validation
            "val": torchvision.transforms.Compose(
                [
                    patch_transforms.ToPIL(),
                    patch_transforms.ToNumpy(),
                    patch_transforms.ToTensor(),
                ]
            ),
            "test": torchvision.transforms.Compose(
                [
                    patch_transforms.ToPIL(),
                    patch_transforms.ToNumpy(),
                    patch_transforms.ToTensor(),
                ]
            ),
        }
    elif transform == "otsu":
        print("Otsu")
        transforms = {
            "train": torchvision.transforms.Compose(
                [
                    patch_transforms.ToPIL(),
                    patch_transforms.ToNumpy(),
                    patch_transforms.Otsu(),
                    patch_transforms.ToTensor(),
                ]
            ),
            # No data augmentation for validation
            "val": torchvision.transforms.Compose(
                [
                    patch_transforms.ToPIL(),
                    patch_transforms.ToNumpy(),
                    patch_transforms.Otsu(),
                    patch_transforms.ToTensor(),
                ]
            ),
            "test": torchvision.transforms.Compose(
                [
                    patch_transforms.ToPIL(),
                    patch_transforms.ToNumpy(),
                    patch_transforms.Otsu(),
                    patch_transforms.ToTensor(),
                ]
            ),
        }

    else:
        transforms = {"train": None, "val": None, "test": None}

    datasets = {
        x: dataset.MultimodalDataset(
            label_map=patient_labels[
                x
            ],  # dataframe with PATID and labels for all patients in train, val or test
            data_dirs=data_dirs,
            n_patches=n_wsi_patches,
            patch_size=wsi_patch_size,
            transform=transforms[x],
            exclude_patients=exclude_patients,
            return_patient_id=return_patient_id,
            crop=crop,
            tile_size=tile_size,
            Orig_size=Orig_size,
            n_crops=n_crops,
            database = database
        )
        for x in ["train", "val", "test"]
    }

    if path_save:
        if not os.path.isdir(path_save):
            os.makedirs(path_save)
        torch.save(
            {"datasets": datasets, "batch_size": batch_size},
            os.path.join(path_save, "datasets.pt"),
        )

    print("Data modalities:")
    for mod in modalities:
        print("  ", mod)
    print()
    print("Dataset sizes (# patients):")
    for x in datasets.keys():
        print(f"   {x}: {len(datasets[x])}")
    print()
    print("Batch size:", batch_size)

   
    g1 = torch.Generator()
    g1.manual_seed(0)
    g2 = torch.Generator()
    g2.manual_seed(0)
    g3 = torch.Generator()
    g3.manual_seed(0)

    
    collate_fn = None

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g1,
            collate_fn=collate_fn,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g2,
            collate_fn=collate_fn,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g3,
            collate_fn=collate_fn,
        ),
    }

    return dataloaders


def compose_run_tag(model, lr, dataloaders, log_dir, suffix="", fold=None):

    """Compose run tag to use as file name prefix.
    Used for Tensorboard log file and model weights.
    Parameters
    ----------
    Returns
    -------
    Run tag string.
    """

    def add_string(string, addition, sep="_"):
        if not string:
            return addition
        else:
            return string + sep + addition

    data = None
    for modality in model.data_modalities:
        data = add_string(data, modality)
        if modality == "HE":
            n = dataloaders["train"].dataset.np
            size = dataloaders["train"].dataset.psize[0]
            string = f"1patch{size}px" if n == 1 else f"{n}patches{size}px"
            data = add_string(data, string, sep="")

    run_tag = f"{data}_init_lr{lr}"

    if model.fusion_method:
        if model.fusion_method != "max" and len(model.data_modalities) == 1:
            run_tag += f"_{model.fusion_method}Aggr"

    run_tag += suffix
    print(f'Run tag: "{run_tag}"')

    # Stop if TensorBoard log directory already exists
    if fold:
        tb_log_dir = os.path.join(log_dir, run_tag, "_" + fold)
    else:
        tb_log_dir = os.path.join(log_dir, run_tag)
    """assert not os.path.isdir(tb_log_dir), (
        "Tensorboard log directory " + f'already exists:\n"{tb_log_dir}"'
    )"""
    return run_tag


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
