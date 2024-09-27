import os
import random
import csv
import warnings
import json
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import imageio
import collections
from transforms import ToTensor
import transforms as patch_transforms
import torchvision
import glob

class _BaseDataset(Dataset):
    def __init__(
        self,
        label_map,
        data_dirs={
            "HE": None,
            "CD3": None,
            "CD4": None,
            "CD7": None,
            "CD8": None,
            "CD16": None,
            "CD20": None,
            "CD21": None,
            "CD68": None,
            "CD138": None,
            "SMA": None,
            "DNA1": None,
            "DNA2": None,
            "PDGFRb": None,
            "ICOS": None,
            "Cytok": None,
            "CellType": None,
        },
        exclude_patients=None,
        database=None
    ):
        self.label_map = label_map
        self.data_dirs = data_dirs
        self.database = database        
        self.patient_ids = list(label_map["name"])

        valid_mods = [
            "HE",
            "CD3",
            "CD4",
            "CD8",
            "CD20",
            "CD21",
            "CD68",
            "CD138",
            "CD7",
            "CD16",
            "PDGFRb",
            "DNA1",
            "DNA2",
            "SMA",
            "ICOS",
            "Cytok",
            "CellType",
        ]
        assert all(
            k in valid_mods for k in self.data_dirs.keys()
        ), f'Accepted data modalitites (in "data_dirs") are: {valid_mods}'

        assert not all(
            v is None for v in self.data_dirs.values()
        ), f"At least one input data modality is required: {valid_mods}"

        # Check missing data: drop patients missing all data
        patients_missing_all_data = self._patients_missing_all_data()

        if patients_missing_all_data:
            print(
                f"Excluding {len(patients_missing_all_data)} patient(s)"
                + " missing all data."
            )
            self.patient_ids = [
                pid for pid in self.patient_ids if pid not in patients_missing_all_data
            ]

        if exclude_patients is not None:
            self.patient_ids = [
                pid for pid in self.patient_ids if pid not in exclude_patients
            ]
            kept = len(self.patient_ids)
            print(f"Keeping {kept} patient(s) not in exclude list.")

    def __len__(self):
        return len(self.patient_ids)

    def _get_patient_ids(self, path_to_data):       
        
        files = os.listdir(path_to_data)
        # None if empty directories
        pids = set(
            [i for i in files if (any(os.scandir(os.path.join(path_to_data, i))))]
        )  # non empty folders = folders with data

        return pids

    def _patients_missing_all_data(self):
        missing_all_data = []

        for (
            data_dir
        ) in (
            self.data_dirs.values()
        ):  
            if data_dir is not None:
                pids_in_data = self._get_patient_ids(data_dir)
                missing_data = []
                if not self.database:
                    missing_data = [
                        pid for pid in self.patient_ids if pid not in pids_in_data
                    ]

                # Break if a data modality has all data
                if not missing_data:
                    break
                elif not missing_all_data:
                    missing_all_data = missing_data
                else:  # Keep patients missing all checked data so far
                    missing_all_data = [
                        pid for pid in missing_all_data if pid in missing_data
                    ]

        return missing_all_data


class MultimodalDataset(_BaseDataset):
    """Dataset iterating over patient IDs.
    Note: Data is filled in with all zeros for every patient before checking
    availability, as a mechanism to allow data dropout. Because of this, any
    patient originally missing all input will be run with all-zero data. To
    exclude such examples, missing data is checked at instantiation and any
    patients missing all data are dropped.

    Parameters
    ----------
    label_map: DataFrame
        Patient IDs and labels as DataFrame.
    data_dirs: dict
        Data directories in the format {'HE': 'path/to/dir',
                                        'CD3': 'path/to/dir',
                                        'CD20': 'path/to/dir',
                                        'CD21': 'path/to/dir',
                                        'CD68': 'path/to/dir'}
                                        'CD138': 'path/to/dir'}
    n_patches: int
        Number of WSI patches to load per label (i.e. patient). Required
    patch_size: int
        Number of pixels defining the side length of the square patches. Required
    transform: callable
        Optional transform to apply to WSI patches.
    dropout: float [0, 1]
        Probability of dropping one data modality (applied if at least two are
        available).
        Set dropout =  0 if unimodal data
    exclude_patients: list of str
        Optional list of patient ids to exclude.
    return_patient_id: bool
        Whether to add patient id to output.
    """

    def __init__(
        self,
        label_map,
        data_dirs={"CD21": None, "CD68": None, "CD138": None},
        n_patches=None,
        patch_size=None,
        transform=None,
        dropout=0,
        exclude_patients=None,
        return_patient_id=False,
        crop=None,
        tile_size=None,
        Orig_size=None,
        n_crops=None,
        database=None
    ):

        super().__init__(label_map, data_dirs, exclude_patients,database)
        self.modality_loaders = {
            "HE": self._get_patches,
            "CD3": self._get_patches,
            "CD4": self._get_patches,
            "CD7": self._get_patches,
            "CD8": self._get_patches,
            "CD16": self._get_patches,
            "CD20": self._get_patches,
            "CD21": self._get_patches,
            "CD68": self._get_patches,
            "CD138": self._get_patches,
            "SMA": self._get_patches,
            "DNA1": self._get_patches,
            "DNA2": self._get_patches,
            "PDGFRb": self._get_patches,
            "ICOS": self._get_patches,
            "Cytok": self._get_patches,
            "CellType": self._get_patches,
        }
        assert 0 <= dropout <= 1, '"dropout" must be in [0, 1].'
        self.dropout = dropout
        if sum([v is not None for v in self.data_dirs.values()]) == 1:
            if self.dropout > 0:
                warnings.warn('Input data is unimodal: "dropout" set to 0.')
                self.dropout = 0

        ###
        self.data_dirs = {
            mod: (data_dirs[mod] if mod in data_dirs.keys() else None)
            for mod in self.modality_loaders
        }

        assert (
            n_patches is not None and n_patches > 0
        ), '"n_patches" must be greater than 0 .'

        # if transform:
        self.np = n_patches
        self.psize = patch_size, patch_size
        self.transform = transform
        self.crop = crop
        self.tile_size = tile_size
        self.database = database
        if Orig_size:
            self.Orig_size = (int(Orig_size[0]), int(Orig_size[1]))        
        self.n_crops = n_crops
        # else:
        #    self.np = self.psize = self.transform = None

        self.return_patient_id = return_patient_id

    ###
    def _read_patient_file(self, path):

        with open(path) as json_file:
            data = json.load(json_file)
        values = data["tiles"]
        
        return values

    def _get_patches(self, data_dir, patient_id, coo, p_size):
        """Read WSI patches for selected patient.
        Patient files list absolute paths to available WSI patches for the
        repective patient.
        """
        ## 
        if self.database == 'Immucan':
            patient_file = os.path.join(data_dir, patient_id.split("_")[0], patient_id + ".json")
        else:
            patient_file = os.path.join(data_dir, patient_id, patient_id + ".json")

        
        try:
            patch_files = self._read_patient_file(patient_file)
        except:  # If data is missing create all-zero tensor
            return torch.zeros([self.np, 3, self.psize[0], self.psize[0]])

        patch_files = random.sample(patch_files, self.np)
        patches = [io.imread(p) for p in patch_files]

        if self.transform is not None:
            patches = torch.stack(
                [self.transform(patch, coo, p_size) for patch in patches]
            )  ## i.e torch.Size([5, 3, 256, 256])
        elif self.crop:
            # print("Crop")
            transform_crop = {
                "train": torchvision.transforms.Compose(
                    [
                        patch_transforms.ToPIL(),
                        patch_transforms.ToNumpy(),
                        patch_transforms.random_crop(coo, p_size),
                        patch_transforms.CropsToTensor(),
                    ]
                ),
                # No data augmentation for validation
                "val": torchvision.transforms.Compose(
                    [
                        patch_transforms.ToPIL(),
                        patch_transforms.ToNumpy(),
                        patch_transforms.random_crop(coo, p_size),
                        patch_transforms.CropsToTensor(),
                    ]
                ),
                "test": torchvision.transforms.Compose(
                    [
                        patch_transforms.ToPIL(),
                        patch_transforms.ToNumpy(),
                        patch_transforms.random_crop(coo, p_size),
                        patch_transforms.CropsToTensor(),
                    ]
                ),
            }

            """patches = torch.stack(
                [transform_crop[self.phase](patch) for patch in patches]
            )"""
            patches = transform_crop[self.phase](patches[0])  

        else:            
            ten_norm = ToTensor()
            tensor_list = [ten_norm(arr) for arr in patches]            
            patches = torch.stack(tensor_list, dim=0)

        return patches

    def _drop_data(self, data):
        available_modalities = []

        # Check available modalities in current mini-batch
        for modality, values in data.items():
            if len(torch.nonzero(values)) > 0:  # Keep if data is available
                available_modalities.append(modality)

        # Drop data modality
        n_mod = len(available_modalities)

        if n_mod > 1:
            if random.random() < self.dropout:
                drop_modality = random.choice(available_modalities)
                data[drop_modality] = torch.zeros_like(data[drop_modality])

        return data

    def get_patient_data(self, patient_id):        
        try:
            label = int(self.label_map[self.label_map["name"] == patient_id]["label"])
        except:  
            filt = self.label_map[self.label_map["name"] == patient_id][
                "label"
            ].reset_index()
            label = filt.iloc[0]["label"]

        data = {}

        if self.crop:
            h, w = int(self.Orig_size[0]), int(self.Orig_size[1])
            random.seed(self.epoch)
            
            coo = [
                [
                    random.randint(0, w - self.tile_size),
                    random.randint(0, h - self.tile_size),
                ]
                for i in range(self.n_crops)
            ]
            coo = np.array(coo)
        else:
            coo = None
            p_size = None

        # Load selected patient's data and build a dictionary with tensors per modality
        for modality in self.data_dirs:
            data_source = self.data_dirs[modality]
            if data_source is not None:
                data[modality] = self.modality_loaders[modality](
                    data_source,
                    patient_id,
                    coo,
                    self.tile_size,  
                )

                data[modality] = data[modality].float()

        # Data dropout
        if self.dropout > 0:
            n_modalities = len([k for k in data])
            if n_modalities > 1:
                data = self._drop_data(data)

        return data, label

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        # print(patient_id)
        data, label = self.get_patient_data(patient_id)

        if self.return_patient_id:
            return data, label, patient_id

        return data, label
