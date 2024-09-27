"""Deep Learning-based multimodal data model"""

import warnings

import torch

from sub_models import FC, WsiNet, Fusion, FC_fixed
from utils_baseline import check_parameters_between_two_models, count_parameters


class MultiModal(torch.nn.Module):
    """Deep Learning model for MiltiMmodal RA pathotype prediction."""

    def __init__(
        self,
        data_modalities,
        classes,
        dropout,
        k_init,
        fusion_method="max",
        device=None,
        freeze_up_to=None,
    ):
        super(MultiModal, self).__init__()
        self.data_modalities = data_modalities
        self.dropout = dropout
        self.k_init=k_init
        self.mfs = modality_feature_size = 512
        self.freeze_up_to = freeze_up_to
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
            mod in valid_mods for mod in data_modalities
        ), f"Accepted input data modalitites are: {valid_mods}"

        assert len(data_modalities) > 0, "At least one input must be provided."

        if fusion_method == "cat":
            self.num_features = 0
        else:
            self.num_features = self.mfs

        self.submodels = {}

        # HE patches --------------------------------------------------------#
        if "HE" in self.data_modalities:
            self.HE_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["HE"] = self.HE_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD3 patches ---------------------------------------------------------------#
        if "CD3" in self.data_modalities:
            self.CD3_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD3"] = self.CD3_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD20 patches --------------------------------------------------------------#
        if "CD20" in self.data_modalities:
            self.CD20_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD20"] = self.CD20_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD21 ---------------------------------------------------------------#
        if "CD21" in self.data_modalities:
            self.CD21_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD21"] = self.CD21_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD68 ---------------------------------------------------------------#
        if "CD68" in self.data_modalities:
            self.CD68_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD68"] = self.CD68_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD138 ---------------------------------------------------------------#
        if "CD138" in self.data_modalities:
            self.CD138_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD138"] = self.CD138_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        ########---------------------------------------------------------------#
        # CD7 ---------------------------------------------------------------#
        if "CD7" in self.data_modalities:
            self.CD7_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD7"] = self.CD7_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs
        ########---------------------------------------------------------------#
        # CD16 ---------------------------------------------------------------#
        if "CD16" in self.data_modalities:
            self.CD16_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD16"] = self.CD16_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs
        ########---------------------------------------------------------------#
        # PDGFRb ---------------------------------------------------------------#
        if "PDGFRb" in self.data_modalities:
            self.PDGFRb_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["PDGFRb"] = self.PDGFRb_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs
        ########---------------------------------------------------------------#
        # DNA1 ---------------------------------------------------------------#
        if "DNA1" in self.data_modalities:
            self.DNA1_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["DNA1"] = self.DNA1_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs
        # DNA2 ---------------------------------------------------------------#
        if "DNA2" in self.data_modalities:
            self.DNA2_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["DNA2"] = self.DNA2_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # SMA ---------------------------------------------------------------#
        if "SMA" in self.data_modalities:
            self.SMA_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["SMA"] = self.SMA_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD4 ---------------------------------------------------------------#
        if "CD4" in self.data_modalities:
            self.CD4_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD4"] = self.CD4_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CD8 ---------------------------------------------------------------#
        if "CD8" in self.data_modalities:
            self.CD8_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CD8"] = self.CD8_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # ICOS ---------------------------------------------------------------#
        if "ICOS" in self.data_modalities:
            self.ICOS_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["ICOS"] = self.ICOS_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # "Cytok" ---------------------------------------------------------------#
        if "Cytok" in self.data_modalities:
            self.Cytok_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["Cytok"] = self.Cytok_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # "CellType" ---------------------------------------------------------------#
        if "CellType" in self.data_modalities:
            self.CellType_submodel = WsiNet(
                dropout=self.dropout,
                k_init=self.k_init,
                output_vector_size=self.mfs,
                freeze_up_t=self.freeze_up_to,
            )
            self.submodels["CellType"] = self.CellType_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs
        # Instantiate multimodal aggregator ----------------------------------#
        if len(data_modalities) > 1:
            self.aggregator = Fusion(fusion_method, self.mfs, device)
        else:
            if fusion_method is not None:
                warnings.warn("Input data is unimodal: no fusion procedure.")

        # Fully-connected and final layers ------------------------------------#
        n_fc_layers = 4 
        n_neurons = 128  
        

        ############################
        """self.fc_block = FC(
            in_features=self.num_features, out_features=n_neurons, n_layers=n_fc_layers
        )"""

        self.fc_block = FC_fixed(dropout,k_init)

        if classes == 2:
            last_layer = 1
        else:
            last_layer = classes

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons, out_features=last_layer),            
        )

    def forward(self, x): 
        multimodal_features = (
            tuple()
        )  

        # Run data through modality sub-models (generate feature vectors) ----#
        for modality in x:
            multimodal_features += (self.submodels[modality](x[modality]),)

        # Feature fusion/aggregation -----------------------------------------#
        if len(multimodal_features) > 1:
            x = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {"modalities": multimodal_features, "fused": x}
        else:  # skip if running unimodal data
            x = multimodal_features[0]
            feature_repr = {"modalities": multimodal_features[0]}

        # Outputs ------------------------------------------------------------#
        x = self.fc_block(x)  # 
        feature_repr["feat_fc_block6"] = x ### return the features of the MLP layer
        final = self.final_layer(x)  

        # Return non-zero features (not missing input data)
        output_features = tuple()

        for modality in multimodal_features:
            modality_features = torch.stack(
                [
                    batch_element
                    for batch_element in modality
                    if batch_element.sum() != 0
                ]
            )
            output_features += (modality_features,)

        feature_repr["modalities"] = output_features

        return feature_repr, final
