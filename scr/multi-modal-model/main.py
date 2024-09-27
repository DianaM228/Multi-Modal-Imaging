# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import torch
import utils_baseline
from model import Model
from predictor import Predictor
import argparse
import shutil as sh
import wandb


def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal test based on the paper Long‑term cancer survival prediction using multimodal deep learning"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true",
        default=False,
    )
    required_named_args = parser.add_argument_group("required named arguments")
    required_named_args.add_argument(
        "-i",
        "--input",
        help="Path to folder containing subfolders per sequence with patients tiles",
        required=True,
    )
    required_named_args.add_argument(
        "-o",
        "--output",
        help="Path to folder to save the trained models",
        required=True,
    )
    required_named_args.add_argument(
        "-f",
        "--file",
        help="Path+name of the file containing patients IDs and labels (Patients_Amount_Per_Groups_Subset1_1.xlsx) ",
        required=True,
    )
    optional_named_args = parser.add_argument_group("optional named arguments")
    optional_named_args.add_argument(
        "-m",
        "--markers",
        help="Sequences to be included in the test. Enter the name separated by a comma",
        default="HE,CD3,CD20",
        required=False,
    )
    optional_named_args.add_argument(
        "-ex",
        "--exclude",
        help="Path+name of file with patients to exclude",
        default=None,
        required=False,
    )
    optional_named_args.add_argument(
        "-b",
        "--batch",
        help="Define batch size",
        default=4,
        required=False,
        type=int,
    )
    optional_named_args.add_argument(
        "-e",
        "--epochs",
        help="Set the number of epochs to train the model",
        default=50,
        required=False,
        type=int,
    )
    optional_named_args.add_argument(
        "-fusion",
        "--fusion_method",
        help="Define the fusion method",
        default="max",
        choices=["max", "sum", "prod", "embrace", "attention"],
        required=False,
    )
    optional_named_args.add_argument(
        "-l",
        "--initial_lr",
        help="Define the initial learning rate",
        default=0.01,
        required=False,
        type=float,
    )
    optional_named_args.add_argument(
        "-lsch",
        "--lr_scheduler",
        help="Activate or deactivate lr scheduler",
        default="0",
        required=False,
        choices=["0", "1"],
    )
    optional_named_args.add_argument(
        "-c",
        "--classes",
        help="classes for the classification task",
        default=2,
        required=False,
        type=int,
    )
    optional_named_args.add_argument(
        "-s",
        "--save_all",
        help="save all the embeddings",
        default=True,
        required=False,
    )
    optional_named_args.add_argument(
        "-t",
        "--tile_size",
        help="to configure crop transformation. Set size of the image to crop",
        default=None,
        type=int,
        required=False,
    )
    optional_named_args.add_argument(
        "-nc",
        "--number_crops",
        help="to define how many crops to extract from image",
        default=1,
        type=int,
        required=False,
    )
    optional_named_args.add_argument(
        "-oz",
        "--original_size",
        help="to specify the size of the original image.Enter height and width separated by a comma",
        default=None,
        required=False,
    )
    optional_named_args.add_argument(
        "-fl",
        "--freeze",
        help="To define until which of the layers to freeze the model",
        default=None,  ## The model is completely freeze by default
        choices=["0", "1"],  # 1 = Freezed till layer4 - ["layer1", "layer2", "layer3", "layer4"],
        required=False,
    )
    optional_named_args.add_argument(
        "-dh",
        "--debug_hardcore",
        help="Debug hardcore style",
        default=False,
        choices=["0", "1"],
        required=False,
    )
    optional_named_args.add_argument(
        "-uw",
        "--use_wandb",
        help="Use weights and biases",
        default=False,
        choices=["0", "1"],
        required=False,
    )
    optional_named_args.add_argument(
        "-val",
        "--validation",
        help="select validation strategy. By default is one split valdation",
        default=None,
        choices=["cv", "loo"],
        required=False,
    )
    optional_named_args.add_argument(
        "-split",
        "--data_split",
        help="split the data with or without test (train-val ot train-val-test).Doing oversampling or undersampling",
        default="tv",
        choices=["tv", "tvo", "tvt", "custom", "tvtu", "tvto"],
        required=False,
    )
    optional_named_args.add_argument(
        "-custom_val",
        "--custom_validation",
        help="Give a pickle file with the custom split",
        default=None,
        required=False,
    )
    optional_named_args.add_argument(
        "-p",
        "--percentage",
        help="extra percentage of mayor class to reduce data imbalance",
        default=0,
        type=int,
        required=False,
    )
    optional_named_args.add_argument(
        "-d",
        "--dropout",
        help="Select the dropout for the Fully-connected layer",
        default=None,
        type=float,
        required=False,
    )
    optional_named_args.add_argument(
        "-kinit",
        "--kaiming_init",
        help="activate kaiming initialization for MLP",
        default=None,
        action= "store_true",       
        required=False,
    )
    optional_named_args.add_argument(
        "-da",
        "--data_augmentation",
        help="Activate data augmentation",
        default=None,
        choices=["DA", "otsu", "crop"],
        required=False,
    )
    optional_named_args.add_argument(
        "-rc",
        "--random_crop",
        help="Activate data augmentation with random_crop (for the momment fix=10 images of size 256x256)",
        default="0",
        choices=["0", "1"],
        required=False,
    )
    optional_named_args.add_argument(
        "-wl",
        "--weighted_loss",
        help="activate weighted loss to give more weight to class 1 (minor class)",
        default="0",
        choices=["0", "1"],
        required=False,
    )

    args = parser.parse_args()
    DATA = args.input
    save_results = args.output  
    labels_file_path = args.file
    data_modalities = args.markers.split(",")
    exclude_path = args.exclude
    batch = args.batch
    epochs = args.epochs
    init_lr = args.initial_lr
    classes = args.classes
    tile_size = args.tile_size
    freeze = "layer4" if bool(int(args.freeze)) else None
    actv_lsch = bool(int(args.lr_scheduler))
    data_split = args.data_split
    custom_val = args.custom_validation
    validation = args.validation
    dropout = args.dropout
    k_init = args.kaiming_init
    transform_data = args.data_augmentation
    debug_hardcore = bool(int(args.debug_hardcore))
    rcrop = bool(int(args.random_crop))
    wloss = bool(int(args.weighted_loss))
    n_crops = args.number_crops

    if args.original_size:
        Orig_size = args.original_size.split(",")
    else:
        Orig_size = None

    if transform_data == "crop":
        crop = True
        if tile_size is None or Orig_size is None or n_crops is None:
            print(
                "tile_size, Orig_size and n_crops needed when using crop transformation"
            )
            sys.exit(1)
    else:
        crop = False

    if rcrop:
        if tile_size is None or n_crops is None:
            print("tile_size and n_crops needed when rcrop is activated")
            sys.exit(1)

    is_wandb = bool(int(args.use_wandb))
    wandb_key = "d85e5abe138f6f18a208edf1088726f5c6af31dd"
    wandb_project = "IMC_TMA_1"

    if is_wandb:  # If we are using WANDB
        wandb_mode = None
        wandb.login(key=wandb_key)
    else:
        wandb_mode = "disabled"

    wandb_config = {
        "data_modalities": args.markers,
        "batch": args.batch,
        "epochs": args.epochs,
        "classes": args.classes,
        "tile_size": args.tile_size,
        "freeze": args.freeze,
        "dropout": args.dropout,
        "Data_Aug": transform_data,
        "lr_scheduler": actv_lsch,
        "lr": args.initial_lr,
        "wloss": args.weighted_loss,
        "kaiming_init":args.kaiming_init
    }

    with wandb.init(
        mode=wandb_mode, save_code=True, project=wandb_project, config=wandb_config
    ):
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_metric", summary="max")
        wandb.define_metric("val_metric", summary="max")

        if args.verbose:
            print("DATASET: ", DATA)
            print("Freeze: ", freeze)

        utils_baseline.seed_everything()

        if torch.cuda.is_available():
            print(">>> PyTorch detected CUDA <<<")

        if os.getcwd() not in sys.path:
            sys.path.append(os.path.join(os.getcwd(), "scripts"))

        
        if custom_val:
            data_dir = pd.read_pickle(custom_val)
            print("\n", custom_val, "\n")
        if data_split:            
            if data_split == "tv":
                data_dir = utils_baseline.train_val_data(
                    labels_file_path, validation=validation
                )
            elif data_split == "tvo":
                data_dir = utils_baseline.train_val_data(
                    labels_file_path, validation=validation, oversampling=True
                )
            elif data_split == "tvt":
                data_dir = utils_baseline.train_val_test_data(
                    labels_file_path, validation=validation
                )
            elif data_split == "tvtu":
                data_dir = utils_baseline.train_val_test_data_balanced_Under(
                    labels_file_path, validation=validation
                )
            elif data_split == "tvto":
                data_dir = utils_baseline.train_val_test_data(
                    labels_file_path, validation=validation, oversampling=True
                )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if args.verbose:
            print("Device: ", device, "\n")

        if exclude_path:
            exclude_patients = pd.read_excel(exclude_path)
            list_excluded = list(exclude_patients["PATID"])
        else:
            list_excluded = None

        ### k-folds validation
        avg_train = []
        avg_val = []
        avg_test = []
        for ind, f in enumerate(data_dir["data"]):
            print("\n", f, "\n")
            data_tvt = {
                "train": data_dir["data"][f]["train"],
                "val": data_dir["data"][f]["val"],
                "test": data_dir["test"],
            }

            dataloaders = utils_baseline.get_dataloaders(
                data_location=DATA,
                modalities=data_modalities,
                data=data_tvt,
                wsi_patch_size=tile_size,  
                n_wsi_patches=1,
                transform=transform_data,
                exclude_patients=list_excluded,
                batch_size=batch,
                return_patient_id=True,
                path_save=save_results,
                crop=crop,
                tile_size=tile_size,
                Orig_size=Orig_size,
                n_crops=n_crops,
            )
            
            multimodal = Model(
                dataloaders=dataloaders,
                classes=classes,
                wloss=wloss,
                dropout=dropout,
                k_init=k_init,
                fusion_method=args.fusion_method,
                device=device,
                freeze_up_to=freeze,
                
            )

            multimodal.model_blocks

            multimodal.model

            run_tag = utils_baseline.compose_run_tag(
                model=multimodal,
                lr=init_lr,
                dataloaders=dataloaders,
                log_dir=save_results,
                suffix="",
            )
            save_fold = os.path.join(save_results, run_tag, f)
            if not os.path.isdir(save_fold):
                os.makedirs(save_fold)

            try:
                sh.move(
                    os.path.join(save_results, "datasets.pt"),
                    os.path.join(save_results, run_tag, "datasets" + str(ind) + ".pt"),
                )
            except:
                pass

            fit_args = {
                "lr": init_lr,
                "num_epochs": epochs,
                "info_freq": 5,
                "lr_factor": 0.5,
                "scheduler_patience": 32,
                "log_dir": save_fold,
                "lr_scheduler": actv_lsch,
                "rcrop": rcrop,
                "crop": crop,
                "tile_size": tile_size,
                "n_crops": n_crops,
            }

            multimodal.fit(**fit_args)

            print("Best 3 epochs", multimodal.best_model_weights.keys())
            print("Best 3 results", multimodal.best_metric_values)
            print("Current", multimodal.current_metric, "\n")

            
            best_epoch = max(
                multimodal.best_metric_values, key=multimodal.best_metric_values.get
            )
            avg_val.append(multimodal.best_metric_values[best_epoch].item())
            print(
                "fold ",
                ind,
                "Val acc: ",
                multimodal.best_metric_values[best_epoch].item(),
            )

            # Save best model
            multimodal.save_weights(
                saved_epoch=best_epoch, prefix=run_tag, weight_dir=save_fold
            )

            # Save last model
            multimodal.save_weights(
                saved_epoch="current", prefix=run_tag, weight_dir=save_fold
            )

            ### Predict test data and save embeddings with last weights
            #####
            multimodal_b = Model(
                dataloaders=dataloaders,
                classes=classes,
                wloss=wloss,
                dropout=dropout,
                k_init=k_init,
                fusion_method=args.fusion_method,
                device=device,
                freeze_up_to=freeze,
            )

            ### Last model weights
            multimodal_last = Model(
                dataloaders=dataloaders,
                classes=classes,
                wloss=wloss,
                dropout=dropout,
                k_init=k_init,
                fusion_method=args.fusion_method,
                device=device,
                freeze_up_to=freeze,
            )
            
            Last_ep = "epoch"+str(epochs)
            file_name_last = os.path.join(
                    save_fold,
                    f"{run_tag}_{Last_ep}_"
                    + f"metric{multimodal.current_metric[Last_ep]:.2f}.pth",
                )
            
            multimodal_last.load_weights(file_name_last)
            


            if not args.save_all:
                # last model
                print("Test Last model", "\n")
                performance = utils_baseline.Evaluation(
                    model=multimodal_last,
                    dataset=dataloaders["test"].dataset,
                    device=device,
                    rcrop=rcrop,
                    tile_size=tile_size,
                    n_crops=n_crops,
                )
                performance.compute_metrics(
                    path_save=save_fold, classes=classes, set_data="LastModel_test"
                )

                print("Test Best model", "\n")
                file_name = os.path.join(
                    save_fold,
                    f"{run_tag}_{best_epoch}_"
                    + f"metric{multimodal.best_metric_values[best_epoch]:.2f}.pth",
                )
                                
                multimodal_b.load_weights(file_name)

                performance = utils_baseline.Evaluation(
                    model=multimodal_b,
                    dataset=dataloaders["test"].dataset,
                    device=device,
                    rcrop=rcrop,
                    tile_size=tile_size,
                    n_crops=n_crops,
                )
                performance.compute_metrics(
                    path_save=save_fold,
                    classes=classes,
                    set_data="BestW_test",
                )
            else:
                # save embeddings for final model
                print("\n", "FINAL MODEL", "\n")
                print("test", "\n")
                performance = utils_baseline.Evaluation(
                    model=multimodal_last,
                    dataset=dataloaders["test"].dataset,
                    device=device,
                    rcrop=rcrop,
                    tile_size=tile_size,
                    n_crops=n_crops,
                )
                performance.compute_metrics(
                    path_save=save_fold, classes=classes, set_data="LastModel_test"
                )

                print("val", "\n")
                performance = utils_baseline.Evaluation(
                    model=multimodal_last,
                    dataset=dataloaders["val"].dataset,
                    device=device,
                    rcrop=rcrop,
                    tile_size=tile_size,
                    n_crops=n_crops,
                )
                performance.compute_metrics(
                    path_save=save_fold, classes=classes, set_data="LastModel_val"
                )

               
                #############
                #  save embeddings for best model
                print("\n", "BEST MODEL", "\n")
                file_name = os.path.join(
                    save_fold,
                    f"{run_tag}_{best_epoch}_"
                    + f"metric{multimodal.best_metric_values[best_epoch]:.2f}.pth",
                )

                multimodal_b.load_weights(file_name)
                print("Test", "\n")
                performance = utils_baseline.Evaluation(
                    model=multimodal_b,
                    dataset=dataloaders["test"].dataset,
                    device=device,
                    rcrop=rcrop,
                    tile_size=tile_size,
                    n_crops=n_crops,
                )
                acct = performance.compute_metrics(
                    path_save=save_fold,
                    classes=classes,
                    set_data="BestW_test",
                )
                avg_test.append(acct)
                print("fold ", ind, "Val acc: ", acct)

                print("Val", "\n")
                performance = utils_baseline.Evaluation(
                    model=multimodal_b,
                    dataset=dataloaders["val"].dataset,
                    device=device,
                    rcrop=rcrop,
                    tile_size=tile_size,
                    n_crops=n_crops,
                )
                performance.compute_metrics(
                    path_save=save_fold,
                    classes=classes,
                    set_data="BestW_val",
                )

                

if __name__ == "__main__":
    main()
