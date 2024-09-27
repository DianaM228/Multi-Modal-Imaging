import sys
import os
import pandas as pd
import torch
import utils_AE
from AE import ConvAE
from AE import loss_VAE_with_schedule
import argparse
import shutil as sh
import wandb
import glob
from natsort import natsorted
from train_val_test import train_vae
import matplotlib.pyplot as plt
from torchvision import transforms as t
import albumentations as albu


def main():
    parser = argparse.ArgumentParser(description="main AE for IMC images ")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    required_named_args = parser.add_argument_group("required named arguments")
    required_named_args.add_argument(
        "-i",
        "--input_path",
        help="path to the directory where is the image database",
        required=True,
    )
    required_named_args.add_argument(
        "-b", "--batch", type=int, help="batch size", required=True
    )
    required_named_args.add_argument(
        "-e", "--epochs", type=int, help="number of epoch", required=True
    )
    required_named_args.add_argument(
        "-l", "--learning_rate", type=float, help="learning rate", required=True
    )
    required_named_args.add_argument(
        "-o", "--output_path", help="output path", required=True
    )
    required_named_args.add_argument(
        "-ch",
        "--channels",
        help="set the number of channels of the input image",
        default=None,
        required=True,
        type=int,
    )
    required_named_args.add_argument(
        "-model",
        "--model",
        help="choose a model from AE.py",
        default="ConvAE",
        choices=[
            "ConvAE",
        ],
        required=True,
    )
    optional_named_args = parser.add_argument_group("optional named arguments")
    optional_named_args.add_argument(
        "-lrS",
        "--lr_scheduler",
        dest="lr_scheduler",
        help="Train the model with Plateau lr schedule ",
        action="store_true",
        default=None,
        required=False,
    )
    optional_named_args.add_argument(
        "-early",
        "--early_stop",
        dest="early_stop",
        help="Train the model with early stop options",
        action="store_true",
        default=None,
        required=False,
    )
    optional_named_args.add_argument(
        "-val",
        "--validation",
        dest="validation",
        help="Select cross-validation or leave-one-out evaluation, by default 90/10 split",
        choices=["cv", "loo"],
        default=None,
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
        "-g_clip",
        "--clip_gradient",
        help="activate gradient clip",
        default=False,
        choices=["0", "1"],
        required=False,
    )
    optional_named_args.add_argument(
        "-b_weight",
        "--beta_weight",
        help="set beta weight for kl term in the loss",
        default=1,
        required=False,
        type=float,
    )
    optional_named_args.add_argument(
        "-loss",
        "--loss",
        help="choose loss function",
        default="MSE",
        choices=["MSE", "BCE"],
        required=False,
    )
    optional_named_args.add_argument(
        "-norm_loss",
        "--normalize_loss",
        help="activate loss normalization",
        default=False,
        choices=["0", "1"],
        required=False,
    )
    optional_named_args.add_argument(
        "-z",
        "--z_size",
        help="set the latent space size",
        default=200,
        required=False,
        type=int,
    )
    optional_named_args.add_argument(
        "-lr_schedule",
        "--lr_schedule",
        dest="lr_schedule",
        help="Train the model with Plateau lr schedule",
        action="store_true",
        default=None,
    )
    optional_named_args.add_argument(
        "-weight_l2",
        "--weight_l2",
        dest="weight_l2",
        help="Weigth for L2 regularization",
        default=None,
        type=float,
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
        "-da",
        "--data_augmentation",
        help="increase dataset using multiple data augmentation strategies",
        action="store_true",
        required=False,
        default=None,
    )
    optional_named_args.add_argument(
        "-rcrop",
        "--random_crops",
        help="Enter a number of patch to extract from the original image ",
        required=False,
        default=None,
        type=int
    )
    optional_named_args.add_argument(
        "-reduction",
        "--rec_reduction",
        help="select the reduction for the reconstruction loss ",
        required=False,
        choices=["sum","mean"],
        default="sum",        
    )
    optional_named_args.add_argument(
        "-b_scheduler",
        "--beta_scheduler",
        help="activate scheduler for beta used on kl_loss ",
        required=False,
        action="store_true",
        default=None,        
    )
    optional_named_args.add_argument(
        "-file",
        "--file_split_data",
        help="enter a file to do the data split based on the labels of the listed patients",
        required=False,
        default=None,        
    )
    optional_named_args.add_argument(
        "-dataset",
        "--dataset",
        help="enter the name of the dataset to process",
        required=False,
        choices=["TMA","Immu"],
        default="TMA",        
    )

    # read inmputs
    args = parser.parse_args()
    verbose = args.verbose
    input_path = args.input_path
    output = args.output_path
    batch = args.batch
    epochs = args.epochs
    lr = args.learning_rate
    lrS = args.lr_scheduler
    channels = args.channels
    z_size = args.z_size
    earlyS = args.early_stop
    validation = args.validation
    g_clip = bool(int(args.clip_gradient))
    beta_weight = args.beta_weight
    normalize = bool(int(args.normalize_loss))
    model_ = args.model
    loss_fn = args.loss
    lr_schedule = args.lr_schedule
    weight_l2 = args.weight_l2
    dropout = args.dropout
    Data_augm = args.data_augmentation
    rcrop = args.random_crops
    reduction = args.rec_reduction
    b_scheduler = args.beta_scheduler
    file = args.file_split_data
    dataset = args.dataset

    is_wandb = bool(int(args.use_wandb))
    wandb_key = "d85e5abe138f6f18a208edf1088726f5c6af31dd"
    wandb_project = "IMC_TMA_1"

    if is_wandb:  # If we are using WANDB
        wandb_mode = None
        wandb.login(key=wandb_key)
    else:
        wandb_mode = "disabled"

    if g_clip:
        print("g_clip activated")
    elif rcrop:
        print("rcrop activated")

    wandb_config = {
        "batch": args.batch,
        "epochs": args.epochs,
        "lr_scheduler": args.lr_scheduler,
        "lr": args.learning_rate,
    }

    Parameters = {
        "Path_dataset": args.input_path,
        "Path_results": args.output_path,
    }

    with wandb.init(
        mode=wandb_mode, save_code=True, project=wandb_project, config=wandb_config
    ):
        wandb.define_metric("train_loss_rec", summary="min")
        wandb.define_metric("train_loss_kl", summary="min")
        wandb.define_metric("train_loss_combined", summary="min")

        wandb.define_metric("val_loss_rec", summary="min")
        wandb.define_metric("val_loss_kl", summary="min")
        wandb.define_metric("val_loss_combined", summary="min")

        utils_AE.seed_everything()

        if os.getcwd() not in sys.path:
            sys.path.append(os.path.join(os.getcwd(), "scripts"))

        if dataset == "TMA":
            Path_images = natsorted(
                glob.glob(os.path.join(input_path, "**/*.png"), recursive=True)
            )
        elif dataset == "Immu":
            Path_images = os.listdir(input_path)
            Path_images = [os.path.join(input_path,p) for p in Path_images]

        ## split data for training - validation and/or test
        data_dir = utils_AE.train_val_data(Path_images, 
                                            validation=validation,
                                            file=file,
                                            data_inp=input_path,
                                            dataset=dataset)

        if Data_augm:
            print("data augmentation activated")
            data_augmentation = t.Compose(
                [
                    t.RandomHorizontalFlip(p=0.3),
                    t.RandomVerticalFlip(p=0.3),
                    t.RandomApply([t.RandomRotation(30)], p=0.3),
                    t.RandomPerspective(p=0.3),
                    t.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                    t.RandomApply(
                        [t.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3
                    ),                    
                ]
            )
        else:
            data_augmentation = None

        for ind, f in enumerate(data_dir["data"]):
            print("\n", f, "\n")

            train_list = data_dir["data"][f]["train"]
            val_list = data_dir["data"][f]["val"]
            print("Examples for training: ", len(train_list))
            print("Examples for evaluation: ", len(val_list))

            if not os.path.isdir(os.path.join(output, f)):
                os.makedirs(os.path.join(output, f))

            ##
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = -1  ## to clean each time and avoid error

            model = ConvAE(channels=channels)
            model_type = "AE"
            
            model.to(device)

            if weight_l2:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_l2
                )
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_dataset = utils_AE.ImcDataset(
                train_list, transform=data_augmentation
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch, shuffle=True, num_workers=0
            )
            val_dataset = utils_AE.ImcDataset(val_list)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch, shuffle=False, num_workers=0
            )

            log_dict = train_vae(
                num_epochs=epochs,
                model=model,
                optimizer=optimizer,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                save_model=os.path.join(output, f, "Final_AE_IMC_02.pt"),
                g_clip=g_clip,
                beta_weight=beta_weight,
                Parameters=Parameters,
                normalize=normalize,
                model_type=model_type,
                loss_fn=loss_fn,
                lr_schedule=lr_schedule,
                dropout=dropout,
                rcrop=rcrop,
                reduction=reduction,
                b_scheduler=b_scheduler
            )

            utils_AE.plot_training_loss(
                log_dict["train_reconstruction_loss_per_batch"],
                epochs,
                custom_label=" (reconstruction)",
            )
            plt.savefig(os.path.join(output, f, "train_MSE_loss_batch.png"))

            utils_AE.plot_training_loss(
                log_dict["train_kl_loss_per_batch"], epochs, custom_label=" (KL)"
            )
            plt.savefig(os.path.join(output, f, "train_kl_loss_batch.png"))

            utils_AE.plot_training_loss(
                log_dict["train_combined_loss_per_batch"],
                epochs,
                custom_label=" (combined)",
            )
            plt.savefig(os.path.join(output, f, "train_combined_loss_batch.png"))

            utils_AE.plot_training_loss(
                log_dict["val_combined_loss_per_batch"],
                epochs,
                custom_label=" (combined)",
            )
            plt.savefig(os.path.join(output, f, "val_combined_loss_batch.png"))


if __name__ == "__main__":
    main()
