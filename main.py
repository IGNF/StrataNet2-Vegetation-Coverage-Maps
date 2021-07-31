import sys
from comet_ml import Experiment, OfflineExperiment
import logging
import warnings

warnings.simplefilter(action="ignore")


import functools
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import os
import torch
import torchnet as tnt

import torch.nn as nn

import matplotlib

np.random.seed(42)
torch.cuda.empty_cache()

# We import from other files
from config import args
from utils.utils import *
from data_loader.loader import *
from utils.load_data import load_all_las_from_folder, open_metadata_dataframe
from learning.loss_functions import *
from learning.accuracy import *
from learning.kde_mixture import KdeMixture
from learning.train import train_full
from argparse import ArgumentParser

np.random.seed(42)
torch.cuda.empty_cache()
# fmt: off
parser = ArgumentParser(description="Training")
parser.add_argument('--n_epoch', default=200 if not args.mode=="DEV" else 2, type=int, help="Number of training epochs")
parser.add_argument('--n_epoch_test', default=5 if not args.mode=="DEV" else 1, type=int, help="We evaluate every -th epoch, and every epoch after epoch_to_start_early_stop")
parser.add_argument('--epoch_to_start_early_stop', default=45 if not args.mode=="DEV" else 1, type=int, help="Epoch from which to start early stopping process, after ups and down of training.")
parser.add_argument('--patience_in_epochs', default=30 if not args.mode=="DEV" else 1, type=int, help="Epoch to wait for improvement of MAE_loss before early stopping. Set to np.inf to disable ES.")
parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('--step_size', default=1, type=int,
                    help="After this number of steps we decrease learning rate. (Period of learning rate decay)")
parser.add_argument('--lr_decay', default=0.985, type=float,
                    help="We multiply learning rate by this value after certain number of steps (see --step_size). (Multiplicative factor of learning rate decay)")

# fmt: on

args_local, _ = parser.parse_known_args()
args = update_namespace_with_another_namespace(args, args_local)

create_new_experiment_folder(args)
logger = create_a_logger(args)
experiment = launch_comet_experiment(args)

# DATA
(
    all_points_nparray,
    nparray_clouds_dict,
    xy_centers_dict,
) = load_all_las_from_folder(args)
df_gt, placettes_names = open_metadata_dataframe(
    args, pl_id_to_keep=nparray_clouds_dict.keys()
)

logger.info("args: \n" + str(args))
logger.info(f"Dataset contains {len(nparray_clouds_dict)} plots.")

# KDE Mixture
z_all = all_points_nparray[:, 2]
kde_mixture = KdeMixture(z_all, args)

# cross-validation
all_folds_loss_train_dicts = []
all_folds_loss_test_dicts = []
cloud_info_list_by_fold = {}
kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)


for args.current_fold_id, (train_ind, test_ind) in enumerate(
    kf.split(placettes_names), start=1
):
    logger.info(f"Cross-validation FOLD = {args.current_fold_id}")
    experiment.log_metric("Fold_ID", args.current_fold_id)
    train_list = placettes_names[train_ind]
    test_list = placettes_names[test_ind]

    test_set = tnt.dataset.ListDataset(
        test_list,
        functools.partial(
            cloud_loader,
            dataset=nparray_clouds_dict,
            df_gt=df_gt,
            train=False,
            args=args,
        ),
    )
    train_set = tnt.dataset.ListDataset(
        train_list,
        functools.partial(
            cloud_loader,
            dataset=nparray_clouds_dict,
            df_gt=df_gt,
            train=True,
            args=args,
        ),
    )

    (
        _,
        all_epochs_train_loss_dict,
        all_epochs_test_loss_dict,
        cloud_info_list,
    ) = train_full(
        args,
        train_set,
        test_set,
        test_list,
        xy_centers_dict,
        kde_mixture,
    )
    cloud_info_list_by_fold[args.current_fold_id] = cloud_info_list
    log_last_stats_of_fold(
        all_epochs_train_loss_dict,
        all_epochs_test_loss_dict,
        args.current_fold_id,
        args,
    )
    all_folds_loss_train_dicts.append(all_epochs_train_loss_dict)
    all_folds_loss_test_dicts.append(all_epochs_test_loss_dict)

    if args.mode == "DEV" and args.current_fold_id >= 1:
        break

post_cross_validation_logging(
    all_folds_loss_train_dicts,
    all_folds_loss_test_dicts,
    cloud_info_list_by_fold,
    args,
)

if args.full_model_training:

    logger.info("Training on all data.")
    args.current_fold_id = -1

    full_train_set = tnt.dataset.ListDataset(
        placettes_names,
        functools.partial(
            cloud_loader,
            dataset=nparray_clouds_dict,
            df_gt=df_gt,
            train=True,
            args=args,
        ),
    )
    full_test_set = tnt.dataset.ListDataset(
        placettes_names,
        functools.partial(
            cloud_loader,
            dataset=nparray_clouds_dict,
            df_gt=df_gt,
            train=False,
            args=args,
        ),
    )

    model = train_full(
        args,
        full_train_set,
        full_test_set,
        placettes_names,
        xy_centers_dict,
        kde_mixture,
    )

    model.save_state(args)
