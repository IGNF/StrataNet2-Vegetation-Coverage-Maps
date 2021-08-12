import sys
from comet_ml import Experiment, OfflineExperiment
import logging
import warnings

warnings.simplefilter(action="ignore")


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import os
import torch

import torch.nn as nn

import matplotlib

np.random.seed(42)
torch.cuda.empty_cache()

# We import from other files
from config import args
from utils.utils import *
from data_loader.loader import *
from utils.load_data import load_pickled_dataset, prepare_and_save_plots_dataset
from learning.accuracy import *
from learning.kde_mixture import get_fitted_kde_mixture_from_dataset
from learning.train import train_full
from utils.load_data import load_pickled_dataset
from argparse import ArgumentParser

np.random.seed(42)
torch.cuda.empty_cache()


setup_experiment_folder(args, task="learning")
logger = create_a_logger(args)
experiment = launch_comet_experiment(args)

logger.info("args: \n" + str(args))

# try:
#     dataset = load_pickled_dataset(args)
# except FileNotFoundError:
dataset = prepare_and_save_plots_dataset(args)

logger.info(f"Dataset contains {len(dataset)} plots.")


# KDE Mixture
args.kde_mixture = get_fitted_kde_mixture_from_dataset(dataset, args)


def cross_validate():
    # cross-validation
    all_folds_loss_train_dicts = []
    all_folds_loss_test_dicts = []
    cloud_info_list_by_fold = {}
    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)
    for args.current_fold_id, (train_idx, val_idx) in enumerate(
        kf.split(dataset), start=1
    ):
        logger.info(f"Cross-validation FOLD = {args.current_fold_id}")
        experiment.log_metric("Fold_ID", args.current_fold_id)

        # CROSSVAL FOLD
        train_set, test_set = get_train_val_datasets(
            dataset, args, train_idx=train_idx, val_idx=val_idx
        )
        (
            _,
            all_epochs_train_loss_dict,
            all_epochs_test_loss_dict,
            cloud_info_list,
        ) = train_full(
            train_set,
            test_set,
            args,
        )

        # UPDATE LOGS
        log_last_stats_of_fold(
            all_epochs_train_loss_dict,
            all_epochs_test_loss_dict,
            args,
        )
        all_folds_loss_train_dicts.append(all_epochs_train_loss_dict)
        all_folds_loss_test_dicts.append(all_epochs_test_loss_dict)
        cloud_info_list_by_fold[args.current_fold_id] = cloud_info_list

        if args.mode == "DEV" and args.current_fold_id >= 1:
            break

    # UPDATE LOGS
    post_cross_validation_logging(
        all_folds_loss_train_dicts,
        all_folds_loss_test_dicts,
        cloud_info_list_by_fold,
        args,
    )


cross_validate()
