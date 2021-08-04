import os
import sys
import glob
from comet_ml import Experiment, OfflineExperiment
import logging
import warnings
import pickle
from argparse import ArgumentParser

warnings.simplefilter(action="ignore")


import functools
import numpy as np
import pandas as pd

# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
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
from utils.load_data import load_pseudo_labelled_datasets
from learning.loss_functions import *
from learning.kde_mixture import KdeMixture
from learning.accuracy import log_last_stats_of_fold, post_cross_validation_logging
from learning.train import train_full, initialize_model
from model.point_net2 import PointNet2
from learning.kde_mixture import get_fitted_kde_mixture_from_dataset


np.random.seed(42)
torch.cuda.empty_cache()
# fmt: off
parser = ArgumentParser(description="Pre-Training")
parser.add_argument("--n_epoch", default=200 if not args.mode == "DEV" else 2, type=int, help="Number of training epochs",)
parser.add_argument("--n_epoch_test", default=1 if not args.mode == "DEV" else 1, type=int, help="We evaluate every -th epoch, and every epoch after epoch_to_start_early_stop",)
parser.add_argument("--epoch_to_start_early_stop", default=1 if not args.mode == "DEV" else 1, type=int, help="Epoch from which to start early stopping process, after ups and down of training.",)
parser.add_argument("--patience_in_epochs", default=10 if not args.mode == "DEV" else 1, type=int, help="Epoch to wait for improvement of MAE_loss before early stopping. Set to np.inf to disable ES.",)
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--step_size", default=1, type=int, help="After this number of steps we decrease learning rate. (Period of learning rate decay)",)
parser.add_argument("--lr_decay", default=0.75, type=float, help="We multiply learning rate by this value after certain number of steps (see --step_size). (Multiplicative factor of learning rate decay)",)
# fmt: on

args_local, _ = parser.parse_known_args()
args = update_namespace_with_another_namespace(args, args_local)


setup_experiment_folder(args, task="pretraining")
logger = create_a_logger(args)
experiment = launch_comet_experiment(args)

logger.info("Loading pretrained data...")
dataset = load_pseudo_labelled_datasets(args)
n_plots = len(dataset)
logger.info(f"Training on N={n_plots} pseudo-labeled plots.")

args.kde_mixture = get_fitted_kde_mixture_from_dataset(dataset, args)

N_PLOTS_IN_VAL_TEST = min(int(0.2 * n_plots), 100)
train_idx, val_idx = np.split(np.arange(n_plots), [n_plots - N_PLOTS_IN_VAL_TEST])
train_set, test_set = get_train_val_datasets(
    dataset, args, train_idx=train_idx, val_idx=val_idx
)

all_folds_loss_train_dicts = []
all_folds_loss_test_dicts = []
cloud_info_list_by_fold = {}
args.current_fold_id = -1
(
    model,
    all_epochs_train_loss_dict,
    all_epochs_test_loss_dict,
    cloud_info_list,
) = train_full(
    train_set,
    test_set,
    args,
)

model.save_state(args)
cloud_info_list_by_fold[args.current_fold_id] = cloud_info_list
log_last_stats_of_fold(
    all_epochs_train_loss_dict,
    all_epochs_test_loss_dict,
    args,
)
all_folds_loss_train_dicts.append(all_epochs_train_loss_dict)
all_folds_loss_test_dicts.append(all_epochs_test_loss_dict)
post_cross_validation_logging(
    all_folds_loss_train_dicts, all_folds_loss_test_dicts, cloud_info_list_by_fold, args
)
