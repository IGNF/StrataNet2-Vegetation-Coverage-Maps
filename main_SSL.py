import os
import sys
import glob
import time
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
from utils.load_data import load_all_las_from_folder, open_metadata_dataframe
from learning.loss_functions import *
from learning.kde_mixture import KdeMixture
from learning.accuracy import *
from learning.train import train_full
from model.point_net import PointNet
from model.point_cloud_classifier import PointCloudClassifier


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


def main():

    create_new_experiment_folder(args)
    logger = create_a_logger(args)
    experiment = launch_comet_experiment(args)

    # DATA
    args.labeled_dataset_pkl_path = (
        args.las_parcelles_folder_path[:-1] + "_pickled_labeled"
    )
    labeled = glob.glob(os.path.join(args.labeled_dataset_pkl_path, "*"))
    p_data_all = {}
    for i, p in enumerate(labeled):
        with open(p, "rb") as pfile:
            p_data = pickle.load(pfile)
        p_data_all.update(p_data)
        if args.mode == "DEV":
            if i == 1:
                break
    xy_centers_dict = {k: data["plot_center"] for k, data in p_data_all.items()}
    logger.info(f"Training on N={len(p_data_all)} pseudo-labeled plots.")
    args.n_input_feats = len(args.input_feats)
    logger.info("args: \n" + str(args))

    # KDE Mixture
    z_all = [c_data["plot_points_arr"][:, 2] for c_data in p_data_all.values()]
    np.random.shuffle(z_all)
    z_all = np.concatenate(z_all[: 5 * 10 ** 5])
    logger.info(f"Fitting Mixture KDE on N={len(z_all)} z values.")
    kde_mixture = KdeMixture(z_all, args)
    del z_all

    # Cross-validation
    all_folds_loss_train_dicts = []
    all_folds_loss_test_dicts = []
    fold_id = -2
    cloud_info_list_by_fold = {}

    p_data_all_keys = np.array(list(p_data_all.keys()))
    np.random.shuffle(p_data_all_keys)
    train_list, test_list = np.split(p_data_all_keys, [len(p_data_all_keys) - 100])

    experiment.log_metric("Fold_ID", fold_id)

    train_set = tnt.dataset.ListDataset(
        train_list,
        functools.partial(
            cloud_loader_from_pickle,
            dataset=p_data_all,
            args=args,
            return_X_y_only=True,
        ),
    )
    test_set = tnt.dataset.ListDataset(
        test_list,
        functools.partial(
            cloud_loader_from_pickle,
            dataset=p_data_all,
            args=args,
            return_X_y_only=True,
        ),
    )

    # TRAINING on fold
    model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
    PCC = PointCloudClassifier(args)

    start_time = time.time()
    (
        trained_model,
        all_epochs_train_loss_dict,
        all_epochs_test_loss_dict,
        cloud_info_list,
    ) = train_full(
        args,
        fold_id,
        train_set,
        test_set,
        test_list,
        xy_centers_dict,
        model,
        PCC,
        kde_mixture,
    )
    logger.info(
        "training time "
        + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
    )

    cloud_info_list_by_fold[fold_id] = cloud_info_list

    # Print the fold results
    log_last_stats_of_fold(
        all_epochs_train_loss_dict,
        all_epochs_test_loss_dict,
        fold_id,
        args,
    )
    all_folds_loss_train_dicts.append(all_epochs_train_loss_dict)
    all_folds_loss_test_dicts.append(all_epochs_test_loss_dict)

    # create inference results csv
    stats_for_all_folds(all_folds_loss_train_dicts, all_folds_loss_test_dicts, args)
    cloud_info_list_all_folds = [
        dict(p, **{"fold_id": fold_id})
        for fold_id, infos in cloud_info_list_by_fold.items()
        for p in infos
    ]
    df_inference = pd.DataFrame(cloud_info_list_all_folds)
    df_inference = calculate_performance_indicators_V1(df_inference)
    inference_path = os.path.join(
        args.stats_path, "PCC_inference_all_pseudoplacettes.csv"
    )
    df_inference.to_csv(inference_path, index=False)

    with experiment.context_manager("summary"):
        m = df_inference.mean().to_dict()
        experiment.log_metrics(m)
        experiment.log_table(inference_path)
    logger.info(f"Saved infered, cross-validated results to {inference_path}")

    logger.info(
        "Total run time: "
        + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
    )


if __name__ == "__main__":
    main()
