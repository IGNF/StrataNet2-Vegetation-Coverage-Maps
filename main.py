from learning.kde_mixture import KdeMixture
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
import time
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
from learning.train import train_full
from model.point_net import PointNet
from model.point_cloud_classifier import PointCloudClassifier
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


def main():

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
    logger.info("Dataset contains " + str(len(nparray_clouds_dict)) + " plots.")

    # KDE Mixture
    z_all = all_points_nparray[:, 2]
    args.n_input_feats = len(args.input_feats)  # number of input features
    logger.info("args: \n" + str(args))  # save all the args parameters
    kde_mixture = KdeMixture(z_all, args)

    # cross-validation
    all_folds_loss_train_dicts = []
    all_folds_loss_test_dicts = []
    cloud_info_list_by_fold = {}
    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)
    logger.info("Starting cross-validation")
    start_time = time.time()
    fold_id = 1
    for train_ind, test_ind in kf.split(placettes_names):
        logger.info("Cross-validation FOLD = %d" % (fold_id))
        experiment.log_metric("Fold_ID", fold_id)
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

        PCC = PointCloudClassifier(args)
        model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
        if args.use_PT_model:
            args.trained_model_path = get_trained_model_path_from_experiment(
                args.path, args.PT_model_id
            )
            model.load_state(args.trained_model_path)
            model.set_patience_attributes(args)
            model.eval()

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
        cloud_info_list_by_fold[fold_id] = cloud_info_list
        log_last_stats_of_fold(
            all_epochs_train_loss_dict,
            all_epochs_test_loss_dict,
            fold_id,
            args,
        )
        all_folds_loss_train_dicts.append(all_epochs_train_loss_dict)
        all_folds_loss_test_dicts.append(all_epochs_test_loss_dict)

        logger.info(
            "training time "
            + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
        )
        fold_id += 1

        if args.mode == "DEV" and fold_id >= 1:
            break

    stats_for_all_folds(all_folds_loss_train_dicts, all_folds_loss_test_dicts, args)
    cloud_info_list_all_folds = [
        dict(p, **{"fold_id": fold_id})
        for fold_id, infos in cloud_info_list_by_fold.items()
        for p in infos
    ]
    df_inference = pd.DataFrame(cloud_info_list_all_folds)
    df_inference = calculate_performance_indicators_V1(df_inference)
    df_inference = calculate_performance_indicators_V2(df_inference)
    df_inference = calculate_performance_indicators_V3(df_inference)
    inference_path = os.path.join(args.stats_path, "PCC_inference_all_placettes.csv")
    df_inference.to_csv(inference_path, index=False)

    with experiment.context_manager("summary"):
        m = df_inference.mean().to_dict()
        experiment.log_metrics(m)
        experiment.log_table(inference_path)
    logger.info(f"Saved infered, cross-validated results to {inference_path}")

    if not args.mode == "DEV" and args.full_model_training:
        # TRAIN full model
        logger.info("Training on all data.")

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

        start_time = time.time()
        (
            trained_model,
            all_epochs_train_loss_dict,
            all_epochs_test_loss_dict,
            cloud_info_list,
        ) = train_full(
            args,
            -1,
            full_train_set,
            full_test_set,
            placettes_names,
            xy_centers_dict,
            kde_mixture,
        )

        # save the trained model
        PATH = os.path.join(
            args.stats_path,
            "model_ss_"
            + str(args.subsample_size)
            + "_dp_"
            + str(args.diam_pix)
            + "_full.pt",
        )
        torch.save(trained_model, PATH)

    logger.info(
        "Total run time: "
        + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
    )


if __name__ == "__main__":
    main()
