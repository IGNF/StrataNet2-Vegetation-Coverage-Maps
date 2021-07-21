import os
import sys
import glob
import time
from comet_ml import Experiment, OfflineExperiment
import logging
import warnings
import pickle

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


np.random.seed(42)
torch.cuda.empty_cache()


def main():
    ##### SETUP THE EXPERIMENT

    # Create the experiment and its local folder
    create_new_experiment_folder(args)  # Define output paths
    if args.offline_experiment:
        experiment = OfflineExperiment(
            project_name="lidar_pac",
            offline_directory=os.path.join(args.path, "experiments/"),
            auto_log_co2=False,
        )
    else:
        experiment = Experiment(
            project_name="lidar_pac",
            auto_log_co2=False,
            disabled=args.disabled,
        )
    experiment.log_parameters(vars(args))
    if args.comet_name:
        experiment.add_tags([args.mode])
        experiment.set_name(args.comet_name)
    else:
        experiment.add_tag(args.mode)
    args.experiment = experiment  # be sure that this is not saved in text somewhere...

    logger = create_a_logger(args)

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

    logger.info(f"Training on N={len(p_data_all)} pseudo-labeled plots.")

    xy_centers_dict = {k: data["plot_center"] for k, data in p_data_all.items()}
    z_all = [c_data["plot_points_arr"][:, 2] for c_data in p_data_all.values()]
    z_all = np.concatenate(z_all)
    logger.info(f"Fitting Mixture KDE on N={len(z_all)} z values.")
    args.n_input_feats = len(args.input_feats)
    logger.info("args: \n" + str(args))

    # Fit a mixture of thre KDE
    kde_mixture = KdeMixture(z_all, args)

    # None lists that will stock stats for each fold, so we can compute the mean at the end
    all_folds_loss_train_dicts = []
    all_folds_loss_test_dicts = []

    # cross-validation
    start_time = time.time()
    fold_id = -1
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

    # TODO: Extract model declaration from full_train for this, to use the same model.
    # ATTENTION: epochs will not be comparable. Attention to learning rate decay...

    # TRAINING on fold
    model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
    PCC = PointCloudClassifier(args)

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

    # Print the fold results
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
