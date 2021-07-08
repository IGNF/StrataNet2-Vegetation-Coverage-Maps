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

# Weird behavior: loading twice in cell appears to remove an elsewise occuring error.
for i in range(2):
    try:
        matplotlib.use("TkAgg")  # rerun this cell if an error occurs.
    except:
        pass

np.random.seed(42)
torch.cuda.empty_cache()

# We import from other files
from config import args
from utils.useful_functions import *
from data_loader.loader import *
from utils.load_las_data import load_all_las_from_folder, open_metadata_dataframe
from model.loss_functions import *
from model.accuracy import *
from em_gamma.get_gamma_parameters_em import *
from model.train import train_full


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
        experiment = Experiment(project_name="lidar_pac", auto_log_co2=False)
    experiment.log_parameters(vars(args))
    if args.comet_name:
        experiment.add_tags([args.mode, args.comet_name])  # does not work...
        experiment.set_name(args.comet_name)
    else:
        experiment.add_tag(args.mode)
    args.experiment = experiment  # be sure that this is not saved in text somewhere...

    logger = create_a_logger(args)

    ##### RUN THE EXPERIMENT
    # Load Las files for placettes
    (
        all_points_nparray,
        nparray_clouds_dict,
        xy_centers_dict,
    ) = load_all_las_from_folder(args)
    logger.info("Dataset contains " + str(len(nparray_clouds_dict)) + " plots.")

    # Load ground truth csv file
    # Name, 'COUV_BASSE', 'COUV_SOL', 'COUV_INTER', 'COUV_HAUTE', 'ADM'
    df_gt, placettes_names = open_metadata_dataframe(
        args, pl_id_to_keep=nparray_clouds_dict.keys()
    )

    # Fit a mixture of 2 gamma distribution if not already done
    z_all = all_points_nparray[:, 2]
    args.z_max = np.max(
        z_all
    )  # maximum z value for data normalization, obtained from the normalized dataset analysis
    args.n_input_feats = len(args.input_feats)  # number of input features
    logger.info(str(args))  # save all the args parameters
    params = run_or_load_em_analysis(z_all, args)
    logger.info(str(params))

    # We use several folds for cross validation (set the number in args)
    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)

    # None lists that will stock stats for each fold, so we can compute the mean at the end
    all_folds_loss_train_dicts = []
    all_folds_loss_test_dicts = []

    # cross-validation
    start_time = time.time()
    fold_id = 1
    cloud_info_list_by_fold = {}
    logger.info("Starting cross-validation")
    for train_ind, test_ind in kf.split(placettes_names):
        logger.info("Cross-validation FOLD = %d" % (fold_id))
        experiment.log_metric("Fold_ID", fold_id)
        train_list = placettes_names[train_ind]
        test_list = placettes_names[test_ind]

        # generate the train and test dataset
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

        # TRAINING on fold

        (
            trained_model,
            all_epochs_train_loss_dict,
            all_epochs_test_loss_dict,
            cloud_info_list,
        ) = train_full(
            args, fold_id, train_set, test_set, test_list, xy_centers_dict, params
        )

        cloud_info_list_by_fold[fold_id] = cloud_info_list

        # save the trained model
        PATH = os.path.join(
            args.stats_path,
            "model_ss_"
            + str(args.subsample_size)
            + "_dp_"
            + str(args.diam_pix)
            + "_fold_"
            + str(fold_id)
            + ".pt",
        )
        torch.save(trained_model, PATH)

        # We compute stats per fold
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

        if args.mode == "DEV" and fold_id >= 2:
            break

    # create inference results csv
    stats_for_all_folds(all_folds_loss_train_dicts, all_folds_loss_test_dicts, args)
    cloud_info_list_all_folds = [
        dict(p, **{"fold_id": fold_id})
        for fold_id, infos in cloud_info_list_by_fold.items()
        for p in infos
    ]
    df_inference = pd.DataFrame(cloud_info_list_all_folds)
    df_inference = calculate_performance_indicators(df_inference)
    inference_path = os.path.join(args.stats_path, "PCC_inference_all_placettes.csv")
    df_inference.to_csv(inference_path, index=False)

    with experiment.context_manager("summary"):
        m = df_inference.mean().to_dict()
        experiment.log_metrics(m)
        experiment.log_table(inference_path)
    logger.info(f"Saved infered, cross-validated results to {inference_path}")

    if not args.mode == "DEV":
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
            params,
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
