""" 
Use this script to analyse the output of the cross-validation, or any result file with target labels and predicted coverages.
This includes producing different confusion matrices and studying correlation between errors.
"""
import os, sys

repo_absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_absolute_path)
import glob
import time
import pandas as pd
from argparse import ArgumentParser
from utils.utils import (
    create_dir,
    update_namespace_with_another_namespace,
    launch_comet_experiment,
    setup_experiment_folder,
)
from utils.load_data import format_results_df
from learning.accuracy import (
    calculate_performance_indicators_V1,
    calculate_performance_indicators_V2,
    calculate_performance_indicators_V3,
)
from learning.accuracy import log_confusion_matrices, adjust_predictions_based_on_margin
from config import args
import scipy

parser = ArgumentParser(description="predictions_analysis")
parser.add_argument(
    "--results_file",
    default="",
    type=str,
    help="Path (abs or rel) to the csv file with results",
)
parser.add_argument(
    "--disabled",
    default=False,
    action="store_true",
    help="Wether we disable Comet for this run.",
)

args_local, _ = parser.parse_known_args()
args = update_namespace_with_another_namespace(args, args_local)
args.current_fold_id = -1
args.current_epoch = "last"
args.comet_name = "___".join(args.results_file.split("/")[-2:])

experiment = launch_comet_experiment(args)

setup_experiment_folder(args, task="predictions_analysis")


df_inference = pd.read_csv(args.results_file)

# Complete file is this is a folder with only ground truths + predictions.
if "acc2_veg_b" not in df_inference:
    df_inference = format_results_df(df_inference)
    try:
        df_inference = calculate_performance_indicators_V1(df_inference)
        df_inference = calculate_performance_indicators_V2(df_inference)
        df_inference = calculate_performance_indicators_V3(df_inference)
    except KeyError:
        logger.info(
            "Cannot calculate class-based performance indicators due to continuous ground truths."
        )

with args.experiment.context_manager("confusion"):
    for args.normalize_cm in ["true", "all", "pred"]:
        with args.experiment.context_manager(args.normalize_cm):
            log_confusion_matrices(args, df_inference, log=not args.disabled)


# Study a specific anticorrelation
df_inference["signed_error2_veg_b"] = (
    df_inference["error2_veg_b"]
    * 2
    * ((df_inference["pred_veg_b"] >= df_inference["vt_veg_b"]) - 0.5)
)
df_inference["signed_error2_veg_moy"] = (
    df_inference["error2_veg_moy"]
    * 2
    * ((df_inference["pred_veg_moy"] >= df_inference["vt_veg_moy"]) - 0.5)
)
spearmanr, pvalue = scipy.stats.pearsonr(
    df_inference["signed_error2_veg_b"], df_inference["signed_error2_veg_moy"]
)
print(spearmanr, pvalue)


# Adjust predictions to have acc2 = 1 if prediction is within 10% of the target class boundaries.
df_inference_with_margin = adjust_predictions_based_on_margin(df_inference)
with args.experiment.context_manager("confusion_10pp"):
    for args.normalize_cm in ["true", "all", "pred"]:
        with args.experiment.context_manager(args.normalize_cm):
            log_confusion_matrices(
                args,
                df_inference_with_margin,
                name_prefix="confusion_10pp",
                log=not args.disabled,
            )

# Focus on subset with lots of high vegetation
df_no_forest = df_inference_with_margin[df_inference_with_margin["vt_veg_h"] < 0.90]
for args.normalize_cm in ["true", "all", "pred"]:
    with args.experiment.context_manager(args.normalize_cm):
        log_confusion_matrices(
            args,
            df_no_forest,
            name_prefix="FORESTNONE_confusion_10pp",
            log=not args.disabled,
        )

# Focus on subset without lots of high vegetation
df_forest = df_inference_with_margin[df_inference_with_margin["vt_veg_h"] >= 0.90]
for args.normalize_cm in ["true", "all", "pred"]:
    with args.experiment.context_manager(args.normalize_cm):
        log_confusion_matrices(
            args,
            df_forest,
            name_prefix="FOREST_confusion_10pp",
            log=not args.disabled,
        )
