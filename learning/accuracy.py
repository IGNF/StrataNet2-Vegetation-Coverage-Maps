import numpy as np
import pandas as pd
from functools import reduce
import logging
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.utils import create_dir, format_float_as_percentage

logger = logging.getLogger(__name__)

# values should be in [0,1] since we deal with ratios of coverage
bins_centers = np.round(np.array([0.0, 0.10, 0.25, 0.33, 0.50, 0.75, 0.90, 1.00]), 3)
bins_borders = np.append((bins_centers[:-1] + bins_centers[1:]) / 2, 1.05)
# bins_borders = np.array([0.05 , 0.175, 0.29 , 0.415, 0.625, 0.825, 0.95 , 1.05 ])

# check that borders are at equal distance of centers of surrounding classes
assert all(
    x == y
    for x, y in zip(
        list(
            map(
                lambda x: np.round(abs(x[0] - x[1]), 3),
                zip(bins_borders[:-1], bins_centers[:-1]),
            )
        ),
        list(
            map(
                lambda x: np.round(abs(x[0] - x[1]), 3),
                zip(bins_borders, bins_centers[1:]),
            )
        ),
    )
)
# we round up to be coherent with current metrics
bins_borders = np.floor(bins_borders * 100 + 0.5) / 100
# add the lowest border of class 0 to borders
bb_ = [0] + bins_borders.tolist()
# map center to ther borders in a dict
center_to_border_dict = {
    center: borders for center, borders in zip(bins_centers, zip(bb_[:-1], bb_[1:]))
}


def get_closest_class_center_index(y):
    """From y in 0-1 range, get index of closest coverage class center (e.g. 4 for y=0.47)"""
    closest_center_index = np.argsort(abs(bins_centers - y))[0]
    return closest_center_index


def get_closest_class_center(y):
    """From y in 0-1 range, get closest coverage class center (e.g. 0.5 for y=0.47)"""
    closest_center_index = get_closest_class_center_index(y)
    closest_center = bins_centers[closest_center_index]
    return closest_center


def get_neighboor_external_bounds(y):
    assert 0 <= y <= 1
    y_neigh_lower, y_neigh_higher = get_neighboor_centers(y)
    lower_bound = center_to_border_dict[y_neigh_lower][0]
    higher_bound = center_to_border_dict[y_neigh_higher][1]
    return lower_bound, higher_bound


def get_neighboor_centers(y):
    """From a center y, get neighboor centers. Returns the same center if it is 0 or 100"""
    assert 0 <= y <= 1
    y_neigh_lower = bins_centers[max(0, np.argwhere(bins_centers == y) - 1)]
    y_neigh_higher = bins_centers[
        min(len(bins_centers) - 1, np.argwhere(bins_centers == y) + 1)
    ]
    return y_neigh_lower.item(), y_neigh_higher.item()


# TODO: simplify by providing a param version=1,2,3
def compute_mae(y_pred, y, center_to_border_dict=None):
    """
    Returns the absolute distance of predicted value to ground truth.
    Args center_to_border_dict is for compatibility issues in quanfification error analysis.
    """
    return abs(y_pred - y)


def compute_mae2(y_pred, y, center_to_border_dict=center_to_border_dict):
    """
    Returns the absolute distance of predicted value to groun truth class boundaries.
    """
    borders = center_to_border_dict[y]
    if borders[0] <= y_pred <= borders[1]:
        return 0.0
    else:
        return min(abs(borders[0] - y_pred), abs(borders[1] - y_pred))


def compute_mae3(y_pred, y, center_to_border_dict=center_to_border_dict):
    """
    Returns the absolute distance of predicted value to groun truth class boundaries.
    """

    lower_bound, higher_bound = get_neighboor_external_bounds(y)

    if lower_bound <= y_pred <= higher_bound:
        return 0.0
    else:
        return min(abs(lower_bound - y_pred), abs(higher_bound - y_pred))


def compute_accuracy(y_pred, y, center_to_border_dict=center_to_border_dict):
    """Get Acc2 from y_pred and y (ground truth, center of class)."""
    bounds = center_to_border_dict[y]
    if bounds[0] <= y_pred <= bounds[1]:
        return 1
    else:
        return 0


def compute_accuracy2(
    y_pred, y, margin=0.1, center_to_border_dict=center_to_border_dict
):
    """Get Acc2 from y_pred and y (ground truth, center of class)."""
    bounds = center_to_border_dict[y]
    if (bounds[0] - margin) <= y_pred <= (bounds[1] + margin):
        return 1
    else:
        return 0


def compute_accuracy3(
    y_pred, y, margin=0.1, center_to_border_dict=center_to_border_dict
):
    """Get Acc2 from y_pred and y (ground truth, center of class)."""
    lower_bound, higher_bound = get_neighboor_external_bounds(y)
    if lower_bound <= y_pred <= higher_bound:
        return 1
    else:
        return 0


def calculate_performance_indicators_V1(df):
    """Compute indicators of performances from df of predictions and GT:
    - MAE: absolute distance of predicted value to ground truth
    - Accuracy: 1 if predicted value falls within class boundaries
    Note: Predicted and ground truths coverage values are ratios between 0 and 1.
    """
    # round to 3rd to avoid artefacts like 0.8999999 for 0.9 as key of dict
    df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]] = (
        df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]].astype(np.float).round(3)
    )
    # MAE errors
    df["error_veg_b"] = (df["pred_veg_b"] - df["vt_veg_b"]).abs()
    df["error_veg_moy"] = (df["pred_veg_moy"] - df["vt_veg_moy"]).abs()
    df["error_veg_h"] = (df["pred_veg_h"] - df["vt_veg_h"]).abs()
    df["error_veg_b_and_moy"] = df[["error_veg_b", "error_veg_moy"]].mean(axis=1)
    df["error_all"] = df[["error_veg_b", "error_veg_moy", "error_veg_h"]].mean(axis=1)

    # Accuracy
    try:
        df["acc_veg_b"] = df.apply(
            lambda x: compute_accuracy(x.pred_veg_b, x.vt_veg_b), axis=1
        )
        df["acc_veg_moy"] = df.apply(
            lambda x: compute_accuracy(x.pred_veg_moy, x.vt_veg_moy), axis=1
        )
        df["acc_veg_h"] = df.apply(
            lambda x: compute_accuracy(x.pred_veg_h, x.vt_veg_h), axis=1
        )
        df["acc_veg_b_and_moy"] = df[["acc_veg_b", "acc_veg_moy"]].mean(axis=1)
        df["acc_all"] = df[["acc_veg_b", "acc_veg_moy"]].mean(axis=1)
    except KeyError:
        logger.info(
            "Cannot calculate class-based performance indicators due to continuous ground truths."
        )
    return df


def calculate_performance_indicators_V2(df):
    """Compute indicators of performances from df of predictions and GT:
    - MAE2: absolute distance of predicted value to ground truth class boundaries.
    - Accuracy2: 1 if predicted value falls within class boundaries + a margin of 10%
    Note: Predicted and ground truths coverage values are ratios between 0 and 1.
    """
    # round to 3rd to avoid artefacts like 0.8999999 for 0.9 as key of dict
    df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]] = (
        df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]].astype(np.float).round(3)
    )
    # MAE2 errors
    df["error2_veg_b"] = df.apply(
        lambda x: compute_mae2(x.pred_veg_b, x.vt_veg_b), axis=1
    )
    df["error2_veg_moy"] = df.apply(
        lambda x: compute_mae2(x.pred_veg_moy, x.vt_veg_moy), axis=1
    )
    df["error2_veg_h"] = df.apply(
        lambda x: compute_mae2(x.pred_veg_h, x.vt_veg_h), axis=1
    )
    df["error2_veg_b_and_moy"] = df[["error2_veg_b", "error2_veg_moy"]].mean(axis=1)
    df["error2_all"] = df[["error2_veg_b", "error2_veg_moy", "error2_veg_h"]].mean(
        axis=1
    )

    # Accuracy 2
    df["acc2_veg_b"] = df.apply(
        lambda x: compute_accuracy2(x.pred_veg_b, x.vt_veg_b),
        axis=1,
    )
    df["acc2_veg_moy"] = df.apply(
        lambda x: compute_accuracy2(x.pred_veg_moy, x.vt_veg_moy),
        axis=1,
    )
    df["acc2_veg_h"] = df.apply(
        lambda x: compute_accuracy2(x.pred_veg_h, x.vt_veg_h),
        axis=1,
    )
    df["acc2_veg_b_and_moy"] = df[["acc2_veg_b", "acc2_veg_moy"]].mean(axis=1)
    df["acc2_all"] = df[["acc2_veg_b", "acc2_veg_moy", "acc2_veg_h"]].mean(axis=1)

    return df


def calculate_performance_indicators_V3(df):
    """Compute indicators of performances from df of predictions and GT:
    - MAE3: absolute distance of predicted value to next class boundaries.
    - Accuracy3: 1 if predicted value falls within class boundaries + neighboor classes
    Note: Predicted and ground truths coverage values are ratios between 0 and 1.
    """
    # round to 3rd to avoid artefacts like 0.8999999 for 0.9 as key of dict
    df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]] = (
        df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]].astype(np.float).round(3)
    )
    # MAE3 errors
    df["error3_veg_b"] = df.apply(
        lambda x: compute_mae3(x.pred_veg_b, x.vt_veg_b), axis=1
    )
    df["error3_veg_moy"] = df.apply(
        lambda x: compute_mae3(x.pred_veg_moy, x.vt_veg_moy), axis=1
    )
    df["error3_veg_h"] = df.apply(
        lambda x: compute_mae3(x.pred_veg_h, x.vt_veg_h), axis=1
    )
    df["error3_veg_b_and_moy"] = df[["error3_veg_b", "error3_veg_moy"]].mean(axis=1)
    df["error3_all"] = df[["error3_veg_b", "error2_veg_moy", "error3_veg_h"]].mean(
        axis=1
    )

    # Accuracy 3
    df["acc3_veg_b"] = df.apply(
        lambda x: compute_accuracy3(x.pred_veg_b, x.vt_veg_b),
        axis=1,
    )
    df["acc3_veg_moy"] = df.apply(
        lambda x: compute_accuracy3(x.pred_veg_moy, x.vt_veg_moy),
        axis=1,
    )
    df["acc3_veg_h"] = df.apply(
        lambda x: compute_accuracy3(x.pred_veg_h, x.vt_veg_h),
        axis=1,
    )
    df["acc3_veg_b_and_moy"] = df[["acc3_veg_b", "acc3_veg_moy"]].mean(axis=1)
    df["acc3_all"] = df[["acc3_veg_b", "acc3_veg_moy", "acc3_veg_h"]].mean(axis=1)

    return df


def log_confusion_matrices(args, prediction_summaries):
    """Apply log_confusion_matrix for all three strata."""
    for strata in ["veg_b", "veg_moy", "veg_h"]:
        log_confusion_matrix(args, prediction_summaries, strata)


def log_confusion_matrix(args, prediction_summaries, strata):
    """From a list of prediction summary (as produced by get_cloud_prediction_summary), log a CM to comet."""

    cm = compute_confusion_matrix(prediction_summaries, strata)

    labels = [format_float_as_percentage(center) for center in bins_centers]
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, (ax) = plt.subplots(1, 1, figsize=np.array([1, 1]) * 15)
    filename = strata
    title = (
        filename
        + f" [N={len(prediction_summaries)}] \n (fold={args.current_fold_id}|epoch={args.current_epoch})"
    )
    plt.title(title)
    cm_display.plot(
        ax=ax, cmap=plt.get_cmap("Blues"), colorbar=False, values_format=".0%"
    )
    plt.tight_layout()

    output_path = os.path.join(
        args.stats_path, f"img/confusion_matrices/{filename}.png"
    )
    create_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=100)
    plt.clf()

    args.experiment.log_image(output_path, step=args.current_epoch)


def compute_confusion_matrix(prediction_summaries, strata):
    """From a list of prediction summary (as produced by get_cloud_prediction_summary), compute a confusion matrix."""
    y_true = np.array([item["vt_" + strata] for item in prediction_summaries])
    y_predicted = np.array([item["pred_" + strata] for item in prediction_summaries])

    y_true = np.vectorize(get_closest_class_center_index)(y_true)
    y_predicted = np.vectorize(get_closest_class_center_index)(y_predicted)
    cm = confusion_matrix(
        y_true, y_predicted, labels=range(len(bins_centers)), normalize="true"
    )
    return cm


# We compute all possible mean stats per loss for all folds
def stats_for_all_folds(all_folds_loss_train_lists, all_folds_loss_test_lists, args):
    """
    all_folds_loss_[train/test]_lists : lists of n_folds dictionnaries of losses (the results of full_train) where
    an epoch key is present.
    """
    experiment = args.experiment

    # TRAIN : average all by epoch and print last record stats
    with experiment.context_manager(f"train_mean"):
        df = pd.DataFrame(
            data=reduce(lambda l1, l2: l1 + l2, all_folds_loss_train_lists)
        )
        df = df.groupby("step").mean()
        for step, metrics in df.to_dict("index").items():
            experiment.log_metrics(metrics, epoch=metrics["epoch"], step=step)

        last_mean = df[df.index == df.index.max()].to_dict("records")[0]
        total_loss = last_mean["total_loss"]
        MAE_loss = last_mean["MAE_loss"]
        log_loss = last_mean["log_loss"]
        logging.info(
            "MEAN - Train Loss: %1.2f Train Loss Abs (MAE): %1.2f Train Loss Log: %1.2f Train Los"
            % (
                total_loss,
                MAE_loss,
                log_loss,
            ),
        )

    # TEST : average all by epoch and print last record stats
    with experiment.context_manager(f"val_mean"):
        df = pd.DataFrame(
            data=reduce(lambda l1, l2: l1 + l2, all_folds_loss_test_lists)
        )
        df = df.groupby("step").mean()
        for step, metrics in df.to_dict("index").items():
            experiment.log_metrics(metrics, epoch=metrics["epoch"], step=step)
        last_mean = df[df.index == df.index.max()].to_dict("records")[0]
        total_loss = last_mean["total_loss"]
        MAE_loss = last_mean["MAE_loss"]
        log_loss = last_mean["log_loss"]
        logging.info(
            "MEAN - Validation Loss: %1.2f Loss Abs (MAE): %1.2f Loss Log: %1.2f Loss"
            % (
                total_loss,
                MAE_loss,
                log_loss,
            ),
        )
        MAE_veg_b = last_mean["MAE_veg_b"]
        MAE_veg_moy = last_mean["MAE_veg_moy"]
        MAE_veg_h = last_mean["MAE_veg_h"]
        logging.info(
            "MEAN - Validation MAE: Vb : %1.2f Vm : %1.2f Vh: %1.2f"
            % (
                MAE_veg_b,
                MAE_veg_moy,
                MAE_veg_h,
            ),
        )


# We log the loss stats per fold
def log_last_stats_of_fold(
    all_epochs_train_loss_dict,
    all_epochs_test_loss_dict,
    args,
):
    last_dict_train = max(all_epochs_train_loss_dict, key=lambda x: x["epoch"])
    total_loss = last_dict_train["total_loss"]
    MAE_loss = last_dict_train["MAE_loss"]
    log_loss = last_dict_train["log_loss"]
    logging.info(
        "Fold %3d Train Loss: %1.2f Train Loss Abs (MAE): %1.2f Train Loss Log: %1.2f"
        % (
            args.current_fold_id,
            total_loss,
            MAE_loss,
            log_loss,
        ),
    )

    last_dict_test = max(all_epochs_test_loss_dict, key=lambda x: x["epoch"])

    total_loss = last_dict_test["total_loss"]
    MAE_loss = last_dict_test["MAE_loss"]
    log_loss = last_dict_test["log_loss"]
    logging.info(
        "Fold %3d Test Loss: %1.2f Test Loss Abs (MAE): %1.2f Test Loss Log: %1.2f"
        % (
            args.current_fold_id,
            total_loss,
            MAE_loss,
            log_loss,
        ),
    )


def print_epoch_losses(i_epoch, epoch_loss_dict, train):
    """Log epoch losses."""
    NORMALCOLOR = "\033[0m"
    if train:
        COLOR = "\033[100m"
        task = "train"
    else:
        COLOR = "\033[104m"
        task = "test"

    total_loss = epoch_loss_dict["total_loss"]
    MAE_loss = epoch_loss_dict["MAE_loss"]
    log_loss = epoch_loss_dict["log_loss"]
    logger.info(
        COLOR
        + "Epoch %3d -> %s Loss: %1.2f %s Loss Abs (MAE): %1.2f %s Loss Log: %1.2f %s"
        % (
            i_epoch,
            task,
            total_loss,
            task,
            MAE_loss,
            task,
            log_loss,
            task,
        )
        + NORMALCOLOR
    )


def post_cross_validation_logging(
    summary_context_name,
    all_folds_loss_train_dicts,
    all_folds_loss_test_dicts,
    cloud_info_list_by_fold,
    args,
):
    stats_for_all_folds(all_folds_loss_train_dicts, all_folds_loss_test_dicts, args)
    cloud_info_list_all_folds = [
        dict(p, **{"fold_id": args.current_fold_id})
        for args.current_fold_id, infos in cloud_info_list_by_fold.items()
        for p in infos
    ]
    df_inference = pd.DataFrame(cloud_info_list_all_folds)
    try:
        df_inference = calculate_performance_indicators_V1(df_inference)
        df_inference = calculate_performance_indicators_V2(df_inference)
        df_inference = calculate_performance_indicators_V3(df_inference)
    except KeyError:
        logger.info(
            "Cannot calculate class-based performance indicators due to continuous ground truths."
        )

    inference_path = os.path.join(
        args.stats_path, f"PCC_inference_all_placettes_{summary_context_name}.csv"
    )
    df_inference.to_csv(inference_path, index=False)

    with args.experiment.context_manager(summary_context_name):
        m = df_inference.mean().to_dict()
        args.experiment.log_metrics(m)
        args.experiment.log_table(inference_path)
        log_confusion_matrices(
            args,
            cloud_info_list_all_folds,
        )

    logger.info(f"Saved infered, cross-validated results to {inference_path}")
