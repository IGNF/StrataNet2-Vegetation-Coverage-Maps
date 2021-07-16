import numpy as np
import pandas as pd
from functools import reduce
import logging

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

# TODO: use this in metric computation
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


# we derive indicators of performance
def calculate_performance_indicators(df):
    """Compute indicators of performances from df of predictions and GT:
    - MAE: absolute distance of predicted value to ground truth
    - Accuracy: 1 if predicted value falls within class boundaries
    - MAE2: absolute distance of predicted value to ground truth class boundaries.
    - Accuracy2: 1 if predicted value falls within class boundaries + a margin of 10%
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
        adm_loss = last_mean["adm_loss"]
        logging.info(
            "MEAN - Train Loss: %1.2f Train Loss Abs (MAE): %1.2f Train Loss Log: %1.2f Train Loss Adm: %1.2f"
            % (
                total_loss,
                MAE_loss,
                log_loss,
                adm_loss if adm_loss is not np.nan else 0,
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
        adm_loss = last_mean["adm_loss"]
        logging.info(
            "MEAN - Validation Loss: %1.2f Loss Abs (MAE): %1.2f Loss Log: %1.2f Loss Adm: %1.2f"
            % (
                total_loss,
                MAE_loss,
                log_loss,
                adm_loss if adm_loss is not np.nan else 0,
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
    fold_id,
    args,
):
    last_dict_train = max(all_epochs_train_loss_dict, key=lambda x: x["epoch"])
    total_loss = last_dict_train["total_loss"]
    MAE_loss = last_dict_train["MAE_loss"]
    log_loss = last_dict_train["log_loss"]
    adm_loss = last_dict_train["adm_loss"]
    logging.info(
        "Fold %3d Train Loss: %1.2f Train Loss Abs (MAE): %1.2f Train Loss Log: %1.2f Train Loss Adm: %1.2f"
        % (
            fold_id,
            total_loss,
            MAE_loss,
            log_loss,
            adm_loss if adm_loss is not np.nan else 0,
        ),
    )

    last_dict_test = max(all_epochs_test_loss_dict, key=lambda x: x["epoch"])

    total_loss = last_dict_test["total_loss"]
    MAE_loss = last_dict_test["MAE_loss"]
    log_loss = last_dict_test["log_loss"]
    adm_loss = last_dict_test["adm_loss"]
    logging.info(
        "Fold %3d Test Loss: %1.2f Test Loss Abs (MAE): %1.2f Test Loss Log: %1.2f Test Loss Adm: %1.2f"
        % (
            fold_id,
            total_loss,
            MAE_loss,
            log_loss,
            adm_loss if adm_loss is not np.nan else 0,
        ),
    )

    return last_dict_train, last_dict_test


# We perform tensorboard visualisation by writing the stats to the writer
# TODO: get rid of this as we use comet_ml
def write_to_writer(writer, args, i_epoch, epoch_loss_dict, train):
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
    adm_loss = epoch_loss_dict["adm_loss"]
    logging.info(
        COLOR
        + "Epoch %3d -> %s Loss: %1.2f %s Loss Abs (MAE): %1.2f %s Loss Log: %1.2f %s Loss Adm: %1.2f"
        % (
            i_epoch,
            task,
            total_loss,
            task,
            MAE_loss,
            task,
            log_loss,
            task,
            adm_loss if adm_loss is not None else 0,
        )
        + NORMALCOLOR
    )
    writer.add_scalar(f"Loss/{task}/abs_adm", adm_loss, i_epoch + 1)
    writer.add_scalar(f"Loss/{task}_total", total_loss, i_epoch + 1)
    writer.add_scalar(f"Coverage_MAE/{task}", MAE_loss, i_epoch + 1)
    writer.add_scalar(f"Loss/{task}_log", log_loss, i_epoch + 1)

    return writer
