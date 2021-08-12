# We import from other files
import imp

from comet_ml import Experiment
from torch.optim import optimizer
from utils.visualize_predictions import *
from data_loader.loader import *
from model.project_to_2d import *
from learning.loss_functions import *
from learning.accuracy import (
    get_closest_class_center_index,
    bins_centers,
    log_confusion_matrices,
)
from utils.utils import create_dir
import torchnet as tnt
import gc
import os
from PIL import Image

np.random.seed(42)


@torch.no_grad()
def evaluate(
    model,
    test_set,
    args,
    last_epoch=False,
):
    """Eval on test set and inference if this is the last epoch
    Outputs are: average losses (printed), infered values (csv) , k trained models, stats, and images.
    Everything is saved under /experiments/ folder.
    """

    model.eval()

    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs_gl = tnt.meter.AverageValueMeter()
    loss_meter_abs_ml = tnt.meter.AverageValueMeter()
    loss_meter_abs_hl = tnt.meter.AverageValueMeter()

    cloud_prediction_summaries = []
    last_G_tensor_list = []
    for cloud_data in loader:

        plot_center = cloud_data["plot_center"][0]
        plot_name = cloud_data["plot_id"][0]
        clouds = cloud_data["cloud"]
        gt_coverages = cloud_data["coverages"]
        if args.cuda is not None:
            gt_coverages = gt_coverages.cuda(args.cuda)

        coverages_pointwise, proba_pointwise = model(clouds)
        pred_pl = project_to_plotwise_coverages(coverages_pointwise, clouds, args)

        loss_abs = get_absolute_loss(pred_pl, gt_coverages)
        loss_log, p_all_pdf_all = get_NLL_loss(proba_pointwise, clouds, args)

        loss_e = get_entropy_loss(proba_pointwise)
        loss = loss_abs + args.m * loss_log + args.e * loss_e

        loss_meter.add(loss.item())
        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        gc.collect()

        component_losses = get_absolute_loss_by_strata(pred_pl, gt_coverages)
        loss_abs_gl, loss_abs_ml, loss_abs_hl = component_losses
        loss_meter_abs_gl.add(loss_abs_gl.item())
        loss_meter_abs_hl.add(loss_abs_hl.item())
        loss_meter_abs_ml.add(loss_abs_ml.item())

        if last_epoch or plot_name in args.plot_name_to_visualize_during_training:
            png_path = create_predictions_interpretations(
                pred_pl,
                gt_coverages,
                coverages_pointwise[0],
                clouds[0],
                p_all_pdf_all,
                plot_name,
                plot_center,
                args,
            )

        pred_pl_cpu = pred_pl.cpu().numpy()[0]
        gt_coverages_cpu = gt_coverages.cpu().numpy()[0]
        cloud_prediction_summary = get_cloud_prediction_summary(
            plot_name, pred_pl_cpu, gt_coverages_cpu, coverages_pointwise
        )
        cloud_prediction_summaries.append(cloud_prediction_summary)

        if last_epoch and isinstance(args.experiment, Experiment):
            last_G_tensor_list.append(
                [model.last_G_tensor.cpu().numpy(), plot_name, png_path]
            )

    if last_epoch or (
        (args.current_epoch % args.log_confusion_matrix_frequency == 0)
        and (args.log_confusion_matrix_frequency > 0)
    ):
        log_confusion_matrices(args, cloud_prediction_summaries)

    if last_epoch:
        if isinstance(args.experiment, Experiment):
            log_MAE_histograms(args, cloud_prediction_summaries)
            log_embeddings(last_G_tensor_list, args)

    return (
        {
            "total_loss": loss_meter.value()[0],
            "MAE_loss": loss_meter_abs.value()[0],
            "log_loss": loss_meter_log.value()[0],
            "MAE_veg_b": loss_meter_abs_gl.value()[0],
            "MAE_veg_moy": loss_meter_abs_ml.value()[0],
            "MAE_veg_h": loss_meter_abs_hl.value()[0],
            "step": args.current_step_in_fold,
        },
        cloud_prediction_summaries,
    )


def get_cloud_prediction_summary(
    plot_name, pred_pl_cpu, gt_coverages_cpu, coverages_pointwise
):
    return {
        "pl_id": plot_name,
        "pl_N_points": coverages_pointwise.shape[1],
        "pred_veg_b": pred_pl_cpu[0],
        "pred_sol_nu": pred_pl_cpu[1],
        "pred_veg_moy": pred_pl_cpu[2],
        "pred_veg_h": pred_pl_cpu[3],
        "vt_veg_b": gt_coverages_cpu[0],
        "vt_sol_nu": gt_coverages_cpu[1],
        "vt_veg_moy": gt_coverages_cpu[2],
        "vt_veg_h": gt_coverages_cpu[3],
    }


def log_embeddings(last_G_tensor_list, args):
    image_data = [
        Image.open(a[2]).convert("RGB").resize((100, 160)) for a in last_G_tensor_list
    ]
    args.experiment.log_embedding(
        [a[0] for a in last_G_tensor_list],
        [a[1] for a in last_G_tensor_list],
        image_data=image_data,
        image_transparent_color=(0, 0, 0),
        image_size=image_data[0].size,
        title="G_tensor",
    )


def log_MAE_histograms(args, cloud_prediction_summaries):
    args.experiment.log_histogram_3d(
        [
            abs(info["pred_veg_b"] - info["vt_veg_b"])
            for info in cloud_prediction_summaries
        ],
        name="val_MAE_veg_b",
        step=args.current_fold_id,
        epoch=args.current_epoch,
    )
    args.experiment.log_histogram_3d(
        [
            abs(info["pred_veg_moy"] - info["vt_veg_moy"])
            for info in cloud_prediction_summaries
        ],
        name="val_MAE_veg_moy",
        step=args.current_fold_id,
        epoch=args.current_epoch,
    )
    args.experiment.log_histogram_3d(
        [
            abs(info["pred_veg_h"] - info["vt_veg_h"])
            for info in cloud_prediction_summaries
        ],
        name="val_MAE_veg_h",
        step=args.current_fold_id,
        epoch=args.current_epoch,
    )
