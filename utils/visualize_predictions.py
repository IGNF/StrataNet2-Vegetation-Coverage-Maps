import numpy as np
import torch
from osgeo import gdal, osr
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec

from inference.infer_utils import (
    get_geotransform,
    infer_and_project_on_rasters,
)
from utils.utils import create_dir
from inference.geotiff_raster import save_rasters_to_geotiff_file

logger = logging.getLogger(__name__)

plt.rcParams["font.size"] = 25


@torch.no_grad()
def create_predictions_interpretations(
    pred_pl,
    gt,
    coverages_pointwise,
    clouds,
    likelihood_norm,
    plot_name,
    xy_centers_dict,
    args,
):
    """
    We do final data reprojection to the 2D space by associating the points to the pixels.
    """

    pred_pointwise = coverages_pointwise[0]
    current_cloud = clouds[0]
    plot_center = xy_centers_dict[plot_name]

    rasters = infer_and_project_on_rasters(current_cloud, pred_pointwise, args)

    text_pred_vs_gt, preds_nparray, gt_nparray = get_pred_summary_text(pred_pl, gt)

    text_pred_vs_gt = "LOW, soil, MID, HIGH \n" + text_pred_vs_gt
    logger.info("\n" + plot_name + " " + text_pred_vs_gt)
    # We create an image with 5 or 6 subplots:
    # 1. original point cloud, 2. LV image, 3. pointwise prediction point cloud, 4. MV image, 5.Stratum probabilities point cloud, 6.(optional) HV image
    png_path = visualize(
        rasters,
        current_cloud,
        pred_pointwise,
        plot_name,
        args,
        text_pred_vs_gt=text_pred_vs_gt,
        p_all_pdf_all=likelihood_norm,
        predictions=preds_nparray,
        gt=gt_nparray,
    )
    cross_validating = args.current_fold_id >= 0
    if cross_validating:
        args.experiment.log_image(png_path, overwrite=True)

    if args.plot_geotiff_file:

        geo = get_geotransform(
            plot_center,
            args,
        )
        save_rasters_to_geotiff_file(
            nb_channels=3,
            new_tiff_name=args.plot_path + plot_name + ".tif",
            width=args.diam_pix,
            height=args.diam_pix,
            datatype=gdal.GDT_Float32,
            data_array=rasters,
            geotransformation=geo,
        )

    return png_path


def visualize(
    rasters,
    cloud,
    prediction,
    pl_id,
    args,
    text_pred_vs_gt=None,
    p_all_pdf_all=None,
    predictions=["Vb", "soil", "Vm", "Vh"],
    gt=["Vb", "soil", "Vm", "Vh", "Adm"],
):
    (
        image_low_veg,
        image_med_veg,
        image_high_veg,
    ) = rasters

    row, col = 3, 2
    fig = plt.figure(figsize=(20, 25))

    # Original point data with fake NIR colors
    ax1 = fig.add_subplot(row, col, 1, projection="3d")
    nir_r_g_indexes = [6, 3, 4]
    c = cloud[nir_r_g_indexes].numpy().transpose()
    ax1.scatter(
        cloud[0],
        cloud[1],
        cloud[2] * args.z_max,
        c=c,
        vmin=0,
        vmax=1,
        s=10,
        alpha=1,
    )
    ax1.auto_scale_xyz
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title(f"{pl_id}")

    # LV stratum raster
    ax2 = fig.add_subplot(row, col, 2)
    color_grad = [
        (0.8, 0.4, 0.1),
        (0.91, 0.91, 0.91),
        (0, 1, 0),
    ]  # first color is brown, second is grey, last is green
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax2.imshow(image_low_veg, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title(f"Low veg. = {predictions[0]:.0%} (gt={gt[0]:.0%})")
    ax2.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
    )
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    PCM = ax2.get_children()[9]
    plt.colorbar(PCM, ax=ax2)

    # Pointwise prediction
    ax3 = fig.add_subplot(row, col, 3, projection="3d")
    ax3.auto_scale_xyz
    colors_pred = prediction.cpu().detach().numpy().transpose()
    color_matrix = [[0, 1, 0], [0.8, 0.4, 0.1], [0, 0, 1], [1, 0, 0]]
    colors_pred = np.matmul(colors_pred, color_matrix)
    ax3.scatter(
        cloud[0],
        cloud[1],
        cloud[2] * args.z_max,
        c=colors_pred,
        s=10,
        vmin=0,
        vmax=1,
        alpha=1,
    )
    ax3.set_title("Pointwise prediction")
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    # MV stratum raster
    ax4 = fig.add_subplot(row, col, 4)
    color_grad = [(1, 1, 1), (0, 0, 1)]  # first color is white, last is blue
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax4.imshow(image_med_veg, cmap=cmap, vmin=0, vmax=1)
    ax4.set_title(f"Medium veg. = {predictions[2]:.0%} (gt={gt[2]:.0%})")
    ax4.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
    )
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    PCM = ax4.get_children()[9]
    plt.colorbar(PCM, ax=ax4)

    # Plot stratum scores
    ax5 = fig.add_subplot(row, col, 5, projection="3d")
    ax5.auto_scale_xyz
    p_all, pdf_all = p_all_pdf_all
    p_all = p_all.cpu().detach().numpy()
    pdf_all = pdf_all.cpu().detach().numpy()
    colors_pred = p_all[pdf_all.argmax(axis=1)[:, None] == range(pdf_all.shape[1])]
    colors_pred = colors_pred / (colors_pred.max() + 0.00001)
    ax5.scatter(
        cloud[0],
        cloud[1],
        cloud[2] * args.z_max,
        c=colors_pred,
        s=10,
        vmin=0,
        vmax=1,
        cmap=plt.get_cmap("copper"),
    )
    ax5.set_title("Score for most-likely strata")
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])

    # Plot high vegetation stratum
    if image_high_veg is not None:
        ax6 = fig.add_subplot(row, col, 6)
        color_grad = [(1, 1, 1), (1, 0, 0)]  # first color is white, last is red
        cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
        ax6.imshow(image_high_veg, cmap=cmap, vmin=0, vmax=1)
        ax6.set_title(f"High veg. = {predictions[3]:.0%} (gt={gt[3]:.0%})")
        ax6.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
        )
        ax6.set_yticklabels([])
        ax6.set_xticklabels([])
        PCM = ax6.get_children()[9]
        plt.colorbar(PCM, ax=ax6)

    if text_pred_vs_gt is not None:
        fig.text(0.5, 0.05, text_pred_vs_gt, ha="center")

    create_dir(args.plot_path)
    save_path = args.plot_path + pl_id + ".png"
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=50)
    plt.clf()
    plt.close("all")
    return save_path


def get_pred_summary_text(pred_pl, gt):
    preds_nparray = np.round(
        np.asarray(pred_pl[0].cpu().detach().numpy().reshape(-1)), 2
    )
    gt_nparray = gt.cpu().numpy()[0]
    text_pred_vs_gt = (
        f"Coverage: Pred {preds_nparray[:4]} GT   {gt_nparray[:-1]}\n "
        + f"Admissibility (GT only) {gt_nparray[-1]:.2f}"
    )
    return text_pred_vs_gt, preds_nparray, gt_nparray
