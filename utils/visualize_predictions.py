import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec

import numpy as np
import torch
from osgeo import gdal, osr
import torch.nn as nn
import logging

from inference.infer_utils import (
    stack_the_rasters_and_get_their_geotransformation,
    infer_and_project_on_rasters,
)

from inference.geotiff_raster import save_rasters_to_geotiff_file

logger = logging.getLogger(__name__)

plt.rcParams["font.size"] = 25


def visualize_article(
    image_soil, image_med_veg, image_high_veg, cloud, pl_id, stats_path, args, txt=None
):

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3)

    # Original data
    ax1 = fig.add_subplot(gs[:, 0:2], projection="3d")

    # Fake color to see vegetation more clearly
    nir_r_g_indexes = [6, 3, 4]
    c = cloud[nir_r_g_indexes].numpy().transpose()

    # TODO: clean here
    # # NDVI calculation
    # r_infra = cloud[[3, 6]].numpy().transpose()
    # r = r_infra[:, 0]
    # infra = r_infra[:, 1]
    # ndvi = (infra - r) / (infra + r)
    # top = cm.get_cmap("Blues_r", 128)
    # bottom = cm.get_cmap("Greens", 128)
    # cmap = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    # cmap = colors.ListedColormap(cmap, name="GreensBlues")

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
    ax1.set_title(pl_id)
    for line in ax1.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax1.yaxis.get_ticklines():
        line.set_visible(False)

    # LV stratum raster
    ax2 = fig.add_subplot(gs[0, 2])
    color_grad = [(0.8, 0.4, 0.1), (0, 1, 0)]  # first color is white, last is green
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax2.imshow(image_soil, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title("Ground level")
    ax2.tick_params(
        axis="both",  # changes apply to both axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    # MV stratum raster
    ax3 = fig.add_subplot(gs[1, 2])
    color_grad = [(1, 1, 1), (0, 1, 0)]  # first color is white, last is green
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax3.imshow(image_med_veg, cmap=cmap, vmin=0, vmax=1)
    ax3.set_title("Medium level")
    ax3.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    # Plot high vegetation stratum
    ax4 = fig.add_subplot(gs[2, 2])
    color_grad = [(1, 1, 1), (0, 1, 0)]  # first color is white, last is red
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax4.imshow(image_high_veg, cmap=cmap, vmin=0, vmax=1)
    ax4.set_title("High level")
    ax4.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])

    if txt is not None:
        fig.text(0.5, 0.05, txt, ha="center")
    plt.savefig(
        stats_path + pl_id + "_article.svg", format="svg", bbox_inches="tight", dpi=50
    )
    plt.clf()
    plt.close("all")


def visualize(
    image_low_veg,
    image_med_veg,
    cloud,
    prediction,
    pl_id,
    stats_path,
    args,
    text_pred_vs_gt=None,
    p_all_pdf_all=None,
    image_high_veg=None,
    predictions=["Vb", "soil", "Vm", "Vh", "Adm"],
    gt=["Vb", "soil", "Vm", "Vh", "Adm"],
):

    if image_low_veg.ndim == 3:
        image_low_veg = image_low_veg[:, :, 0]
        image_med_veg = image_med_veg[:, :, 0]

    # We set figure size depending on the number of subplots
    row, col = 3, 2
    fig = plt.figure(figsize=(20, 25))

    # Original point data
    ax1 = fig.add_subplot(row, col, 1, projection="3d")

    # Fake color to see vegetation more clearly
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
        axis="both",  # changes apply to both axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
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
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    PCM = ax4.get_children()[9]
    plt.colorbar(PCM, ax=ax4)

    # Plot stratum scores
    ax5 = fig.add_subplot(row, col, 5, projection="3d")
    ax5.auto_scale_xyz
    # colors_pred = likelihood_norm[:, [0, 0, 0]].cpu().detach().numpy()
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
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
        )  # labels along the bottom edge are off
        ax6.set_yticklabels([])
        ax6.set_xticklabels([])
        PCM = ax6.get_children()[9]
        plt.colorbar(PCM, ax=ax6)

    if text_pred_vs_gt is not None:
        fig.text(0.5, 0.05, text_pred_vs_gt, ha="center")
    save_path = stats_path + pl_id + ".png"
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=50)
    plt.clf()
    plt.close("all")
    return save_path


@torch.no_grad()
def create_final_images(
    pred_pl,
    gt,
    pred_pointwise_b,
    cloud,
    likelihood_norm,
    plot_name,
    xy_centers_dict,
    plot_path,
    args,
    plot_only_png=True,
    adm=None,
):
    """
    We do final data reprojection to the 2D space (2 stratum - ground vegetation level and medium level, optionally high level)
    by associating the points to the pixels.
    Then we create the images with those stratum
    """
    # we get prediction stats string
    pred_pointwise = pred_pointwise_b[0]
    current_cloud = cloud[0]  # (9, N) tensor
    plot_center = xy_centers_dict[plot_name]  # tuple (x,y)

    # we do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
    image_low_veg, image_med_veg, image_high_veg = infer_and_project_on_rasters(
        current_cloud, pred_pointwise, args
    )

    if args.adm:
        preds_nparray = np.round(
            np.asarray(pred_pl[0].cpu().detach().numpy().reshape(-1)), 2
        )
        adm_ = adm[0].cpu().detach().numpy().round(2)
        gt_nparray = gt.cpu().numpy()[0]
        text_pred_vs_gt = (
            f"Coverage: Pred {preds_nparray[:4]} GT   {gt_nparray[:-1]}\n "
            + f"Admissibility: Pred {adm_:.2f}  GT  {gt_nparray[-1]:.2f}"
        )
    else:
        preds_nparray = np.round(
            np.asarray(pred_pl[0].cpu().detach().numpy().reshape(-1)), 2
        )
        gt_nparray = gt.cpu().numpy()[0]
        text_pred_vs_gt = (
            f"Coverage: Pred {preds_nparray[:4]} GT   {gt_nparray[:-1]}\n "
            + f"Admissibility (GT only) {gt_nparray[-1]:.2f}"
        )

    text_pred_vs_gt = "LOW, soil, MID, HIGH \n" + text_pred_vs_gt
    logger.info("\n" + plot_name + " " + text_pred_vs_gt)
    # We create an image with 5 or 6 subplots:
    # 1. original point cloud, 2. LV image, 3. pointwise prediction point cloud, 4. MV image, 5.Stratum probabilities point cloud, 6.(optional) HV image
    png_path = visualize(
        image_low_veg,
        image_med_veg,
        current_cloud,
        pred_pointwise,
        plot_name,
        plot_path,
        args,
        text_pred_vs_gt=text_pred_vs_gt,
        p_all_pdf_all=likelihood_norm,
        image_high_veg=image_high_veg,
        predictions=preds_nparray,
        gt=gt_nparray,
    )
    if args.current_fold_id >= 0:
        args.experiment.log_image(png_path, overwrite=True)

    if not plot_only_png:

        img_to_write, geo = stack_the_rasters_and_get_their_geotransformation(
            plot_center,
            args,
            image_low_veg,
            image_med_veg,
            image_high_veg,
        )
        save_rasters_to_geotiff_file(
            nb_channels=args.nb_stratum,
            new_tiff_name=plot_path + plot_name + ".tif",
            width=args.diam_pix,
            height=args.diam_pix,
            datatype=gdal.GDT_Float32,
            data_array=img_to_write,
            geotransformation=geo,
        )
        if args.nb_stratum == 3:
            visualize_article(
                image_low_veg,
                image_med_veg,
                image_high_veg,
                current_cloud,
                plot_name,
                plot_path,
                args,
                txt=text_pred_vs_gt,
            )
    return png_path