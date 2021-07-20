# Dependency imports
import os
import glob
from math import cos, pi, ceil
import numpy as np
import torch
import numpy.ma as ma  # masked array
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import getsizeof
import rasterio
from rasterio.merge import merge
import rasterio.features
from rasterio.transform import Affine
from shapely import geometry
import shapefile
from shapely.geometry.point import Point
import torch.nn as nn

# We import from other files
from utils.utils import (
    create_dir,
    get_filename_no_extension,
    get_files_of_type_in_folder,
)
from utils.load_data import (
    load_and_clean_single_las,
    transform_features_of_plot_cloud,
)
from data_loader.loader import rescale_cloud_data

sns.set()
np.random.seed(42)

import logging

logger = logging.getLogger(__name__)


def infer_and_project_on_rasters(current_cloud, pred_pointwise, args):
    """
    We do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
    current_cloud: (2, N) 2D tensor
     image_low_veg, image_med_veg, image_high_veg
    """

    # we get unique pixel coordinate to serve as group for raster prediction
    # Values are between 0 and args.diam_pix-1, sometimes (extremely rare) at args.diam_pix wich we correct

    scaling_factor = 10 * (args.diam_pix / args.diam_meters)  # * pix/normalized_unit
    xy = current_cloud[:2, :]
    xy = (
        torch.floor(
            (xy + 0.0001) * scaling_factor
            + torch.Tensor(
                [[args.diam_meters // 2], [args.diam_meters // 2]]
            ).expand_as(xy)
        )
    ).int()
    xy = torch.clip(xy, 0, args.diam_pix - 1)
    xy = xy.cpu().numpy()
    _, _, inverse = np.unique(xy.T, axis=0, return_index=True, return_inverse=True)

    # we get the values for each unique pixel and write them to rasters
    image_low_veg = np.full((args.diam_pix, args.diam_pix), np.nan)
    image_med_veg = np.full((args.diam_pix, args.diam_pix), np.nan)
    if args.nb_stratum == 3:
        image_high_veg = np.full((args.diam_pix, args.diam_pix), np.nan)
    else:
        image_high_veg = None
    for i in np.unique(inverse):
        where = np.where(inverse == i)[0]
        k, m = xy.T[where][0]
        maxpool = nn.MaxPool1d(len(where))
        max_pool_val = (
            maxpool(pred_pointwise[:, where].unsqueeze(0))
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )
        sum_val = pred_pointwise[:, where].sum(axis=1)

        if args.norm_ground:  # we normalize ground level coverage values
            proba_low_veg = sum_val[0] / (sum_val[:2].sum())
        else:  # we do not normalize anything, as bare soil coverage does not participate in absolute loss
            proba_low_veg = max_pool_val[0]
        image_low_veg[m, k] = proba_low_veg

        proba_med_veg = max_pool_val[2]
        image_med_veg[m, k] = proba_med_veg

        if args.nb_stratum == 3:
            proba_high_veg = max_pool_val[3]
            image_high_veg[m, k] = proba_high_veg

    # We flip along y axis as the 1st raster row starts with 0
    image_low_veg = np.flip(image_low_veg, axis=0)
    image_med_veg = np.flip(image_med_veg, axis=0)
    if args.nb_stratum == 3:
        image_high_veg = np.flip(image_high_veg, axis=0)
    return image_low_veg, image_med_veg, image_high_veg


def stack_the_rasters_and_get_their_geotransformation(
    plot_center_xy, args, image_low_veg, image_med_veg, image_high_veg
):
    """ """
    # geotransform reference : https://gdal.org/user/raster_data_model.html
    # top_left_x, pix_width_in_meters, _, top_left_y, pix_heighgt_in_meters (neg for north up picture)

    geo = [
        plot_center_xy[0] - args.diam_meters // 2,  # xmin
        args.diam_meters / args.diam_pix,
        0,
        plot_center_xy[1] + args.diam_meters // 2,  # ymax
        0,
        -args.diam_meters / args.diam_pix,
        # negative b/c in geographic raster coordinates (0,0) is at top left
    ]

    if args.nb_stratum == 2:
        img_to_write = np.concatenate(([image_low_veg], [image_med_veg]), 0)
    else:
        img_to_write = np.concatenate(
            ([image_low_veg], [image_med_veg], [image_high_veg]), 0
        )
    return img_to_write, geo


def divide_parcel_las_and_get_disk_centers(
    args, las_filename, parcel_shape, save_fig_of_division=True
):
    """
    Identify centers of plots whose squares cover at least partially every pixel of the parcel
    We consider the square included in a plot with r=10m. Formula for width of
    the square is  W = 2 * (cos(45°) * r) since max radius in square equals r as well.
    We add an overlap of s*0.625 i.e. a pixel in currently produced plots of size 32 pix = 10
    :param las_folder: path
    :param las_filenae: "004000715-5-18.las" like string
    :param sf: shapefile of parcels
    :returns:
        centers_nparray: a nparray of centers coordinates
        points_nparray: a nparray of full cloud coordinates
    Note: outputs are not normalized
    """

    points_nparray, xy_centers = load_and_clean_single_las(las_filename)
    size_MB = getsizeof(round(getsizeof(points_nparray) / 1024 / 1024, 2))
    logger.info(f"Size of LAS file is {size_MB}MB")

    x_las, y_las = points_nparray[:, 0], points_nparray[:, 1]

    # DEBUG
    # # subsample = False
    # if subsample:
    #     subsampling = 500
    #     subset = np.random.choice(points_nparray.shape[0],size=subsampling, replace=False)
    #     x_las = x_las[subset]
    #     y_las = y_las[subset]

    x_min = x_las.min()
    y_min = y_las.min()
    x_max = x_las.max()
    y_max = y_las.max()

    # Get or calculate dimensions of disk and max square in said disk
    plot_radius_meters = 10  # This is hardcoded, but should not change at any time.
    cos_of_45_degrees = cos(pi / 4)
    within_circle_square_width_meters = 2 * cos_of_45_degrees * plot_radius_meters
    plot_diameter_in_pixels = args.diam_pix  # 32 by default
    plot_diameter_in_meters = 2 * plot_radius_meters
    s = 1  # size of overlap in pixels
    square_xy_overlap = (
        s * plot_diameter_in_meters / plot_diameter_in_pixels
    )  # 0.625 by default
    movement_in_meters = within_circle_square_width_meters - square_xy_overlap

    logger.info(
        f"Square dimensions are {within_circle_square_width_meters:.2f}m*{within_circle_square_width_meters:.2f}m"
        + f"but we move {movement_in_meters:.2f}m at a time to have {square_xy_overlap:.2f}m of overlap"
    )

    x_range_of_parcel_in_movements = ceil((x_max - x_min) / (movement_in_meters)) + 1
    y_range_of_parcel_in_movements = ceil((y_max - y_min) / (movement_in_meters)) + 1

    start_x = x_min + movement_in_meters / 4
    start_y = y_min + movement_in_meters / 4
    grid_pixel_xy_centers = [[start_x, start_y]]

    for i_dx in range(x_range_of_parcel_in_movements):
        current_x = start_x + i_dx * movement_in_meters  # move along x axis
        for i_dy in range(y_range_of_parcel_in_movements):
            current_y = start_y + i_dy * movement_in_meters  # move along y axis
            new_plot_center = [current_x, current_y]
            grid_pixel_xy_centers.append(new_plot_center)

    # Ignore plot center if not in shape of shapefile
    grid_pixel_xy_centers = [
        x
        for x in grid_pixel_xy_centers
        if parcel_shape.buffer(args.diam_meters // 2).contains(Point(x[0], x[1]))
    ]

    # visualization
    if save_fig_of_division:
        # we need to normalize coordinates points for easier visualization
        save_image_of_parcel_division_into_plots(
            args,
            las_filename,
            x_las,
            y_las,
            x_min,
            y_min,
            x_max,
            y_max,
            within_circle_square_width_meters,
            s,
            square_xy_overlap,
            grid_pixel_xy_centers,
        )

    return grid_pixel_xy_centers, points_nparray


def save_image_of_parcel_division_into_plots(
    args,
    las_filename,
    x_las,
    y_las,
    x_min,
    y_min,
    x_max,
    y_max,
    within_circle_square_width_meters,
    s,
    square_xy_overlap,
    grid_pixel_xy_centers,
):
    """
    Visualize and save to PNG file the division of a large parcel into many disk subplots.
    """
    las_id = get_filename_no_extension(las_filename)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_min_c = x_min - x_center
    x_max_c = x_max - x_center
    y_min_c = y_min - y_center
    y_max_c = y_max - y_center

    # xy to dataframe for visualization
    coordinates = np.array(np.stack([x_las - x_center, y_las - y_center], axis=1))
    coordinates = pd.DataFrame(data=coordinates)
    coordinates.columns = ["x_n", "y_n"]

    sampling_size_for_kde = (
        10000  # fixed size which could lead to poor kde in large parcels.
    )
    if len(coordinates) > sampling_size_for_kde:
        coordinates = coordinates.sample(n=sampling_size_for_kde, replace=False)

    # centers to dataframe for visualization
    centers = np.array(grid_pixel_xy_centers - np.array([x_center, y_center]))
    centers = pd.DataFrame(data=centers)
    centers.columns = ["x_n", "y_n"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"aspect": "equal"})
    ax.grid(False)
    ax.set_aspect("equal")  # Not working right now
    plt.xlim(x_min_c - 5, x_max_c + 5)
    plt.ylim(y_min_c - 5, y_max_c + 5)
    plt.ylabel("y_n", rotation=0)
    plt.title(
        f'Cutting in r=10m plots for parcel "{las_id}"'
        + f"\n Contained squares: W={within_circle_square_width_meters:.2f}m with overlap={square_xy_overlap:.2f}m (i.e. {s}pix)"
    )
    # plot kde of parcel
    fig.tight_layout()
    sns.kdeplot(
        data=coordinates,
        x="x_n",
        y="y_n",
        fill=True,
        alpha=0.5,
        color="g",
        clip=[[x_min_c, x_max_c], [y_min_c, y_max_c]],
    )  # thresh=0.2

    # plot disks and squares
    for _, (x, y) in centers.iterrows():
        a_circle = plt.Circle(
            (x, y), 10, fill=True, alpha=0.1, edgecolor="white", linewidth=1
        )
        ax.add_patch(a_circle)
        a_circle = plt.Circle((x, y), 10, fill=False, edgecolor="white", linewidth=0.3)
        ax.add_patch(a_circle)

    sns.scatterplot(data=centers, x="x_n", y="y_n", s=5)

    # plot boundaries of parcel
    plt.axhline(
        y=y_min_c,
        xmin=x_min_c,
        xmax=x_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    plt.axhline(
        y=y_max_c,
        xmin=x_min_c,
        xmax=x_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    plt.axvline(
        x=x_min_c,
        ymin=y_min_c,
        ymax=y_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    plt.axvline(
        x=x_max_c,
        ymin=y_min_c,
        ymax=y_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    # fig.show()

    cutting_plot_save_folder_path = os.path.join(args.stats_path, f"img/cuttings/")
    create_dir(cutting_plot_save_folder_path)
    cutting_plot_save_path = os.path.join(
        cutting_plot_save_folder_path, f"cut_{las_id}.png"
    )

    plt.savefig(cutting_plot_save_path, dpi=200)
    plt.clf()
    plt.close("all")


def extract_points_within_disk(points_nparray, center, radius=10):
    """From a (2, N) np.array with x, y as first features, extract points within radius
    from the center = (x_center, y_center)"""
    xy = points_nparray[:, :2]  # (N, 2)
    contained_points = points_nparray[
        ((xy - center) ** 2).sum(axis=1) <= (radius * radius)
    ]  # (N, f)

    return contained_points


# TODO: correct order of operations here.
def get_and_prepare_cloud_around_center(parcel_points_nparray, plot_center, args):
    plots_point_nparray = extract_points_within_disk(parcel_points_nparray, plot_center)

    if plots_point_nparray.shape[0] == 0:
        return None

    # TODO: for clarity: make operations on the same axes instead of transposing inbetween
    plots_point_nparray = transform_features_of_plot_cloud(plots_point_nparray, args)
    plots_point_nparray = plots_point_nparray.transpose()
    plots_point_nparray = rescale_cloud_data(plots_point_nparray, plot_center, args)

    # add a batch dim before trying out dataloader
    # TODO: remove this useless batch dim (or use a DataLoader...)
    plots_point_nparray = np.expand_dims(plots_point_nparray, axis=0)
    plot_points_tensor = torch.from_numpy(plots_point_nparray)
    return plot_points_tensor


def log_inference_times(plot_name, timer, shp_records, file_append_mode):
    """
    Add a row to file with the time of inference contained in Timer object t.
    """
    times_row = {plot_name: {task: np.round(d, 1) for task, d in timer.timers.items()}}
    times_row = pd.DataFrame(times_row).transpose()
    times_row["duration_total_seconds"] = times_row.sum(axis=1)
    rec = shp_records[plot_name]
    times_row["surface_m2"] = rec._area
    times_row["surface_ha"] = np.round((rec._area) / 10000, 2)
    times_row["duration_seconds_by_hectar"] = (
        times_row["duration_total_seconds"] / times_row["surface_ha"]
    )
    times_row.reset_index().rename(columns={"index": "name"}).to_csv(
        file_append_mode, index=False, header=file_append_mode.tell() == 0
    )


def get_parcel_info_and_predictions(tif, records):
    """From a prediction tif given by  its path and the records obtained from a shapefile,
    get the parcel metadata as well as the predictions : coverage and admissibility
    """
    mosaic = rasterio.open(tif).read()

    # Vb, Vmoy, Vh, Vmoy_hard
    band_means = np.nanmean(mosaic[:5], axis=(1, 2))

    # TODO: admissibility computed at merging
    admissibility = np.nanmean(np.nanmax([[mosaic[0]], [mosaic[0]]], axis=0))

    tif_name = get_filename_no_extension(tif).replace("prediction_raster_parcel_", "")
    rec = records[tif_name]

    metadata = {
        "NOM": tif_name,
        "SURFACE_m2": rec._area,
        "SURFACE_ha": np.round((rec._area) / 10000, 2),
        "SURF_ADM_ha": rec.SURF_ADM,
        "REPORTED_ADM": float(rec.ADM),
    }
    infered_values = {
        "pred_veg_b": band_means[0],
        "pred_veg_moy": band_means[1],
        "pred_veg_h": band_means[2],
        "adm_max_over_veg_b_and_veg_moy": admissibility,
        "pred_veg_moy_hard": band_means[3],
        "fifth_band_mean_ie_weights_or_admissibility": band_means[4],
    }
    metadata.update(infered_values)
    return metadata


def make_parcel_predictions_csv(parcel_shapefile_path, stats_path):
    sf = shapefile.Reader(parcel_shapefile_path)
    records = {rec.ID: rec for rec in sf.records()}
    predictions_tif = glob.glob(
        os.path.join(stats_path, "**/prediction_raster_parcel_*.tif"),
        recursive=True,
    )
    infos = []
    for tif_filename in predictions_tif:
        info = get_parcel_info_and_predictions(tif_filename, records)
        infos.append(info)

    # export to a csv
    df_inference = pd.DataFrame(infos)
    csv_path = os.path.join(stats_path, "PCC_inference_all_parcels.csv")
    df_inference.to_csv(csv_path, index=False)
    return df_inference, csv_path


def get_list_las_files_not_infered_yet(stats_path, las_parcelles_folder_path):
    """
    List paths of las parcel files which for which we do not have a global prediction raster yet.
    """
    las_filenames = get_files_of_type_in_folder(las_parcelles_folder_path, ".las")
    las_filenames = [
        l
        for l in las_filenames
        if os.path.join(
            stats_path,
            f"img/rasters/{get_filename_no_extension(l)}/prediction_raster_parcel_{get_filename_no_extension(l)}.tif",
        )
        not in glob.glob(
            stats_path + "/**/prediction_raster_parcel_*.tif", recursive=True
        )
    ]
    return las_filenames
