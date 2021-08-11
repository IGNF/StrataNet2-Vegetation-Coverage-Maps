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
import rasterio
from rasterio.merge import merge
import rasterio.features
from rasterio.transform import Affine
from shapely import geometry
import shapefile
from shapely.geometry.point import Point
import torch.nn as nn
import shapely

from utils.load_data import load_las_file, pre_transform, get_plot_center
from utils.utils import get_filename_no_extension, create_dir
from data_loader.loader import center_cloud

sns.set()
np.random.seed(42)

import logging

logger = logging.getLogger(__name__)


def get_shape(shp, query_object_name):
    """Get shapely shape in shapefile by its ID."""
    oid = next(rec.oid for rec in shp.records() if rec.ID == query_object_name)
    parcel_shape = shapely.geometry.shape(shp.shape(oid).__geo_interface__)
    return parcel_shape


def get_xy_range(cloud):
    x_las, y_las = cloud[:, 0], cloud[:, 1]
    x_min, y_min = cloud[:2].min(1)
    x_max, y_max = cloud[:2].max(1)
    return x_min, x_max, y_min, y_max


def extract_cloud(plot_center, parcel_cloud, radius=10):
    """From a (2, N) np.array with x, y as first features, extract points within radius
    from the plot_center = (x_center, y_center)"""
    xy = parcel_cloud[:2]  # (N, 2)
    points_idx = ((xy - np.expand_dims(plot_center, 1)) ** 2).sum(axis=0) <= (
        radius * radius
    )
    cloud = parcel_cloud[:, points_idx]  # (N, f)

    return cloud


# TODO: entwine parcel_cloud for faster search
def extract_cloud_data(query, parcel_cloud, args):
    """Extract cloud points from around plot_center and prepare its data."""
    plot_idx = query["plot_idx"]
    plot_center = query["plot_center"]

    cloud = extract_cloud(plot_center, parcel_cloud, radius=args.diam_meters // 2)
    N_points_in_cloud = cloud.shape[1]

    MIN_N_POINTS_FOR_INFERENCE = 50
    if N_points_in_cloud < MIN_N_POINTS_FOR_INFERENCE:
        return None

    cloud = pre_transform(cloud, args)
    plot_name = define_a_plot_name(plot_idx)
    plot_id = define_plot_id(plot_name, plot_center)
    cloud_data = {
        "cloud": cloud,
        "plot_center": plot_center,
        "plot_id": plot_id,
        "index": plot_idx,
        "N_points_in_cloud": N_points_in_cloud,
    }
    return cloud_data


def define_a_plot_name(plot_idx):
    """Define unique plot name"""
    return f"PP" + str(plot_idx).zfill(8)


def define_plot_id(plot_name, plot_center):
    """Define plot id, keeping track of coordinates."""
    plot_id = f"{plot_name}_X{int(plot_center[0])}_Y{int(plot_center[1])}"
    return plot_id


def divide_parcel_las_and_get_disk_centers(
    args, las_filename, parcel_shape, division_fig_save_path=""
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
        cloud: a nparray of full cloud coordinates
    Note: outputs are not normalized
    """

    parcel_cloud = load_las_file(las_filename)
    x_min, x_max, y_min, y_max = get_xy_range(parcel_cloud)

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
    plot_centers = [[start_x, start_y]]

    for i_dx in range(x_range_of_parcel_in_movements):
        current_x = start_x + i_dx * movement_in_meters  # move along x axis
        for i_dy in range(y_range_of_parcel_in_movements):
            current_y = start_y + i_dy * movement_in_meters  # move along y axis
            new_plot_center = [current_x, current_y]
            plot_centers.append(new_plot_center)

    # Ignore plot center if not in shape of shapefile
    plot_centers = [
        np.array(plot_center, dtype=np.float32)
        for plot_center in plot_centers
        if parcel_shape.buffer(args.diam_meters // 2).contains(
            Point(plot_center[0], plot_center[1])
        )
    ]

    # visualization
    if division_fig_save_path:
        # we need to normalize coordinates points for easier visualization
        parcel_id = get_filename_no_extension(las_filename)
        save_image_of_parcel_division_into_plots(
            parcel_cloud,
            parcel_id,
            plot_centers,
            division_fig_save_path,
            args,
        )

    return plot_centers, parcel_cloud


def save_image_of_parcel_division_into_plots(
    parcel_cloud,
    parcel_id,
    plot_centers,
    division_fig_save_path,
    args,
):
    """
    Visualize and save to PNG file the division of a large parcel into many disk subplots.
    """

    parcel_center = get_plot_center(parcel_cloud)
    parcel_cloud = center_cloud(parcel_cloud, parcel_center)

    # xy to dataframe for visualization
    coordinates = parcel_cloud[:2]
    coordinates = pd.DataFrame(data=coordinates.transpose())
    coordinates.columns = ["x_n", "y_n"]

    SAMPLING_FOR_KDE_VISUALIZATION = (
        10000  # fixed size which could lead to poor kde in large parcels.
    )
    if len(coordinates) > SAMPLING_FOR_KDE_VISUALIZATION:
        coordinates = coordinates.sample(
            n=SAMPLING_FOR_KDE_VISUALIZATION, replace=False
        )

    # plot centers to dataframe for visualization
    plot_centers = np.stack(plot_centers).transpose()
    plot_centers = center_cloud(plot_centers, parcel_center)
    plot_centers = pd.DataFrame(data=plot_centers.transpose())
    plot_centers.columns = ["x_n", "y_n"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"aspect": "equal"})
    ax.grid(False)
    ax.set_aspect("equal")  # Not working right now
    x_min_c, x_max_c, y_min_c, y_max_c = get_xy_range(parcel_cloud)
    plt.xlim(x_min_c - 5, x_max_c + 5)
    plt.ylim(y_min_c - 5, y_max_c + 5)
    plt.ylabel("y_n", rotation=0)
    plt.title(
        f"Parcel {parcel_id} \nsplit in N={len(plot_centers)} plots (r={args.diam_pix//2})"
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
    )

    # plot disks and squares
    for _, (x, y) in plot_centers.iterrows():
        a_circle = plt.Circle(
            (x, y), 10, fill=True, alpha=0.1, edgecolor="white", linewidth=1
        )
        ax.add_patch(a_circle)
        a_circle = plt.Circle((x, y), 10, fill=False, edgecolor="white", linewidth=0.3)
        ax.add_patch(a_circle)

    sns.scatterplot(data=plot_centers, x="x_n", y="y_n", s=5)
    dir_path = os.path.dirname(division_fig_save_path)
    create_dir(dir_path)
    plt.savefig(division_fig_save_path, dpi=200)
    plt.clf()
    plt.close("all")
