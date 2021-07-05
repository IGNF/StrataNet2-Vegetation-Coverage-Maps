import os
import sys
from utils.useful_functions import (
    get_filename_no_extension,
    get_files_of_type_in_folder,
)
import numpy as np
import pandas as pd
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import warnings
import random

random.seed(0)
from scipy.interpolate import SmoothBivariateSpline
import numpy_indexed as npi
from random import random, shuffle

np.random.seed(0)

warnings.simplefilter(action="ignore")


def load_all_las_from_folder(args):

    # We open las files and create a training dataset
    nparray_clouds_dict = {}  # dict to store numpy array with each plot separately
    xy_centers_dict = (
        {}
    )  # we keep track of plots means to reverse the normalisation in the future

    # We iterate through las files and transform them to np array
    las_filenames = get_files_of_type_in_folder(args.las_placettes_folder_path, ".las")

    # DEBUG
    if args.mode == "DEV":
        shuffle(las_filenames)
        las_filenames = las_filenames[: (5 * 5)]  # nb plot by fold

        # las_files = las_files[:10] + [
        #     l
        #     for l in las_files
        #     if any(n in l for n in ["OBS15", "F68", "2021_POINT_OBS2"])
        # ]

    all_points_nparray = np.empty((0, len(args.input_feats)))
    for las_filename in las_filenames:
        # Parse LAS files
        points_nparray, xy_center = load_and_clean_single_las(las_filename)
        points_nparray = transform_features_of_plot_cloud(
            points_nparray, xy_center, args
        )
        all_points_nparray = np.append(all_points_nparray, points_nparray, axis=0)
        plot_name = get_filename_no_extension(las_filename)
        nparray_clouds_dict[plot_name] = points_nparray
        xy_centers_dict[plot_name] = xy_center

    return all_points_nparray, nparray_clouds_dict, xy_centers_dict


def load_and_clean_single_las(las_filename):
    """Load a LAD file into a np.array, convert coordinates to meters, clean a few anomalies in plots."""
    # Parse LAS files
    las = File(las_filename, mode="r")
    x_las = las.X / 100  # we divide by 100 as all the values in las are in cm
    y_las = las.Y / 100
    z_las = las.Z / 100
    r = las.Red
    g = las.Green
    b = las.Blue
    nir = las.nir
    intensity = las.intensity
    return_num = las.return_num
    num_returns = las.num_returns
    points_nparray = np.asarray(
        [x_las, y_las, z_las, r, g, b, nir, intensity, return_num, num_returns]
    ).T

    # There is a file with 2 points 60m above others (maybe birds), we delete these points
    if las_filename.endswith("Releve_Lidar_F70.las"):
        points_nparray = points_nparray[points_nparray[:, 2] < 640]
    # We do the same for the intensity
    if las_filename.endswith("POINT_OBS8.las"):
        points_nparray = points_nparray[points_nparray[:, -2] < 32768]
    if las_filename.endswith("Releve_Lidar_F39.las"):
        points_nparray = points_nparray[points_nparray[:, -2] < 20000]

    # get the center of a rectangle bounding the points
    xy_centers = [
        (x_las.max() - x_las.min()) / 2.0,
        (y_las.max() - y_las.min()) / 2.0,
    ]
    return points_nparray, xy_centers


def transform_features_of_plot_cloud(points_nparray, xy_center, args):
    """From the loaded points_nparray, process features and add additional ones.
    This is different from [0;1] normalization which is performed in
    1) Add a feature:min-normalized using min-z of the plot
    2) Substract z_min at local level using KNN
    """
    # normalize "z"
    if args.z_normalization_method == "knn":
        points_nparray = normalize_z_with_minz_in_a_radius(
            points_nparray, args.znorm_radius_in_meters
        )
    elif args.z_normalization_method == "spline":
        points_nparray = normalize_z_with_smooth_spline(points_nparray, xy_center, args)
    else:
        sys.exit(f"Unknown normalization method {args.z_normalization_method}")
    # add "z_original"
    zmin_plot = np.min(points_nparray[:, 2])
    points_nparray = np.append(
        points_nparray, points_nparray[:, 2:3] - zmin_plot, axis=1
    )

    return points_nparray


def normalize_z_with_minz_in_a_radius(cloud, znorm_radius_in_meters):
    # # We directly substract z_min at local level
    xyz = cloud[:, :3]
    knn = NearestNeighbors(500, algorithm="kd_tree").fit(xyz[:, :2])
    _, neigh = knn.radius_neighbors(xyz[:, :2], znorm_radius_in_meters)
    z = xyz[:, 2]
    zmin_neigh = []
    for n in range(
        len(z)
    ):  # challenging to make it work without a loop as neigh has different length for each point
        zmin_neigh.append(np.min(z[neigh[n]]))
    cloud[:, 2] = cloud[:, 2] - zmin_neigh
    return cloud


def center_plot(cloud, xy_center):
    """Center the cloud to 0, also return the initial xymin to decenter the cloud"""
    cloud[:, :2] = cloud[:, :2] - xy_center
    return cloud


def decenter_plot(cloud, xy_center):
    cloud[:, :2] = cloud[:, :2] + xy_center
    return cloud


def xy_to_polar_coordinates(xy):
    r = np.sqrt((1.0 * xy * xy).sum(axis=1))
    teta = np.arctan2(
        xy[:, 1], xy[:, 0]
    )  # -pi, pi around (0,0) to (1,0). y and x are args in this order.
    rteta = np.stack([r, teta], axis=1)
    return rteta


def polar_coordinates_to_xy(rteta):
    x = rteta[:, 0] * np.cos(rteta[:, 1])
    y = rteta[:, 0] * np.sin(rteta[:, 1])
    xy = np.stack([x, y], axis=1)
    return xy


def create_buffer_points(cloud, ring_thickness_meters, diam_meters):
    """cloud is centered with xy as first coordinates in meters."""
    candidates_polar = cloud.copy()
    candidates_polar[:, :2] = xy_to_polar_coordinates(candidates_polar[:, :2])
    candidates_polar = candidates_polar[
        candidates_polar[:, 0] > (diam_meters // 2 - ring_thickness_meters)
    ]  # points in external ring
    candidates_polar[:, 0] = candidates_polar[:, 0] + 2 * (
        abs(diam_meters // 2 - candidates_polar[:, 0])
    )  # use border of plot as a mirror
    candidates_polar[:, :2] = polar_coordinates_to_xy(candidates_polar[:, :2])
    return candidates_polar


def normalize_z_with_smooth_spline(cloud, xy_center, args):
    """From a regular grid, find lowest point in each cell/pixel and use them to approximate
    the DTM with a spline. Then, normalize z by flattening the ground using the DTM.
    """
    norm_cloud = cloud.copy()
    cloud_z_min = norm_cloud[:, 2].min()

    # center in order to use polar coordinate
    norm_cloud = center_plot(norm_cloud, xy_center)

    # create buffer
    buffer_points = create_buffer_points(
        norm_cloud, args.ring_thickness_meters, args.diam_meters
    )
    extended_cloud = np.concatenate([norm_cloud, buffer_points])

    # fit
    xy_quantified = (
        extended_cloud[:, :2] // args.spline_pix_size + 0.5
    ) * args.spline_pix_size  # quantify (and center) coordinates
    _, z_argmin = npi.group_by(xy_quantified).argmin(extended_cloud[:, 2])
    extended_cloud = extended_cloud[z_argmin]
    max_n_iter = 10
    for i in range(max_n_iter):
        sbs = SmoothBivariateSpline(
            extended_cloud[:, 0],
            extended_cloud[:, 1],
            extended_cloud[:, 2],
            kx=3,
            ky=3,
            s=None,
        )

        # predict on normcloud
        norm_cloud_iter = norm_cloud.copy()
        sbs_pred = sbs(norm_cloud_iter[:, 0], norm_cloud_iter[:, 1], grid=False)
        sbs_pred[sbs_pred < cloud_z_min] = cloud_z_min
        norm_cloud_iter[:, 2] = norm_cloud_iter[:, 2] - sbs_pred

        # Stop iteration if no points below 0 or if most are close to 0
        mask_below_spline = norm_cloud_iter[:, 2] < 0
        if mask_below_spline.sum() == 0:
            break
        else:
            q = np.quantile(norm_cloud_iter[mask_below_spline, 2], 0.1)  # lowest 10%
            if q > -0.05:
                break
        # find points falling below spline surface and add them to points used to fit the spline
        points_below_spline = norm_cloud[mask_below_spline]
        extended_cloud = np.concatenate([extended_cloud, points_below_spline])

    # Normalize original cloud with finalized Spline
    sbs_pred = sbs(norm_cloud[:, 0], norm_cloud[:, 1], grid=False)
    sbs_pred[sbs_pred < cloud_z_min] = cloud_z_min
    norm_cloud[:, 2] = norm_cloud[:, 2] - sbs_pred

    # Set z=0 for points below the spline
    mask_below_spline = norm_cloud[:, 2] < 0
    norm_cloud[mask_below_spline, 2] = 0.0

    # get back to initial coordinate system
    norm_cloud = decenter_plot(norm_cloud, xy_center)
    return norm_cloud


def open_metadata_dataframe(args, pl_id_to_keep):
    """This opens the ground truth file. It completes if necessary admissibility value using ASP method.
    Values are kept as % as they are transformed during data loading into ratios."""

    df_gt = pd.read_csv(
        args.gt_file_path,
        sep=",",
        header=0,
    )  # we open GT file
    # Here, adapt columns names
    df_gt = df_gt.rename(args.coln_mapper_dict, axis=1)

    # Keep metadata for placettes we are considering
    df_gt = df_gt[df_gt["Name"].isin(pl_id_to_keep)]

    # Correct Soil value to have
    df_gt["COUV_SOL"] = 100 - df_gt["COUV_BASSE"]

    # his is ADM based on ASP definition - NOT USED at the moment
    if "ADM" not in df_gt:
        df_gt["ADM_BASSE"] = df_gt["COUV_BASSE"] - df_gt["NON_ACC_1"]
        df_gt["ADM_INTER"] = df_gt["COUV_HAUTE"] - df_gt["NON_ACC_2"]
        df_gt["ADM"] = df_gt[["ADM_BASSE", "ADM_INTER"]].max(axis=1)

        del df_gt["ADM_BASSE"]
        del df_gt["ADM_INTER"]

    # check that we have all columns we need
    assert all(
        coln in df_gt
        for coln in [
            "Name",
            "COUV_BASSE",
            "COUV_SOL",
            "COUV_INTER",
            "COUV_HAUTE",
            "ADM",
        ]
    )

    placettes_names = df_gt[
        "Name"
    ].to_numpy()  # We extract the names of the plots to create train and test list

    return df_gt, placettes_names
