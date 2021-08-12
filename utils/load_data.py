import math
import os
import sys

import numpy as np
import pandas as pd
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import warnings
import pickle

import numpy_indexed as npi
from random import random, shuffle, sample
import random

random.seed(0)
np.random.seed(0)
from utils.utils import (
    get_filename_no_extension,
    get_files_of_type_in_folder,
)
import utils.geo3dfeatures as geo3d

warnings.simplefilter(action="ignore")


def load_ground_truths_dataframe(args):
    """This opens the ground truth file. It completes if necessary admissibility value using ASP method.
    Values are kept as % as they are transformed during data loading into ratios."""

    df_gt = pd.read_csv(
        args.gt_file_path,
        sep=",",
        header=0,
    )
    df_gt = df_gt.rename({"nom": "Name"}, axis=1)
    df_gt["COUV_SOL"] = 100 - df_gt["COUV_BASSE"]

    assert all(
        coln in df_gt
        for coln in [
            "Name",
            "COUV_BASSE",
            "COUV_SOL",
            "COUV_INTER",
            "COUV_HAUTE",
        ]
    )

    return df_gt


def prepare_and_save_plots_dataset(args):
    """Create a pickled dataset of plots from a folder of plot las files, with structure:
    {plot_id :
        {
            plot_id: str,
            cloud: np.array of shape [N_points, N_features] with x, y, z as first features
            plot_center: [x_center, y_center] list
            N_points_in_cloud: int,
            coverages:  np.array in [0; 1] range, of shape [, 4]
    }
    Plots are also indexed to follow order in ground truth dataframe, in order to have reproductible cross-validation.
    """
    las_filenames = get_files_of_type_in_folder(args.las_plots_folder_path, ".las")
    if args.mode == "DEV":
        las_filenames = sample_filenames_for_dev_crossvalidation(las_filenames, args)

    ground_truths = load_ground_truths_dataframe(args)
    plot_id_to_keep = [
        get_filename_no_extension(filename) for filename in las_filenames
    ]
    ground_truths = ground_truths[ground_truths["Name"].isin(plot_id_to_keep)]
    plot_names = ground_truths.Name.values

    dataset = {}
    for index, plot_name in enumerate(plot_names):
        filename = get_filename_from_plot_name(las_filenames, plot_name)
        plot_id, cloud_data = get_cloud_data(filename, args, ground_truths)
        cloud_data["index"] = index
        dataset.update({plot_id: cloud_data})

    with open(args.plots_pickled_dataset_path, "wb") as pfile:
        pickle.dump(dataset, pfile)

    return dataset


def get_filename_from_plot_name(las_filenames, plot_name):
    """Return filename from a lits based on plot_name (i.e. the basename in a path/to/plot_name.las fashion)"""
    return next(
        filename
        for filename in las_filenames
        if os.path.basename(filename.lower()) == (plot_name.lower() + ".las")
    )


def load_pseudo_labelled_datasets(args):
    """Iteratively load n prepared datasets into one large dataset, for pre-training."""
    input_folder = os.path.join(
        args.las_parcels_folder_path,
        os.path.join("pseudo_labelling/", args.inference_model_id),
    )
    dataset_paths = get_files_of_type_in_folder(input_folder, ".pkl")
    full_dataset = {}
    for dataset_path in dataset_paths:
        with open(dataset_path, "rb") as pfile:
            dataset = pickle.load(pfile)
        full_dataset.update(dataset)
        if args.mode == "DEV":
            N_IN_SUBSET = 30
            full_dataset = dict(sample(full_dataset.items(), N_IN_SUBSET))
            break
    return full_dataset


def get_cloud_data(filename, args, ground_truths):
    """Get cloud data from a single plot las file."""
    cloud = load_las_file(filename)
    cloud = clean(cloud, filename, args)
    cloud = pre_transform(cloud, args)

    plot_id = get_filename_no_extension(filename)
    plot_center = get_plot_center(cloud)
    N_points_in_cloud = cloud.shape[1]
    coverages = get_plot_ground_truth_coverages(ground_truths, plot_id)

    cloud_data = {
        "cloud": cloud,
        "coverages": coverages,
        "plot_center": plot_center,
        "plot_id": plot_id,
        "N_points_in_cloud": N_points_in_cloud,
    }
    return plot_id, cloud_data


def load_pickled_dataset(args):
    """Load a pickled dataset of plots (or pseudoplots) as created in prepare_and_save_plots_dataset."""
    with open(args.plots_pickled_dataset_path, "rb") as pfile:
        return pickle.load(pfile)


def load_las_file(filename):
    """Load a LAD file into a np.array, and perform a few conversions and cleaning:
    - convert coordinates to meters
    - clean a few anomalies in plots.
    Output shape: [n_features, n_points]
    """
    # Parse LAS files
    las = File(filename, mode="r")
    CM_IN_METER = 100
    x_las = las.X / CM_IN_METER
    y_las = las.Y / CM_IN_METER
    z_las = las.Z / CM_IN_METER
    r = las.Red
    g = las.Green
    b = las.Blue
    nir = las.nir
    intensity = las.intensity
    return_num = las.return_num
    num_returns = las.num_returns

    cloud = np.asarray(
        [
            x_las,
            y_las,
            z_las,
            r,
            g,
            b,
            nir,
            intensity,
            return_num,
            num_returns,
        ],
        dtype=np.float32,
    )

    return cloud


def clean(cloud, las_filename, args):
    """Remove a few points with unrealistic values."""

    # There is a file with 2 points 60m above others (maybe birds), we delete these points
    feature_idx = args.input_feats.index("z_flat")
    if las_filename.endswith("Releve_Lidar_F70.las"):
        cloud = cloud[:, cloud[feature_idx] < 640]

    # We do the same for the intensity
    feature_idx = args.input_feats.index("intensity")
    if las_filename.endswith("POINT_OBS8.las"):
        cloud = cloud[:, cloud[feature_idx] < 32768]
    if las_filename.endswith("Releve_Lidar_F39.las"):
        cloud = cloud[:, cloud[feature_idx] < 20000]

    return cloud


def get_plot_ground_truth_coverages(ground_truths, plot_id):
    """Extract ground truths coverages for specified plot."""
    coverages = (
        ground_truths[ground_truths["Name"] == plot_id][
            ["COUV_BASSE", "COUV_SOL", "COUV_INTER", "COUV_HAUTE"]
        ].values
        / 100
    )
    return coverages.astype(float).squeeze()


def get_plot_center(cloud):
    """Get the center of a rectangle bounding the points along x and y."""
    x_las = cloud[0]
    y_las = cloud[1]
    plot_center = [
        (x_las.max() + x_las.min()) / 2.0,
        (y_las.max() + y_las.min()) / 2.0,
    ]
    plot_center = np.array(plot_center, dtype=np.float32)
    return plot_center


def pre_transform(cloud, args):
    """Initial prepare point cloud (before any rescaling/data augmentation).
    This is done only once per plot.
    """

    cloud = normalize_z_with_minz_in_a_radius(cloud, args.znorm_radius_in_meters)
    cloud = append_local_features(cloud, args)
    cloud = cloud.astype(np.float32)

    return cloud


def normalize_z_with_minz_in_a_radius(cloud, znorm_radius_in_meters):
    # # We directly substract z_min at local level
    xyz = cloud[:3, :].transpose()  # [n_points, 3]
    knn = NearestNeighbors(500, algorithm="kd_tree").fit(xyz[:, :2])
    _, neigh = knn.radius_neighbors(xyz[:, :2], znorm_radius_in_meters)
    z = xyz[:, 2]
    zmin_neigh = []
    for n in range(
        len(z)
    ):  # challenging to make it work without a loop as neigh has different length for each point
        zmin_neigh.append(np.min(z[neigh[n]]))
    cloud[2] = cloud[2] - zmin_neigh
    return cloud


def append_local_features(cloud, args):
    """Append some local features to the cloud. Thier order should match the one in config."""
    # TODO: make robust to feat order in config

    kde_tree = geo3d.compute_tree(cloud[:3].T, 100)
    extended_points = [
        append_local_features_to_point(point, kde_tree) for point in cloud.transpose()
    ]
    cloud = np.stack(extended_points).transpose()
    return cloud


def append_local_features_to_point(point, kde_tree):
    """Return a vector with initial point features + additional local features."""
    dist_to_neighbors, neighbors_indexes = geo3d.request_tree(
        point[:3], kde_tree, radius=0.5
    )
    neighbors = kde_tree.data[neighbors_indexes]

    num_neighbors = neighbors.shape[0] - 1

    # TODO: add features that reflect 3D scattering to differentiate trees from medium veg.
    additional_features = []

    # rad_3D = geo3d.radius_3D(dist_to_neighbors)
    # additional_features.append(geo3d.density_3D(0.5, len(neighbors)))
    # z = neighbors[:, 2]
    # additional_features.append(geo3d.std_deviation(z))
    # additional_features.append(geo3d.val_range(z))
    alpha = np.nan
    beta = np.nan

    if len(neighbors) > 2:
        pca = geo3d.fit_pca(neighbors)
        eigenvalues_3D = pca.singular_values_ ** 2
        norm_eigenvalues_3D = geo3d.sum_normalize(eigenvalues_3D)
        alpha, beta = geo3d.triangle_variance_space(norm_eigenvalues_3D)
    if np.isnan(alpha):
        alpha = 1 / 3
        beta = 1 / 3

    additional_features.append(alpha)
    additional_features.append(beta)

    return np.concatenate([point, np.array(additional_features)])


def sample_filenames_for_dev_crossvalidation(filename, args, n_by_fold=6):
    """Select a few plots to perform a faster crossvalidation."""
    selection = [
        l
        for l in filename
        if any(n in l for n in args.plot_name_to_visualize_during_training)
    ]
    shuffle(filename)
    filename = selection + filename[: (args.folds * n_by_fold - len(selection))]
    return filename
