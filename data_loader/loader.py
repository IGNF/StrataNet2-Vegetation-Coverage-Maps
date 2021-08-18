import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import torchnet as tnt
import functools
from collections import namedtuple
import copy


def get_train_val_datasets(dataset, args, train_idx=None, val_idx=None):
    """Split the dataset (list of cloud_data dicts) between train and test sets, with appropriate loading functions."""
    train_set = get_train_dataset(dataset, args, plot_idx=train_idx)
    val_set = get_val_dataset(dataset, args, plot_idx=val_idx)
    return train_set, val_set


def get_train_dataset(dataset, args, plot_idx=None):
    """Get the train dataset, with appropriate loading functions."""
    # plot_ids = sorted(list(dataset.keys()))
    plot_ids = get_index_sorted_plot_ids(dataset)
    if plot_idx is not None:
        train_list = plot_ids[plot_idx]
    else:
        train_list = plot_ids
    train_set = tnt.dataset.ListDataset(
        train_list,
        functools.partial(load_cloud, dataset=dataset, args=args, train=True),
    )
    return train_set


def get_val_dataset(dataset, args, plot_idx=None):
    """Get the val dataset, with appropriate loading functions."""
    plot_ids = get_index_sorted_plot_ids(dataset)
    if plot_idx is not None:
        test_list = plot_ids[plot_idx]
    else:
        test_list = plot_ids
    val_set = tnt.dataset.ListDataset(
        test_list,
        functools.partial(load_cloud, dataset=dataset, args=args, train=False),
    )
    return val_set


def get_index_sorted_plot_ids(dataset):
    """From a dataset of cloud_data items, get list of plot_ids sorted by index."""
    indexed_plot_id = [
        {"plot_id": cloud_data["plot_id"], "index": cloud_data["index"]}
        for cloud_data in dataset.values()
    ]
    indexed_plot_id = sorted(indexed_plot_id, key=lambda item: item["index"])
    plot_ids = np.array([item["plot_id"] for item in indexed_plot_id])
    return plot_ids


def _load_cloud_data(pseudoplot_ID, dataset, args):
    """
    From a dict of dict of cloud data, get cloud data.
    The cloud is in model-friendly [N_features, N_points] format.
    """
    cloud_data = dataset[pseudoplot_ID]
    cloud_data = copy.deepcopy(cloud_data)

    try:
        coverages = cloud_data["coverages"]
    except KeyError:
        cloud_data["coverages"] = np.empty(0)

    return cloud_data


def load_cloud(pseudoplot_ID, dataset, args, train=False):
    """From a list of dict of plots infos, get model-ready data with metainfo for eval."""
    cloud_data = _load_cloud_data(pseudoplot_ID, dataset, args)

    cloud_data["cloud"] = center_cloud(cloud_data["cloud"], cloud_data["plot_center"])
    cloud_data["cloud"] = add_fake_empty_ground_points(args, cloud_data["cloud"])
    cloud_data["xyz"] = cloud_data["cloud"][:3].copy()

    if train:
        cloud_data = augment(args, cloud_data)
    cloud_data["cloud"] = rescale_cloud(cloud_data["cloud"], args)

    cloud_data = sample_cloud_data(cloud_data, args.subsample_size)

    return cloud_data


def add_fake_empty_ground_points(args, cloud):
    """Add a fake point with features filled with 0 except for position, for evey pixel of the final raster"""
    xx, yy = get_x_y_meshgrid(args.diam_meters)
    x = xx + 0 * yy
    y = yy + 0 * xx
    x = x.flatten()
    y = y.flatten()
    r = np.sqrt(x ** 2 + y ** 2)
    fake_points = []
    for x, y, r in zip(x, y, r):
        if r < args.diam_meters // 2:
            fake_points.append([x, y, 0.0] + (args.n_input_feats - 3) * [0.0])
    cloud = np.concatenate(
        [cloud, np.array(fake_points, dtype=np.float32).transpose()], axis=1
    )
    return cloud


def get_x_y_meshgrid(width):
    """Create meshgrids of x and y values centered around 0 and with width size."""
    x = np.arange(-width // 2, width // 2, 1) + 0.5
    y = np.arange(-width // 2, width // 2, 1) + 0.5
    xx, yy = np.meshgrid(x, y, sparse=True)
    return xx, yy


def get_normalized_x_y_meshgrid(width):
    """
    Create normalized meshgrids of x and y values centered around 0 and with width 1 (from -0.5 to 0.5).
    Width defines the number of pixels along an axis.
    """
    xx, yy = get_x_y_meshgrid(width)
    xx = xx / width
    yy = yy / width
    return xx, yy


def center_cloud(cloud, plot_center):
    """Center cloud data along x and y dimensions."""
    x_center, y_center = plot_center
    cloud[0] = cloud[0] - x_center
    cloud[1] = cloud[1] - y_center
    return cloud


def rescale_cloud(cloud, args):
    """
    Normalize data by reducing scale, to feed te neural net.
    :param cloud: np.array of shape (9, N)
    """
    cloud[0] = cloud[0] / 10
    cloud[1] = cloud[1] / 10
    cloud[2] = cloud[2] / args.z_max

    input_feats = args.input_feats
    colors_max = 65536
    for feature in ["red", "green", "blue", "near_infrared"]:
        idx = input_feats.index(feature)
        cloud[idx] = cloud[idx] / colors_max

    intensity_max = 32768
    idx = input_feats.index("intensity")
    cloud[idx] = cloud[idx] / intensity_max

    for feature in ["return_num", "num_returns"]:
        idx = input_feats.index(feature)
        cloud[idx] = (cloud[idx] - 1) / (7 - 1)

    return cloud


def augment(args, cloud_data):
    """Augmentat data for generalization"""

    cloud = cloud_data["cloud"]
    xyz = cloud_data["xyz"]

    # Random rotation around z and flipping along x and/or y axis
    angle, flip_x, flip_y = get_xyz_augmentation_params()
    cloud = rotate_around_z(cloud, angle)
    xyz = rotate_around_z(xyz, angle)
    if flip_x:
        cloud[0] = -cloud[0]
        xyz[0] = -xyz[0]
    if flip_y:
        cloud[1] = -cloud[1]
        xyz[1] = -xyz[1]

    # random gaussian noise for x and y
    xy_norm_factor = 10
    sigma = 0.01 * xy_norm_factor
    clip = 0.03 * xy_norm_factor
    cloud[:2] = (
        cloud[:2]
        + np.clip(
            sigma * np.random.randn(cloud[:2].shape[0], cloud[:2].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )

    input_feats = args.input_feats

    # random gaussian noise for RGB+NIR
    colors_max = 65536
    sigm = 0.01 * colors_max
    clip = 0.03 * colors_max
    for feature in ["red", "green", "blue", "near_infrared"]:
        idx = input_feats.index(feature)
        cloud[idx] = (
            cloud[idx]
            + np.clip(
                sigma * np.random.randn(cloud[idx].shape[0]),
                a_min=-clip,
                a_max=clip,
            ).astype(np.float32)
        )

    # intensity_max = 32768
    # idx = input_feats.index("intensity")

    cloud_data["cloud"] = cloud
    cloud_data["xyz"] = xyz

    return cloud_data


def rotate_around_z(cloud, angle):
    """Rotate cloud with an angle, assuming x and y are first two rows. This modifies cloud."""
    c, s = np.cos(angle), np.sin(angle)
    M = np.array(((c, -s), (s, c)))  # rotation matrix around axis z with angle "angle"
    cloud[:2] = np.dot(cloud[:2].T, M).T  # perform the rotation efficiently
    return cloud


def get_xyz_augmentation_params():
    """Defines the xyz augmentations (rotation, x and y flip) to perform on both (scaled) data and (no rescaled) position."""
    flip_x = np.random.random() > 0.5
    flip_y = np.random.random() > 0.5
    angle = np.radians(np.random.choice(360, 1)[0])
    return angle, flip_x, flip_y


def sample_cloud(cloud, subsample_size):
    """Select subsample_size points out of cloud, with replacement only if necessary."""
    n_points = cloud.shape[1]
    if n_points > subsample_size:
        sampled_points_idx = np.random.choice(n_points, subsample_size, replace=False)
    else:
        sampled_points_idx = np.concatenate(
            [
                np.arange(n_points),
                np.random.choice(n_points, subsample_size - n_points, replace=True),
            ]
        )
    cloud = cloud[:, sampled_points_idx].copy()
    return cloud, sampled_points_idx


def sample_cloud_data(cloud_data, subsample_size):
    """Perform the same subsampling, on cloud and xyz."""
    cloud_data["cloud"], sampled_points_idx = sample_cloud(
        cloud_data["cloud"], subsample_size
    )
    cloud_data["xyz"] = cloud_data["xyz"][:, sampled_points_idx]
    return cloud_data
