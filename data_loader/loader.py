import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import torchnet as tnt
import functools
from collections import namedtuple
import copy
from utils.load_data import remove_color_from_occluded_points


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
    cloud_data["cloud"] = rescale_cloud(cloud_data["cloud"], args)
    if train:
        cloud_data["cloud"] = augment(cloud_data["cloud"], args)
    cloud_data["cloud"] = sample_cloud(cloud_data["cloud"], args.subsample_size)
    cloud_data["xyz"] = cloud_data["cloud"][:3]

    return cloud_data


def center_cloud(cloud, plot_center):
    """Center cloud data along x and y dimensions."""
    x_center, y_center = plot_center
    cloud = cloud.copy()
    cloud[0] = cloud[0] - x_center  # x
    cloud[1] = cloud[1] - y_center  # y
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


def augment(cloud, args):
    """augmentation function
    Does random rotation around z axis and adds Gaussian noise to all the features, except z and return number
    """
    # random rotation around the Z axis
    # angle = random angle 0..2pi
    angle = np.radians(np.random.choice(360, 1)[0])
    c, s = np.cos(angle), np.sin(angle)
    M = np.array(((c, -s), (s, c)))  # rotation matrix around axis z with angle "angle"
    cloud[:2] = np.dot(cloud[:2].T, M).T  # perform the rotation efficiently

    # # Random flipping along x and/or y axis
    if np.random.random() > 0.5:
        cloud[0] = -cloud[0]
    if np.random.random() > 0.5:
        cloud[1] = -cloud[1]

    # random gaussian noise everywhere except z and return number
    sigma, clip = 0.01, 0.03
    cloud[:2] = (
        cloud[:2]
        + np.clip(
            sigma * np.random.randn(cloud[:2].shape[0], cloud[:2].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )
    cloud[3:8] = (
        cloud[3:8]
        + np.clip(
            sigma * np.random.randn(cloud[3:8].shape[0], cloud[3:8].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )
    cloud[3:8] = cloud[3:8] - cloud[3:8].min(1, keepdims=True)

    # Reset colors of augmented points to 0
    cloud = remove_color_from_occluded_points(cloud, args.input_feats)

    return cloud


def sample_cloud(cloud, subsample_size):
    """Select subsample_size points out of cloud, with replacement only if necessary."""
    n_points = cloud.shape[1]
    if n_points > subsample_size:
        selected_points = np.random.choice(n_points, subsample_size, replace=False)
    else:
        selected_points = np.concatenate(
            [
                np.arange(n_points),
                np.random.choice(n_points, subsample_size - n_points, replace=True),
            ]
        )
    cloud = cloud[:, selected_points].copy()
    return cloud
