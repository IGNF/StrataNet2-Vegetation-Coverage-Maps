import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# TODO: reflect this ordser of OP in inference.
def cloud_loader(plot_id, dataset, df_gt, train, args):
    """
    For DataLoader during training of model.
    load a plot and returns points features (normalized xyz + features) and
    ground truth
    INPUT:
    tile_name = string, name of the tile
    train = int, train = 1 iff in the train set
    OUTPUT
    cloud_data, [n x 4] float Tensor containing points coordinates and intensity
    labels, [n] long int Tensor, containing the points semantic labels
    """
    cloud_data = np.array(dataset[plot_id], dtype=np.float32).transpose()
    gt = (
        df_gt[df_gt["Name"] == plot_id][
            ["COUV_BASSE", "COUV_SOL", "COUV_INTER", "COUV_HAUTE", "ADM"]
        ].values
        / 100
    )

    cloud_data = rescale_cloud_data(cloud_data, None, args)

    if train:
        cloud_data = augment(cloud_data)

    cloud_data = sample_cloud(cloud_data, args.subsample_size)

    cloud_data = torch.from_numpy(cloud_data)
    gt = torch.from_numpy(gt).float().squeeze()
    return cloud_data, gt


def cloud_loader_for_prediction(pseudoplot_ID, dataset, args):
    """
    From a list of dict containing pseudoplots from parcels,
    """
    pickled_data = dataset[pseudoplot_ID]
    cloud_data = pickled_data["plot_points_arr"]

    plot_center = pickled_data["plot_center"]
    plot_center = np.array(plot_center, dtype=np.float32)

    cloud_data = cloud_data.astype(np.float32).transpose()
    cloud_data = rescale_cloud_data(cloud_data, plot_center, args)
    cloud_data = sample_cloud(cloud_data, args.subsample_size)

    cloud_data = torch.from_numpy(cloud_data)
    return cloud_data, plot_center, pseudoplot_ID


def rescale_cloud_data(cloud_data, cloud_center, args):
    """
    Normalize data by reducing scale, to feed te neural net.
    :param cloud_data: np.array of shape (9, N)
    """
    # normalizing data
    # Z data was already partially normalized during loading
    input_feats = args.input_feats

    if cloud_center is None:
        # Training scenario : the actual center is not important
        x_center, y_center = (
            np.min(cloud_data[0:2], axis=1) + np.max(cloud_data[0:2], axis=1)
        ) / 2
    else:
        # Inference scenario : the actual center is important for reprojection to rasters
        x_center, y_center = cloud_center

    cloud_data[0] = (cloud_data[0] - x_center) / 10  # x
    cloud_data[1] = (cloud_data[1] - y_center) / 10  # y
    cloud_data[2] = cloud_data[2] / args.z_max  # z (flattened by normalization)

    colors_max = 65536
    for feature in ["red", "green", "blue", "near_infrared"]:
        idx = input_feats.index(feature)
        cloud_data[idx] = cloud_data[idx] / colors_max

    intensity_max = 32768
    idx = input_feats.index("intensity")
    cloud_data[idx] = cloud_data[idx] / intensity_max
    for feature in ["return_num", "num_returns"]:
        idx = input_feats.index(feature)
        cloud_data[idx] = (cloud_data[idx] - 1) / (7 - 1)

    # idx = input_feats.index("scan_angle")
    # cloud_data[idx] = (
    #     cloud_data[idx]
    # ) / 30.0  # angles in degrees    idx = input_feats.index("scan_angle")
    # cloud_data[idx] = (cloud_data[idx]) / 30.0  # angles in degrees

    return cloud_data


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


def augment(cloud_data):
    """augmentation function
    Does random rotation around z axis and adds Gaussian noise to all the features, except z and return number
    """
    # random rotation around the Z axis
    # angle = random angle 0..2pi
    angle = np.radians(np.random.choice(360, 1)[0])
    c, s = np.cos(angle), np.sin(angle)
    M = np.array(((c, -s), (s, c)))  # rotation matrix around axis z with angle "angle"
    cloud_data[:2] = np.dot(cloud_data[:2].T, M).T  # perform the rotation efficiently

    # # Random flipping along x and/or y axis
    if np.random.random() > 0.5:
        cloud_data[0] = -cloud_data[0]
    if np.random.random() > 0.5:
        cloud_data[1] = -cloud_data[1]

    # random gaussian noise everywhere except z and return number
    sigma, clip = 0.01, 0.03
    cloud_data[:2] = (
        cloud_data[:2]
        + np.clip(
            sigma * np.random.randn(cloud_data[:2].shape[0], cloud_data[:2].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )

    cloud_data[3:8] = (
        cloud_data[3:8]
        + np.clip(
            sigma * np.random.randn(cloud_data[3:8].shape[0], cloud_data[3:8].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )

    return cloud_data


# def cloud_collate(batch):
#     """Collates a list of dataset samples into a batch list for clouds
#     and a single array for labels
#     This function is necessary to implement because the clouds have different sizes (unlike for images)
#     """
#     clouds, labels = list(zip(*batch))
#     labels = torch.cat(labels, 0)
#     return clouds, labels


def cloud_loader_from_parcel(parcel_points_nparray, disk_center):

    pass
