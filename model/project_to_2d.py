import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean


def project_to_plotwise_coverages(pred_pointwise, clouds, args):
    """
    Get plotwise coveFrom pointwise
    """
    index_batches = []
    index_group = []
    batches_len = []
    for b, current_cloud in enumerate(clouds):
        xy = current_cloud[:2]
        xy = torch.floor(
            (xy - torch.min(xy, dim=1).values.view(2, 1).expand_as(xy))
            / (torch.max(xy, dim=1).values - torch.min(xy, dim=1).values + 0.0001)
            .view(2, 1)
            .expand_as(xy)
            * args.diam_pix
        ).int()

        unique, index = torch.unique(xy.T, dim=0, return_inverse=True)
        index_b = torch.full(torch.unique(index).size(), b)
        if args.cuda is not None:
            index = index.cuda(args.cuda)
            index_b = index_b.cuda(args.cuda)
        index = index + np.asarray(batches_len).sum()
        index_batches.append(index.type(torch.LongTensor))
        index_group.append(index_b.type(torch.LongTensor))
        batches_len.append(torch.unique(index).size(0))
    index_batches = torch.cat(index_batches)
    index_group = torch.cat(index_group)
    if args.cuda is not None:
        index_batches = index_batches.cuda(args.cuda)
        index_group = index_group.cuda(args.cuda)
    pred_pointwise_cat = torch.cat([preds for preds in pred_pointwise], dim=1)
    pixel_max = scatter_max(pred_pointwise_cat, index_batches)[0]

    c_low_veg_pix = pixel_max[0, :]
    c_bare_soil_pix = 1 - c_low_veg_pix
    c_med_veg_pix = pixel_max[2, :]
    c_high_veg_pix = pixel_max[3, :]

    c_low_veg = scatter_mean(c_low_veg_pix, index_group)
    c_bare_soil = scatter_mean(c_bare_soil_pix, index_group)
    c_med_veg = scatter_mean(c_med_veg_pix, index_group)
    c_high_veg = scatter_mean(c_high_veg_pix, index_group)

    pred_coverages = torch.stack(
        [c_low_veg, c_bare_soil, c_med_veg, c_high_veg]
    ).T  # [B, 3]

    return pred_coverages


def project_to_2d_rasters(cloud: torch.Tensor, coverages_pointwise: torch.Tensor, args):
    """
    We do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
    cloud: (2, N) 2D tensor
    Returns rasters [3, 20m, 20m]
    """

    # we get unique pixel coordinate to serve as group for raster prediction
    # Values are between 0 and args.diam_pix-1, sometimes (extremely rare) at args.diam_pix wich we correct

    scaling_factor = 10 * (args.diam_pix / args.diam_meters)  # * pix/normalized_unit
    xy = cloud[:2, :]
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
    image_high_veg = np.full((args.diam_pix, args.diam_pix), np.nan)

    for i in np.unique(inverse):
        where = np.where(inverse == i)[0]
        k, m = xy.T[where][0]
        maxpool = nn.MaxPool1d(len(where))
        max_pool_val = (
            maxpool(coverages_pointwise[:, where].unsqueeze(0))
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )
        sum_val = coverages_pointwise[:, where].sum(axis=1)

        proba_low_veg = max_pool_val[0]
        proba_med_veg = max_pool_val[2]
        proba_high_veg = max_pool_val[3]

        image_low_veg[m, k] = proba_low_veg
        image_med_veg[m, k] = proba_med_veg
        image_high_veg[m, k] = proba_high_veg

    # We flip along y axis as the 1st raster row starts with 0
    image_low_veg = np.flip(image_low_veg, axis=0)
    image_med_veg = np.flip(image_med_veg, axis=0)
    image_high_veg = np.flip(image_high_veg, axis=0)

    rasters = np.concatenate(([image_low_veg], [image_med_veg], [image_high_veg]), 0)
    return rasters
