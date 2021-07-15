import numpy as np
import torch
from torch_scatter import scatter_max, scatter_mean, scatter_sum


def project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args):
    """
    We compute the coverage scores :
    pred_pl - [Bx4] prediction vector for the plot
    scores -  [(BxN)x2] probas_ground_nonground that a point belongs to stratum 1 or stratum 2
    """
    index_batches = []
    index_group = []
    batches_len = []

    # we project 3D points to 2D plane
    # We use torch scatter to process
    for b in range(len(pred_pointwise_b)):
        current_cloud = cloud[b]
        xy = current_cloud[:2]
        xy = torch.floor(
            (xy - torch.min(xy, dim=1).values.view(2, 1).expand_as(xy))
            / (torch.max(xy, dim=1).values - torch.min(xy, dim=1).values + 0.0001)
            .view(2, 1)
            .expand_as(xy)
            * args.diam_pix
        ).int()

        unique, index = torch.unique(xy.T, dim=0, return_inverse=True)
        index_b = torch.full(
            torch.unique(index).size(), b
        )  # b is index of cloud in the batch
        if PCC.cuda_device is not None:
            index = index.cuda(PCC.cuda_device)
            index_b = index_b.cuda(PCC.cuda_device)
        index = index + np.asarray(batches_len).sum()
        index_batches.append(index.type(torch.LongTensor))
        index_group.append(index_b.type(torch.LongTensor))
        batches_len.append(torch.unique(index).size(0))
    index_batches = torch.cat(index_batches)
    index_group = torch.cat(index_group)
    if PCC.cuda_device is not None:
        index_batches = index_batches.cuda(PCC.cuda_device)
        index_group = index_group.cuda(PCC.cuda_device)
    pixel_max = scatter_max(pred_pointwise.permute(1, 0), index_batches)[0]
    pixel_sum = scatter_sum(pred_pointwise.permute(1, 0), index_batches)

    # We compute prediction values per pixel
    if (
        args.norm_ground
    ):  # we normalize ground level coverage values, so c_low[i]+c_bare[i]=1
        c_low_veg_pix = pixel_sum[0, :] / (pixel_sum[:2, :].sum(0))
        c_bare_soil_pix = pixel_sum[1, :] / (pixel_sum[:2, :].sum(0))
    else:  # we do not normalize anything, as bare soil coverage does not participate in absolute loss
        c_low_veg_pix = pixel_max[0, :]
        c_bare_soil_pix = 1 - c_low_veg_pix
    c_med_veg_pix = pixel_max[2, :]

    if args.nb_stratum == 2:
        # We compute prediction values per plot
        c_low_veg = scatter_mean(c_low_veg_pix, index_group)
        c_bare_soil = scatter_mean(c_bare_soil_pix, index_group)
        c_med_veg = scatter_mean(c_med_veg_pix, index_group)
        # c_other = scatter_mean(c_other_pix, index_group)
        pred_pl = torch.stack([c_low_veg, c_bare_soil, c_med_veg]).T

        pred_pixel = torch.stack([c_low_veg_pix, c_med_veg_pix]).T

    else:  # 3 stratum
        c_high_veg_pix = pixel_max[
            3, :
        ]  # we equally compute raster for high vegetation

        # We compute prediction values per plot
        c_low_veg = scatter_mean(c_low_veg_pix, index_group)
        c_bare_soil = scatter_mean(c_bare_soil_pix, index_group)
        c_med_veg = scatter_mean(c_med_veg_pix, index_group)
        c_high_veg = scatter_mean(c_high_veg_pix, index_group)
        pred_pl = torch.stack([c_low_veg, c_bare_soil, c_med_veg, c_high_veg]).T
        pred_pixel = torch.stack([c_low_veg_pix, c_med_veg_pix, c_high_veg_pix]).T

    if args.adm:
        # If we do consider admissibility, it is with the purpose of getting admissibility rasters (c_adm_pix)
        c_adm_pix = torch.max(pixel_max[[0, 2], :], dim=0)[0]
        c_adm = scatter_mean(c_adm_pix, index_group)
        return pred_pl, c_adm, c_adm_pix
    else:
        # we get 3 or 4 (soft) coverage rasters, one for each stratum
        c_adm = None
        return pred_pl, c_adm, pred_pixel
