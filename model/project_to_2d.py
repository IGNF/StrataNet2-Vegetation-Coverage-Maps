import numpy as np
import torch
from torch_scatter import scatter_max, scatter_mean


def project_to_2d(pred_pointwise, clouds, args):
    """
    We compute the coverage scores :
    pred_pl - [Bx4] prediction vector for the plot
    scores -  [(BxN)x2] probas_ground_nonground that a point belongs to stratum 1 or stratum 2
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

    pred_pl = torch.stack([c_low_veg, c_bare_soil, c_med_veg, c_high_veg]).T  # [B, 3]
    pred_pixel = torch.stack(
        [c_low_veg_pix, c_med_veg_pix, c_high_veg_pix]
    ).T  # [N_pixels_batch, 4]

    return pred_pl, pred_pixel
