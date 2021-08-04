import torch
import numpy as np
from scipy.stats import gamma


EPS = 0.0001


def get_absolute_loss_by_strata(pred_pl, gt):
    """Get MAE loss for  "veg_low",  "veg_moy", "veg_high"."""
    return ((pred_pl[:, [0, 2, 3]] - gt[:, [0, 2, 3]]).pow(2) + EPS).pow(0.5).mean(0)


def get_absolute_loss(pred_pl, gt):
    """Get total MAE loss."""
    return get_absolute_loss_by_strata(pred_pl, gt).mean()


def get_entropy_loss(pred_pixels):
    """Loss entropy on coverage raster (probabilities) to favor class membership probabilities close to 0 or 1."""
    return -(
        pred_pixels[:, 2:] * torch.log(pred_pixels[:, 2:] + EPS)
        + (1 - pred_pixels[:, 2:]) * torch.log(1 - pred_pixels[:, 2:] + EPS)
    ).mean()


def get_NLL_loss(pred_pointwise, cloud, args):
    """Negative log-likelihood based on three KDEs fitted on z feature."""

    z_all = np.empty((0))
    for current_cloud in cloud:
        z = current_cloud[2] * args.z_max
        z_all = np.append(z_all, z)

    z_all = np.asarray(z_all).reshape(-1)

    pdf_ground, pdf_m, pdf_h = args.kde_mixture.predict(z_all)

    pdf_all = np.concatenate(
        (pdf_ground.reshape(-1, 1), pdf_m.reshape(-1, 1), pdf_h.reshape(-1, 1)), 1
    )
    pdf_all = torch.tensor(pdf_all)

    p_ground = pred_pointwise[:, :2].sum(1)
    p_m = pred_pointwise[:, 2:3].sum(1)
    p_h = pred_pointwise[:, 3:4].sum(1)

    if args.cuda is not None:
        pdf_all = pdf_all.cuda(args.cuda)
        p_ground = p_ground.cuda(args.cuda)
        p_m = p_m.cuda(args.cuda)
        p_h = p_h.cuda(args.cuda)

    p_all = torch.cat((p_ground.view(-1, 1), p_m.view(-1, 1), p_h.view(-1, 1)), 1)
    likelihood = torch.mul(p_all, pdf_all)

    return -torch.log(likelihood.sum(1)).mean(), (p_all, pdf_all)
