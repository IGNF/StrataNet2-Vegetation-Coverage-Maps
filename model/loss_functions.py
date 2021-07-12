import torch
import numpy as np
from scipy.stats import gamma


EPS = 0.0001


# Negative loglikelihood loss
def loss_loglikelihood(pred_pointwise, cloud, kde_mixture, PCC, args):

    # We extract heights of every point
    z_all = np.empty((0))
    for current_cloud in cloud:
        z = current_cloud[2] * args.z_max  # we go back from scaled data
        z_all = np.append(z_all, z)

    z_all = np.asarray(z_all).reshape(-1)

    pdf_ground, pdf_m, pdf_h = kde_mixture.predict(z_all)

    pdf_all = np.concatenate(
        (pdf_ground.reshape(-1, 1), pdf_m.reshape(-1, 1), pdf_h.reshape(-1, 1)), 1
    )
    pdf_all = torch.tensor(pdf_all)

    p_ground = pred_pointwise[:, :2].sum(1)
    p_m = pred_pointwise[:, 2:3].sum(1)
    p_h = pred_pointwise[:, 3:4].sum(1)

    if PCC.is_cuda:
        pdf_all = pdf_all.cuda()
        p_ground = p_ground.cuda()
        p_m = p_m.cuda()
        p_h = p_h.cuda()

    p_all = torch.cat((p_ground.view(-1, 1), p_m.view(-1, 1), p_h.view(-1, 1)), 1)
    likelihood = torch.mul(p_all, pdf_all)

    return -torch.log(likelihood.sum(1)).mean(), likelihood, (p_all, pdf_all)


# Admissibility loss
def loss_abs_adm(pred_adm, gt_adm):
    return ((pred_adm - gt_adm[:, -1]).pow(2) + EPS).pow(0.5).mean()


def loss_absolute(pred_pl, gt, args, level_loss=False):
    """
    level_loss: wheather we want to obtain losses for different vegetation levels separately
    Order of separated losses is "veg_low",  "veg_moy" [, "veg_high"]
    """
    if args.nb_stratum == 2:
        if (
            level_loss
        ):  # if we want to get separate losses for ground level and medium level
            return ((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + EPS).pow(0.5).mean(0)
        return ((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + EPS).pow(0.5).mean()
    if args.nb_stratum == 3:
        gt_has_values = ~torch.isnan(gt)
        gt_has_values = gt_has_values[:, [0, 2, 3]]
        if (
            level_loss
        ):  # if we want to get separate losses for ground level and medium level
            return (
                ((pred_pl[:, [0, 2, 3]] - gt[:, [0, 2, 3]]).pow(2) + EPS)
                .pow(0.5)
                .mean(0)
            )
        return (
            (
                (
                    pred_pl[:, [0, 2, 3]][gt_has_values]
                    - gt[:, [0, 2, 3]][gt_has_values]
                ).pow(2)
                + EPS
            )
            .pow(0.5)
            .mean()
        )


def loss_entropy(pred_pixels):
    """Loss entropy on coverage raster (probabilities) to favor coverage values close to 0 or 1 for medium and high vegetation.
    This should avoid low pixel values where there are typically no non-ground vegetation."""
    return -(
        pred_pixels[:, 1:] * torch.log(pred_pixels[:, 1:] + EPS)
        + (1 - pred_pixels[:, 1:]) * torch.log(1 - pred_pixels[:, 1:] + EPS)
    ).mean()
