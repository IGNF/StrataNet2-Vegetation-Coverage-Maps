# We import from other files
import imp

from comet_ml import Experiment
from utils.create_final_images import *
from data_loader.loader import *
from model.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from utils.useful_functions import create_dir
import torchnet as tnt
import gc
import os
from PIL import Image

np.random.seed(42)


@torch.no_grad()
def evaluate(
    model,
    PCC,
    test_set,
    params,
    args,
    test_list,
    xy_centers_dict,
    stats_path,
    stats_file,
    last_epoch=False,
    plot_only_png=True,
    situation="crossval",
):
    """Eval on test set and inference if this is the last epoch
    Outputs are: average losses (printed), infered values (csv) , k trained models, stats, and images.
    Everything is saved under /experiments/ folder.
    """

    model.eval()

    loader = torch.utils.data.DataLoader(
        test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False
    )
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs_gl = tnt.meter.AverageValueMeter()
    loss_meter_abs_ml = tnt.meter.AverageValueMeter()
    loss_meter_abs_hl = tnt.meter.AverageValueMeter()
    loss_meter_abs_adm = tnt.meter.AverageValueMeter()

    cloud_info_list = []
    last_G_tensor_list = []
    for index_cloud, (cloud, gt) in enumerate(loader):
        plot_name = test_list[index_cloud]

        if PCC.is_cuda:
            gt = gt.cuda()

        pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)
        pred_pl, pred_adm, pred_pixels = project_to_2d(
            pred_pointwise, cloud, pred_pointwise_b, PCC, args
        )

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 stratum)
        loss_abs = loss_absolute(pred_pl, gt, args)  # absolut loss
        loss_log, likelihood = loss_loglikelihood(
            pred_pointwise, cloud, params, PCC, args
        )  # negative loglikelihood loss

        if args.ent:
            loss_e = loss_entropy(pred_pixels)

        if args.adm:
            # we compute admissibility loss
            loss_adm = loss_abs_adm(pred_adm, gt)
            if args.ent:
                # Losses : coverage, log-likelihood, admissibility, entropy
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm + args.e * loss_e
            else:
                # Losses: coverage, log-likelihood, and admissibility losses
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm
            loss_meter_abs_adm.add(loss_adm.item())
        else:
            if args.ent:
                # losses: coverage, loss-likelihood, entropy
                loss = loss_abs + args.m * loss_log + args.e * loss_e
            else:
                # losses: coverage, loss-likelihood
                loss = loss_abs + args.m * loss_log

        loss_meter.add(loss.item())
        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        gc.collect()

        # This is where we get results
        # give separate losses for each stratum
        component_losses = loss_absolute(
            pred_pl, gt, args, level_loss=True
        )  # gl_mv_loss gives separated losses for each stratum
        if args.nb_stratum == 2:
            loss_abs_gl, loss_abs_ml = component_losses
        else:
            loss_abs_gl, loss_abs_ml, loss_abs_hl = component_losses
            loss_abs_hl = loss_abs_hl[~torch.isnan(loss_abs_hl)]
            if loss_abs_hl.size(0) > 0:
                loss_meter_abs_hl.add(loss_abs_hl.item())
        loss_meter_abs_gl.add(loss_abs_gl.item())
        loss_meter_abs_ml.add(loss_abs_ml.item())

        # Save visualizatins to visualize final results OR track progress for a selection of plots
        if last_epoch or plot_name in args.plot_name_to_visualize_during_training:
            plot_path = os.path.join(stats_path, f"img/placettes/{situation}/")
            create_dir(plot_path)
            png_path = create_final_images(
                pred_pl,
                gt,
                pred_pointwise_b,
                cloud,
                likelihood,
                plot_name,
                xy_centers_dict,
                plot_path,
                args,
                adm=pred_adm,
                plot_only_png=plot_only_png,
            )  # create final images with stratum values

        if last_epoch:
            # Keep and format prediction from pred_pl
            pred_pl_cpu = pred_pl.cpu().numpy()[0]
            gt_cpu = gt.cpu().numpy()[0]
            cloud_info = {
                "pl_id": plot_name,
                "pl_N_points": pred_pointwise.shape[0],
                "pred_veg_b": pred_pl_cpu[0],
                "pred_sol_nu": pred_pl_cpu[1],
                "pred_veg_moy": pred_pl_cpu[2],
                "pred_veg_h": pred_pl_cpu[3],
                "vt_veg_b": gt_cpu[0],
                "vt_sol_nu": gt_cpu[1],
                "vt_veg_moy": gt_cpu[2],
                "vt_veg_h": gt_cpu[3],
            }
            cloud_info_list.append(cloud_info)
            if isinstance(args.experiment, Experiment):
                # log the embeddings for this plot
                last_G_tensor_list.append(
                    [model.last_G_tensor.cpu().numpy(), plot_name, png_path]
                )

    # Here we log histograms of the absolute errors
    if last_epoch and isinstance(args.experiment, Experiment):
        args.experiment.log_histogram_3d(
            [abs(info["pred_veg_b"] - info["vt_veg_b"]) for info in cloud_info_list],
            name="val_MAE_veg_b",
            step=args.current_fold_id,
            epoch=args.current_fold_id,
        )
        args.experiment.log_histogram_3d(
            [
                abs(info["pred_veg_moy"] - info["vt_veg_moy"])
                for info in cloud_info_list
            ],
            name="val_MAE_veg_moy",
            step=args.current_fold_id,
            epoch=args.current_fold_id,
        )
        args.experiment.log_histogram_3d(
            [abs(info["pred_veg_h"] - info["vt_veg_h"]) for info in cloud_info_list],
            name="val_MAE_veg_h",
            step=args.current_fold_id,
            epoch=args.current_fold_id,
        )

        # Here we log embeddings of test plot for this fold
        image_data = [
            Image.open(a[2]).convert("RGB").resize((500, 720))
            for a in last_G_tensor_list
        ]
        args.experiment.log_embedding(
            [a[0] for a in last_G_tensor_list],
            [a[1] for a in last_G_tensor_list],
            image_data=image_data,
            image_transparent_color=(0, 0, 0),
            image_size=image_data[0].size,
            title="G_tensor",
        )

    return (
        {
            "total_loss": loss_meter.value()[0],
            "MAE_loss": loss_meter_abs.value()[0],
            "log_loss": loss_meter_log.value()[0],
            "MAE_veg_b": loss_meter_abs_gl.value()[0],
            "MAE_veg_moy": loss_meter_abs_ml.value()[0],
            "MAE_veg_h": loss_meter_abs_hl.value()[0],
            "adm_loss": loss_meter_abs_adm.value()[0],
        },
        cloud_info_list,
    )
