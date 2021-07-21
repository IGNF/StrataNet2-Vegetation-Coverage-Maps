from comet_ml import Experiment

import warnings

warnings.simplefilter(action="ignore")

import torchnet as tnt
import gc


# We import from other files
from data_loader.loader import *
from model.reproject_to_2d_and_predict_plot_coverage import *
from learning.loss_functions import *

from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchnet as tnt
from learning.loss_functions import *
from learning.accuracy import *

from learning.test import evaluate
import logging

logger = logging.getLogger(__name__)

np.random.seed(42)


def train(model, PCC, train_set, kde_mixture, optimizer, args):
    """train for one epoch"""
    model.train()

    # the loader function will take care of the batching
    loader = torch.utils.data.DataLoader(
        train_set,
        # collate_fn=cloud_collate,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # will keep track of the loss
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter_abs_adm = tnt.meter.AverageValueMeter()

    for index_batch, (cloud, gt) in enumerate(loader):

        if PCC.cuda_device is not None:
            gt = gt.cuda(PCC.cuda_device)

        optimizer.zero_grad(set_to_none=True)  # put gradient to zero
        pred_pointwise, pred_pointwise_b = PCC.run(
            model, cloud
        )  # compute the pointwise prediction
        pred_pl, pred_adm, pred_pixels = project_to_2d(
            pred_pointwise, cloud, pred_pointwise_b, PCC, args
        )  # compute plot prediction

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 or 3 stratum)
        loss_abs = loss_absolute(pred_pl, gt, args)
        loss_log, _, _ = loss_loglikelihood(
            pred_pointwise, cloud, kde_mixture, PCC, args
        )  # negative loglikelihood loss + likelihood value (not-used)

        if args.ent:
            loss_e = loss_entropy(pred_pixels)

        if args.adm:
            # we compute admissibility loss
            loss_adm = loss_abs_adm(pred_adm, gt)
            loss = loss_abs + args.m * loss_log + 0.5 * loss_adm
            if args.ent:
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm + args.e * loss_e
            else:
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm
            loss_meter_abs_adm.add(loss_adm.item())
        else:
            loss = loss_abs + args.m * loss_log
            if args.ent:
                loss = loss_abs + args.m * loss_log + args.e * loss_e
            else:
                loss = loss_abs + args.m * loss_log

        loss.backward()
        optimizer.step()
        args.current_step_in_fold = args.current_step_in_fold + 1
        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        loss_meter.add(loss.item())
        gc.collect()

    train_losses_dict = {
        "total_loss": loss_meter.value()[0],
        "MAE_loss": loss_meter_abs.value()[0],
        "log_loss": loss_meter_log.value()[0],
        "adm_loss": loss_meter_abs_adm.value()[0],
        "step": args.current_step_in_fold,
    }
    return train_losses_dict


def train_full(
    args,
    fold_id,
    train_set,
    test_set,
    test_list,
    xy_centers_dict,
    model,
    PCC,
    kde_mixture,
):
    """The full training loop.
    If fold_id = -1, this is the full training and we make inferences at last epoch for this test=train set.
    """
    experiment = args.experiment
    args.current_fold_id = fold_id

    writer = SummaryWriter(os.path.join(args.stats_path, f"runs/fold_{fold_id}/"))
    logger.info(
        "Total number of parameters: {}".format(
            sum([p.numel() for p in model.parameters()])
        )
    )

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    args.current_step_in_fold = 0

    all_epochs_train_loss_dict = []
    all_epochs_test_loss_dict = []
    cloud_info_list = None

    for i_epoch in range(1, args.n_epoch + 1):
        train_loss_dict = None
        test_loss_dict = None
        experiment.set_epoch(i_epoch)
        experiment.log_metric("learning_rate", scheduler.get_last_lr())

        # train one epoch
        with experiment.context_manager(f"fold_{args.current_fold_id}_train"):
            train_loss_dict = train(model, PCC, train_set, kde_mixture, optimizer, args)
            writer = write_to_writer(writer, args, i_epoch, train_loss_dict, train=True)

            experiment.log_metrics(
                train_loss_dict, epoch=i_epoch, step=train_loss_dict["step"]
            )
        train_loss_dict.update({"epoch": i_epoch})
        all_epochs_train_loss_dict.append(train_loss_dict)

        with experiment.context_manager(f"fold_{args.current_fold_id}_val"):

            # if not last epoch, we just evaluate performances on test plots, during cross-validation only.
            if (
                (i_epoch % args.n_epoch_test == 0)
                or (i_epoch > args.epoch_to_start_early_stop)
            ) and (fold_id > 0):
                test_loss_dict, _ = evaluate(
                    model,
                    PCC,
                    test_set,
                    kde_mixture,
                    args,
                    test_list,
                    xy_centers_dict,
                    args.stats_path,
                    args.stats_file,
                )
                gc.collect()
                writer = write_to_writer(
                    writer, args, i_epoch, test_loss_dict, train=False
                )
                experiment.log_metrics(
                    test_loss_dict, epoch=i_epoch, step=test_loss_dict["step"]
                )
                test_loss_dict.update({"epoch": i_epoch})
                all_epochs_test_loss_dict.append(test_loss_dict)

                # if we stop early, load model state and generate summary visualizations
                if model.stop_early(
                    test_loss_dict["total_loss"], i_epoch, fold_id, args
                ):
                    logger.info(f"Early stopping at epoch {i_epoch}")
                    break

        if (i_epoch) == args.n_epoch:
            logger.info(f"Last epoch passed n={args.n_epoch}")
            break

        scheduler.step()

    with experiment.context_manager(f"fold_{args.current_fold_id}_val"):

        logger.info(
            f"Load best model of epoch {model.best_metric_epoch} for final inference."
        )
        model.load_best_state(fold_id, args)

        test_loss_dict, cloud_info_list = evaluate(
            model,
            PCC,
            test_set,
            kde_mixture,
            args,
            test_list,
            xy_centers_dict,
            args.stats_path,
            args.stats_file,
            last_epoch=True,
            plot_only_png=args.plot_only_png,
            situation="crossval" if fold_id >= 0 else "full",
        )
        gc.collect()
        if model.stopped_early:
            experiment.log_metric("early_stop_epoch", model.best_metric_epoch)
        else:
            writer = write_to_writer(writer, args, i_epoch, test_loss_dict, train=False)

    writer.flush()

    return model, all_epochs_train_loss_dict, all_epochs_test_loss_dict, cloud_info_list
