from comet_ml import Experiment

import warnings

warnings.simplefilter(action="ignore")

import os
import gc
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchnet as tnt
import logging

from data_loader.loader import *
from model.project_to_2d import *
from learning.loss_functions import *
from learning.test import evaluate
from learning.loss_functions import *
from learning.accuracy import *
from utils.utils import *
from model.point_net import PointNet

logger = logging.getLogger(__name__)

np.random.seed(42)


def train(model, train_set, kde_mixture, optimizer, args):
    """train for one epoch"""
    model.train()

    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()

    for index_batch, (clouds, gt) in enumerate(loader):

        if args.cuda is not None:
            gt = gt.cuda(args.cuda)

        optimizer.zero_grad(set_to_none=True)
        coverages_pointwise, proba_pointwise = model(clouds)
        pred_pl, pred_pixels = project_to_2d(coverages_pointwise, clouds, args)

        loss_abs = get_absolute_loss(pred_pl, gt)
        loss_log, _ = get_NLL_loss(proba_pointwise, clouds, kde_mixture, args)
        loss_e = get_entropy_loss(proba_pointwise)

        loss = loss_abs + args.m * loss_log + args.e * loss_e

        loss.backward()
        args.current_step_in_fold = args.current_step_in_fold + 1
        optimizer.step()

        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        loss_meter.add(loss.item())
        gc.collect()

    train_losses_dict = {
        "total_loss": loss_meter.value()[0],
        "MAE_loss": loss_meter_abs.value()[0],
        "log_loss": loss_meter_log.value()[0],
        "step": args.current_step_in_fold,
    }
    return train_losses_dict


def train_full(
    args,
    train_set,
    test_set,
    test_list,
    xy_centers_dict,
    kde_mixture,
):
    """The full training loop.
    If fold_id = -1, this is the full training and we make inferences at last epoch for this test=train set.
    If fold_id = -2, this is pretraining.
    """
    model = initialize_model(args)
    scheduler, optimizer = get_optimizers(model, args)

    set_predictions_interpretation_folder(args)

    all_epochs_train_loss_dict = []
    all_epochs_test_loss_dict = []
    cloud_info_list = None

    for i_epoch in range(1, args.n_epoch + 1):
        train_loss_dict = None
        test_loss_dict = None
        args.experiment.set_epoch(i_epoch)
        args.experiment.log_metric("learning_rate", scheduler.get_last_lr())

        # train one epoch
        with args.experiment.context_manager(f"fold_{args.current_fold_id}_train"):
            train_loss_dict = train(model, train_set, kde_mixture, optimizer, args)
            print_epoch_losses(i_epoch, train_loss_dict, train=True)
            args.experiment.log_metrics(
                train_loss_dict, epoch=i_epoch, step=train_loss_dict["step"]
            )
            train_loss_dict.update({"epoch": i_epoch})
            all_epochs_train_loss_dict.append(train_loss_dict)

        with args.experiment.context_manager(f"fold_{args.current_fold_id}_val"):

            # if not last epoch, we just evaluate performances on test plots, during cross-validation only.
            if (
                (i_epoch % args.n_epoch_test == 0)
                or (i_epoch > args.epoch_to_start_early_stop)
            ) and (args.crossvalidating):
                test_loss_dict, _ = evaluate(
                    model,
                    test_set,
                    kde_mixture,
                    args,
                    test_list,
                    xy_centers_dict,
                )
                gc.collect()
                print_epoch_losses(i_epoch, test_loss_dict, train=False)
                args.experiment.log_metrics(
                    test_loss_dict, epoch=i_epoch, step=test_loss_dict["step"]
                )
                test_loss_dict.update({"epoch": i_epoch})
                all_epochs_test_loss_dict.append(test_loss_dict)

                # if we stop early, load model state and generate summary visualizations
                if model.stop_early(test_loss_dict["total_loss"], i_epoch, args):
                    logger.info(f"Early stopping at epoch {i_epoch}")
                    break

        if (i_epoch) == args.n_epoch:
            logger.info(f"Last epoch passed n={args.n_epoch}")
            break

        scheduler.step()

    with args.experiment.context_manager(f"fold_{args.current_fold_id}_val"):

        logger.info(
            f"Load best model of epoch {model.best_metric_epoch} for final inference."
        )
        model.load_best_state(args)

        test_loss_dict, cloud_info_list = evaluate(
            model,
            test_set,
            kde_mixture,
            args,
            test_list,
            xy_centers_dict,
            last_epoch=True,
        )
        gc.collect()
        if model.stopped_early:
            args.experiment.log_metric("early_stop_epoch", model.best_metric_epoch)
        else:
            print_epoch_losses(i_epoch, test_loss_dict, train=False)

    return model, all_epochs_train_loss_dict, all_epochs_test_loss_dict, cloud_info_list


def get_optimizers(model, args):
    """Get optimize, and scheduler for LR decay."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    args.current_step_in_fold = 0
    return scheduler, optimizer


def set_predictions_interpretation_folder(args):
    args.crossvalidating = args.current_fold_id >= 0

    current_task = "crossval" if args.crossvalidating else "full"
    args.plot_path = os.path.join(args.stats_path, f"img/plots/{current_task}/")

    create_dir(args.plot_path)


def initialize_model(args):
    """Get a clean NN model, potentially using pretrained weights."""
    model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
    if args.use_PT_model:
        args.trained_model_path = get_trained_model_path_from_experiment(
            args.path, args.PT_model_id
        )
        model.load_state(args.trained_model_path)
        model.set_patience_attributes(args)
        model.train()
    logger.info(
        "Total number of parameters: {}".format(
            sum([p.numel() for p in model.parameters()])
        )
    )
    return model
