import warnings

warnings.simplefilter(action="ignore")

import torchnet as tnt
import gc
from torch.utils.tensorboard import SummaryWriter


# We import from other files
from data_loader.loader import *
from model.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from model.point_net import PointNet
from model.point_cloud_classifier import PointCloudClassifier
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchnet as tnt
from model.loss_functions import *
from model.accuracy import *

from model.test import evaluate

np.random.seed(42)


def train(model, PCC, train_set, params, optimizer, args):
    """train for one epoch"""
    model.train()

    # the loader function will take care of the batching
    loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=cloud_collate,
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

        if PCC.is_cuda:
            gt = gt.cuda()

        optimizer.zero_grad()  # put gradient to zero
        pred_pointwise, pred_pointwise_b = PCC.run(
            model, cloud
        )  # compute the pointwise prediction
        pred_pl, pred_adm, pred_pixels = project_to_2d(
            pred_pointwise, cloud, pred_pointwise_b, PCC, args
        )  # compute plot prediction

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 or 3 stratum)
        loss_abs = loss_absolute(pred_pl, gt, args)
        loss_log, likelihood = loss_loglikelihood(
            pred_pointwise, cloud, params, PCC, args
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

        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        loss_meter.add(loss.item())
        gc.collect()

    return (
        loss_meter.value()[0],
        loss_meter_abs.value()[0],
        loss_meter_log.value()[0],
        loss_meter_abs_adm.value()[0],
    )


def train_full(args, fold_id, train_set, test_set, test_list, xy_centers_dict, params):
    """The full training loop"""
    # initialize the model
    model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
    writer = SummaryWriter(os.path.join(args.stats_path, f"runs/fold_{fold_id}/"))

    print(
        "Total number of parameters: {}".format(
            sum([p.numel() for p in model.parameters()])
        )
    )
    if fold_id == 1:
        print(model)

    # define the classifier
    PCC = PointCloudClassifier(args)

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    cloud_info_list = None

    for i_epoch in range(args.n_epoch):
        scheduler.step()

        # train one epoch
        train_losses = train(model, PCC, train_set, params, optimizer, args)
        writer = write_to_writer(writer, args, i_epoch, train_losses, train=True)
        if (
            i_epoch + 1
        ) == args.n_epoch:  # if last epoch, we creare 2D images with points projections and infer values for all plots
            print("Last epoch")
            test_losses, cloud_info_list = evaluate(
                model,
                PCC,
                test_set,
                params,
                args,
                test_list,
                xy_centers_dict,
                args.stats_path,
                args.stats_file,
                last_epoch=True,
                create_final_images_bool=args.create_final_images_bool,
            )
            gc.collect()
            writer = write_to_writer(writer, args, i_epoch, test_losses, train=False)
        elif (i_epoch + 1) % args.n_epoch_test == 0:
            test_losses, _ = evaluate(
                model,
                PCC,
                test_set,
                params,
                args,
                test_list,
                xy_centers_dict,
                args.stats_path,
                args.stats_file,
            )
            gc.collect()
            writer = write_to_writer(writer, args, i_epoch, test_losses, train=False)
    writer.flush()

    final_train_losses_list = train_losses
    final_test_losses_list = test_losses
    return model, final_train_losses_list, final_test_losses_list, cloud_info_list
