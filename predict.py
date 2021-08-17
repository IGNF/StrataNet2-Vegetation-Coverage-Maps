# global imports
import os
import warnings
import pickle
import functools
import glob
import logging
from tqdm import tqdm

import pandas as pd
from shapely.geometry import shape
import shapefile
import numpy as np
import torch
import torchnet as tnt
from argparse import ArgumentParser

warnings.simplefilter(action="ignore")
np.random.seed(42)
torch.cuda.empty_cache()

# Local imports
from config import args
from utils.utils import (
    create_a_logger,
    setup_experiment_folder,
    get_filename_no_extension,
    create_dir,
    get_unprocessed_files,
    update_namespace_with_another_namespace,
)
from learning.train import find_pretrained_model, initialize_model
from inference.predict_utils import (
    create_geotiff_raster,
    load_dataset,
    create_dataloader,
    filter_dataset,
    update_shapefile_with_predictions,
    define_plot_geotiff_output_path,
)
from inference.prepare_utils import get_shape
from inference.geotiff_raster import merge_geotiff_rasters
from model.project_to_2d import project_to_plotwise_coverages

parser = ArgumentParser(description="predict")
parser.add_argument(
    "--task",
    default="inference",
    choices=["inference", "pseudo_labelling"],
    help="Weither to predict parcel-level rasters of coverages or to predict pseudo-labels into a new dataset for pretraining.",
)

args_local, _ = parser.parse_known_args()
args = update_namespace_with_another_namespace(args, args_local)


# SETUP
setup_experiment_folder(args, task=args.task)
is_pseudo_labelling = args.task == "pseudo_labelling"
logger = create_a_logger(args)

# MODEL
torch.set_grad_enabled(False)
model_path, model_id = find_pretrained_model(args)
assert model_id
model = initialize_model(args, model_path)
model.eval()

# DATA
input_folder = os.path.join(args.las_parcels_folder_path, "prepared")
output_folder = os.path.join(
    args.las_parcels_folder_path, os.path.join(args.task, model_id)
)
create_dir(output_folder)
if not is_pseudo_labelling:
    shp = shapefile.Reader(args.parcel_shapefile_path)

while True:

    unprocessed = get_unprocessed_files(input_folder, output_folder)
    if not unprocessed:
        logging.info(f"No more prepared parcel to predict on in {input_folder}")
        break
    filename = unprocessed.pop(0)
    parcel_id = get_filename_no_extension(filename)

    dataset = load_dataset(filename)
    dataset = filter_dataset(dataset, is_pseudo_labelling)
    dataloader = create_dataloader(dataset, args)
    i = 0
    for cloud_data in tqdm(
        dataloader, desc=f"Inference on {parcel_id}", total=len(dataloader)
    ):
        clouds = cloud_data["cloud"]
        plot_ids = cloud_data["plot_id"]
        plot_centers = cloud_data["plot_center"]

        coverages_pointwise, _ = model(cloud_data)
        if is_pseudo_labelling:
            pred_coverages = project_to_plotwise_coverages(
                coverages_pointwise, clouds, args
            )

            pred_coverages = pred_coverages.cpu().detach().numpy()
            for plot_name, pred in zip(plot_ids, pred_coverages):
                dataset[plot_name].update({"coverages": pred.squeeze()})
        else:
            for idx, plot_id in enumerate(plot_ids):
                plot_center = plot_centers[idx]
                output_path = define_plot_geotiff_output_path(
                    output_folder, parcel_id, plot_id, plot_center
                )
                create_geotiff_raster(
                    coverages_pointwise[idx],
                    clouds[idx],
                    plot_center,
                    output_path,
                    args,
                )
        i = i + 1
        if args.mode == "DEV" and i > 10:
            break

    if is_pseudo_labelling:
        output_path = os.path.join(output_folder, parcel_id + ".pkl")
        with open(output_path, "wb") as pfile:
            pickle.dump(dataset, pfile)
    else:
        intermediate_tiffs_folder = os.path.dirname(output_path)
        final_tiff_path = os.path.join(output_folder, f"{parcel_id}.tif")
        parcel_shape = get_shape(shp, parcel_id)
        message = merge_geotiff_rasters(
            final_tiff_path, intermediate_tiffs_folder, parcel_shape
        )
        logging.info(message)

    if args.mode == "DEV":
        break

if not is_pseudo_labelling:
    update_shapefile_with_predictions(args.parcel_shapefile_path, output_folder)
