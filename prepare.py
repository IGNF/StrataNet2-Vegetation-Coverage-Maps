import os, sys
import glob
import pickle
import warnings
from sys import getsizeof
import functools
import numpy as np
import pandas as pd
from random import shuffle
import torch
from tqdm import tqdm
import shapefile
import torchnet as tnt
from scipy.spatial import cKDTree as KDTree

warnings.simplefilter(action="ignore")
np.random.seed(42)
torch.cuda.empty_cache()

from config import args
from utils.utils import (
    get_filename_no_extension,
    create_dir,
    get_files_of_type_in_folder,
    setup_experiment_folder,
    create_a_logger,
    get_unprocessed_files,
)
from inference.prepare_utils import (
    divide_parcel_las_and_get_disk_centers,
    extract_cloud,
    get_shape,
)
from utils.load_data import pre_transform
from inference.prepare_utils import extract_cloud_data


# SETUP
setup_experiment_folder(args, task="prepare")
logger = create_a_logger(args)

input_folder = os.path.join(args.las_parcels_folder_path, "input/")
output_folder = os.path.join(args.las_parcels_folder_path, "prepared/")
create_dir(output_folder)

shp = shapefile.Reader(args.parcel_shapefile_path)

while True:
    unprocessed = get_unprocessed_files(input_folder, output_folder)
    unprocessed = [
        filename for filename in unprocessed if filename.lower().endswith(".las")
    ]
    if not unprocessed:
        logger.info(f"No prepared parcel found to predict on in {input_folder}")
        break
    else:
        logger.info(f"N={len(unprocessed)} parcels to prepare.")
        shuffle(unprocessed)
    filename = unprocessed.pop(-1)
    parcel_id = get_filename_no_extension(filename)

    parcel_shape = get_shape(shp, parcel_id)
    division_fig_save_path = os.path.join(output_folder, f"divisions/{parcel_id}.png")
    plot_centers, parcel_cloud = divide_parcel_las_and_get_disk_centers(
        args, filename, parcel_shape, division_fig_save_path=division_fig_save_path
    )

    size_MB = round(getsizeof(parcel_cloud) / 1024 / 1024, 2)
    logger.info(f"Size of LAS file is {size_MB}MB")

    queries = [
        {"plot_idx": idx, "plot_center": plot_center}
        for idx, plot_center in enumerate(plot_centers)
    ]
    leaf_size = 50
    parcel_tree = KDTree(parcel_cloud[:2].transpose(), leaf_size)
    dataset = tnt.dataset.ListDataset(
        queries,
        load=functools.partial(
            extract_cloud_data,
            parcel_cloud=parcel_cloud,
            parcel_tree=parcel_tree,
            args=args,
        ),
    )
    plots_data = {}

    for cloud_data in tqdm(
        dataset,
        desc=f"Parcel: {parcel_id}",
    ):
        if cloud_data is not None:
            if cloud_data["N_points_in_cloud"] > 50:
                plots_data[cloud_data["plot_id"]] = cloud_data

    output_path = os.path.join(output_folder, f"{parcel_id}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(plots_data, f)
