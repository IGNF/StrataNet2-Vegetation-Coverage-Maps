# Objective:
# Using a shapefile, and a folder of corresponding LAS
# For each shape,
# Read the las (no rescaling/no augmentation) divide it in circles
# pickle dict (cloud_array, cloud_center, parcel_id)
# We can pickle one parcel at a time, and load one parcel at a time, if needed. https://stackoverflow.com/a/12762056/8086033
# global imports
import os, sys

repo_absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_absolute_path)
# global imports
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import shapefile
from shapely.geometry import shape


warnings.simplefilter(action="ignore")
np.random.seed(42)
torch.cuda.empty_cache()

# Local imports
from config import args
from random import shuffle

from utils.useful_functions import (
    get_filename_no_extension,
    create_dir,
    get_files_of_type_in_folder,
)
from model.point_cloud_classifier import PointCloudClassifier
from model.infer_utils import (
    divide_parcel_las_and_get_disk_centers,
    extract_points_within_disk,
)
from utils.load_las_data import transform_features_of_plot_cloud


def get_transformed_cloud_data_from_center(
    parcel_points_nparray: np.ndarray, plot_center: list, args
):
    """ """
    plots_point_nparray = extract_points_within_disk(parcel_points_nparray, plot_center)

    if plots_point_nparray.shape[0] == 0:
        return None
    else:
        plots_point_nparray = transform_features_of_plot_cloud(
            plots_point_nparray, args
        )
        return plots_point_nparray


def main():
    # Setup: save everything to the dataset_folder
    global args
    args.parcel_dataset_pkl_path = args.las_parcelles_folder_path[:-1] + "_pickled"
    create_dir(args.parcel_dataset_pkl_path)

    # Get the shapefile records and las filenames
    shp = shapefile.Reader(args.parcel_shapefile_path)
    shp_records = {rec.ID: rec for rec in shp.records()}
    las_filenames = get_files_of_type_in_folder(args.las_parcelles_folder_path, ".las")
    las_filenames = [
        l
        for l in las_filenames
        if f"{get_filename_no_extension(l)}.pckl"
        not in glob.glob(args.parcel_dataset_pkl_path, recursive=True)
    ]
    shuffle(las_filenames)

    for las_nb, las_filename in enumerate(las_filenames):

        parcel_ID = get_filename_no_extension(las_filename)

        # DEBUG : remove this debug condition
        # if args.mode == "DEV":
        #     if not las_filename.endswith("004010272-16-2.las"):  # small
        #         continue
        #     else:
        #         print("004010272-16-2.las")
        print(f"Storing data from parcel #{las_nb}/{len(las_filenames)}: {parcel_ID}")

        try:
            parcel_shape = shape(
                shp.shape(shp_records[parcel_ID].oid).__geo_interface__
            )
            (
                grid_pixel_xy_centers,
                parcel_points_nparray,
            ) = divide_parcel_las_and_get_disk_centers(
                args, las_filename, parcel_shape, save_fig_of_division=False
            )
        except ValueError:
            print(f"Problem when loading file {las_filename}.")
            print(ValueError)
            continue

        plots_data = []
        for plot_count, plot_center in enumerate(
            tqdm(
                grid_pixel_xy_centers,
                desc=f"Centers for parcel in {parcel_ID}",
                leave=True,
            )
        ):
            plot_points_tensor = get_transformed_cloud_data_from_center(
                parcel_points_nparray, plot_center, args
            )
            # TODO: to accept plots with low N, we have to account for np.nans in the rasters predictions. Unsure at this point.
            if plot_points_tensor is not None and plot_points_tensor.shape[0] > 3500:
                data = {
                    "parcel_ID": parcel_ID,
                    "pseudoplot_ID": f"PP" + str(plot_count).zfill(6),
                    "plot_points_arr": plot_points_tensor,
                    "plot_center": plot_center,
                }
                plots_data.append(data)
                # if plot_count > 30 and args.mode == "DEV":
                #     break
        with open(
            os.path.join(args.parcel_dataset_pkl_path, f"{parcel_ID}.pckl"), "wb"
        ) as f:
            pickle.dump(plots_data, f)


if __name__ == "__main__":
    main()
