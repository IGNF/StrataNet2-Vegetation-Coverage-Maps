# global imports
import glob
import os
import warnings
import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm
import shapefile

warnings.simplefilter(action="ignore")
np.random.seed(42)
torch.cuda.empty_cache()

# Local imports
from config import args
from utils.useful_functions import (
    create_new_experiment_folder,
    fast_scandir,
    get_filename_no_extension,
    print_stats,
    get_files_of_type_in_folder,
    get_trained_model_path_from_experiment,
)
from model.point_cloud_classifier import PointCloudClassifier
from model.infer_utils import (
    divide_parcel_las_and_get_disk_centers,
    create_geotiff_raster,
    merge_geotiff_rasters,
    get_and_prepare_cloud_around_center,
)

args.z_max = 24.14  # the TRAINING args should be loaded from stats.csv/txt...


def main():

    # Create the result folder
    create_new_experiment_folder(args, infer_mode=True)  # new paths are added to args

    # find parcels .LAS files
    las_filenames = get_files_of_type_in_folder(args.las_parcelles_folder_path, ".las")
    print(las_filenames)

    # Load a saved the classifier
    trained_model_path = get_trained_model_path_from_experiment(
        args.path, args.inference_model_id, use_full_model=False
    )
    model = torch.load(trained_model_path)
    model.eval()
    PCC = PointCloudClassifier(args)

    print_stats(
        args.stats_file,
        f"Trained model loaded from {trained_model_path}",
        print_to_console=True,
    )

    for las_filename in las_filenames:
        # TODO : remove this debug condition
        if args.mode == "DEV":
            if not las_filename.endswith("004000715-5-18.las"):  # small
                continue

        print_stats(
            args.stats_file,
            f"Inference on parcel file {las_filename}",
            print_to_console=True,
        )

        # Divide parcel into plots
        (
            grid_pixel_xy_centers,
            parcel_points_nparray,
        ) = divide_parcel_las_and_get_disk_centers(
            args, las_filename, save_fig_of_division=True
        )
        # print(f"File {las_filename} with shape {parcel_points_nparray.shape}")
        plot_name = get_filename_no_extension(las_filename)
        centers = tqdm(
            grid_pixel_xy_centers,
            desc=f"Centers for parcel in {plot_name}",
            leave=True,
        )
        # TODO: replace this loop by a cleaner ad-hoc DataLoader
        for plot_center in centers:
            plot_points_tensor = get_and_prepare_cloud_around_center(
                parcel_points_nparray, plot_center, args
            )
            if plot_points_tensor is not None:
                pred_pointwise, _ = PCC.run(model, plot_points_tensor)
                create_geotiff_raster(
                    args,
                    pred_pointwise.permute(
                        1, 0
                    ),  # pred_pointwise was permuted from (N_scores, N_points) to (N_points, N_scores) at the end of PCC.run
                    plot_points_tensor[0, :, :],  # (N_feats, N_points) cloud 2D tensor
                    plot_center,
                    plot_name,
                )

        # Then
        merge_geotiff_rasters(args, plot_name)

    # Now, compute the average values from the predicted rasters
    sf = shapefile.Reader(args.parcel_shapefile_path)
    records = {rec.ID: rec for rec in sf.records()}
    predictions_tif = glob.glob(
        os.path.join(args.stats_path, "**/prediction_raster_parcel_*.tif"),
        recursive=True,
    )
    metadata_list = []
    for tif in predictions_tif:
        metadata = get_parcel_info_and_predictions(tif, records)
        metadata_list.append(metadata)
    # export to a csv
    df_inference = pd.DataFrame(metadata_list)
    csv_path = os.path.join(args.stats_path, "PCC_inference_all_parcels.csv")
    df_inference.to_csv(csv_path, index=False)
    print_stats(args.stats_file, f"Saved inference results to {csv_path}")


def get_parcel_info_and_predictions(tif, records):
    """From a prediction tif given by  its path and the records obtained from a shapefile,
    get the parcel metadata as well as the predictions : coverage and admissibility
    """
    mosaic = rasterio.open(tif).read()

    # Vb, Vmoy, Vh, Vmoy_hard
    band_means = np.nanmean(mosaic[:4], axis=(1, 2))

    # TODO: change the calculation of admissibility here
    admissibility = np.nanmean(np.nanmax([[mosaic[0]], [mosaic[0]]], axis=0))

    tif_name = get_filename_no_extension(tif).replace("prediction_raster_parcel_", "")
    rec = records[tif_name]

    metadata = {
        "NOM": tif_name,
        "SURFACE_m2": rec._area,
        "SURFACE_ha": np.round((rec._area) / 10000, 2),
        "SURF_ADM_ha": rec.SURF_ADM,
        "REPORTED_ADM": float(rec.ADM),
    }
    infered_values = {
        "pred_veg_b": band_means[0],
        "pred_veg_moy": band_means[1],
        "pred_veg_h": band_means[2],
        "adm_max_over_veg_b_and_veg_moy": admissibility,
        "pred_veg_moy_hard": band_means[3],
    }
    metadata.update(infered_values)
    return metadata


if __name__ == "__main__":
    main()
