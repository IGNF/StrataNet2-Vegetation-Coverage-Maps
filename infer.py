# global imports
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import shapefile
from shapely.geometry import shape

from codetiming import Timer

warnings.simplefilter(action="ignore")
np.random.seed(42)
torch.cuda.empty_cache()

# Local imports
from config import args
from utils.useful_functions import (
    create_new_experiment_folder,
    get_args_from_prev_config,
    get_filename_no_extension,
    print_stats,
    get_trained_model_path_from_experiment,
    create_a_logger,
)
from model.point_cloud_classifier import PointCloudClassifier
from model.infer_utils import (
    divide_parcel_las_and_get_disk_centers,
    create_geotiff_raster,
    get_list_las_files_not_infered_yet,
    log_inference_times,
    make_parcel_predictions_csv,
    merge_geotiff_rasters,
    get_and_prepare_cloud_around_center,
)


def main():
    # Setup
    global args
    args = get_args_from_prev_config(args, args.inference_model_id)
    create_new_experiment_folder(
        args, infer_mode=True, resume_last_job=args.resume_last_job
    )
    trained_model_path = get_trained_model_path_from_experiment(
        args.path, args.inference_model_id, use_full_model=True
    )
    model = torch.load(trained_model_path)
    model.eval()
    PCC = PointCloudClassifier(args)
    print_stats(
        args.stats_file,
        f"Trained model loaded from {trained_model_path}",
    )
    logger = create_a_logger(args)

    # Get the shapefile records and las filenames
    shp = shapefile.Reader(args.parcel_shapefile_path)
    shp_records = {rec.ID: rec for rec in shp.records()}
    las_filenames = get_list_las_files_not_infered_yet(
        args.stats_path, args.las_parcelles_folder_path
    )
    logger.info(f"N={len(las_filenames)} parcels to infer on.")

    for las_filename in las_filenames:

        parcel_ID = get_filename_no_extension(las_filename)

        # DEBUG : remove this debug condition
        # if args.mode == "DEV":
        #     if not las_filename.endswith("004000715-5-18.las"):  # small
        #         continue

        logger.info(
            f"Inference on parcel file {las_filename}",
        )

        t = Timer(name="duration_divide_seconds")
        t.start()
        try:
            # TODO: extract as a method
            parcel_shape = shape(
                shp.shape(shp_records[parcel_ID].oid).__geo_interface__
            )
            (
                grid_pixel_xy_centers,
                parcel_points_nparray,
            ) = divide_parcel_las_and_get_disk_centers(
                args, las_filename, parcel_shape, save_fig_of_division=True
            )
        except ValueError:
            print_stats(args.stats_file, f"Problem when loading file {las_filename}")
            print(ValueError)
            continue
        t.stop()

        t.name = "duration_predict_seconds"
        t.start()
        # TODO: replace this loop by a cleaner ad-hoc DataLoader ?
        # TODO: parallelize this loop - everything is independant except the loader model which could be multiplied ?
        for plot_center in tqdm(
            grid_pixel_xy_centers,
            desc=f"Centers for parcel in {parcel_ID}",
            leave=True,
        ):
            plot_points_tensor = get_and_prepare_cloud_around_center(
                parcel_points_nparray, plot_center, args
            )
            if plot_points_tensor is not None and plot_points_tensor.shape[-1] > 500:
                with torch.no_grad():
                    pred_pointwise, _ = PCC.run(model, plot_points_tensor)
                    create_geotiff_raster(
                        args,
                        pred_pointwise.permute(
                            1, 0
                        ),  # pred_pointwise was permuted from (N_scores, N_points) to (N_points, N_scores) at the end of PCC.run
                        plot_points_tensor[
                            0, :, :
                        ],  # (N_feats, N_points) cloud 2D tensor
                        plot_center,
                        parcel_ID,
                    )
        t.stop()

        t.name = "duration_merge_seconds"
        t.start()
        merge_geotiff_rasters(args, parcel_ID)
        t.stop()

        # Append to infer_times.csv
        with open(args.times_file, encoding="utf-8", mode="a") as f:
            log_inference_times(parcel_ID, t, shp_records, f)
    # Compute coverage values from ALL predicted rasters and merge with additional metadata from shapefile
    df_inference, csv_path = make_parcel_predictions_csv(
        args.parcel_shapefile_path, args.stats_path
    )
    logger.info(f"Saved inference results to {csv_path}")


if __name__ == "__main__":
    main()
