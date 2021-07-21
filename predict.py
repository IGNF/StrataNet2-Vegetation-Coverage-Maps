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
from codetiming import Timer

warnings.simplefilter(action="ignore")
np.random.seed(42)
torch.cuda.empty_cache()

# Local imports
from config import args
from utils.utils import (
    create_new_experiment_folder,
    get_args_from_prev_config,
    get_filename_no_extension,
    print_stats,
    get_trained_model_path_from_experiment,
    create_a_logger,
    create_dir,
)
from model.point_cloud_classifier import PointCloudClassifier
from model.reproject_to_2d_and_predict_plot_coverage import project_to_2d
from inference.infer_utils import (
    divide_parcel_las_and_get_disk_centers,
    get_list_las_files_not_infered_yet,
    log_inference_times,
    make_parcel_predictions_csv,
    get_and_prepare_cloud_around_center,
    extract_points_within_disk,
)
from utils.load_data import (
    load_and_clean_single_las,
    transform_features_of_plot_cloud,
)
from data_loader.loader import cloud_loader_from_pickle

# from inference.geotiff_raster import create_geotiff_raster, merge_geotiff_rasters


@torch.no_grad()
def main():
    # SETUP
    create_new_experiment_folder(
        args, task="inference", resume_last_job=args.resume_last_job
    )
    logger = create_a_logger(args)
    # TODO: use previous args
    args.z_max = 24.24

    # MODEL
    args.trained_model_path = get_trained_model_path_from_experiment(
        args.path, args.inference_model_id, use_full_model=False
    )
    model = torch.load(args.trained_model_path)
    model.eval()
    PCC = PointCloudClassifier(args)

    # DATA
    args.unlabeled_dataset_pkl_path = (
        args.las_parcelles_folder_path[:-1] + "_pickled_unlabeled"
    )
    args.labeled_dataset_pkl_path = (
        args.las_parcelles_folder_path[:-1] + "_pickled_labeled"
    )
    create_dir(args.labeled_dataset_pkl_path)

    while True:

        unlabeled = glob.glob(os.path.join(args.unlabeled_dataset_pkl_path, "*"))
        labeled = glob.glob(os.path.join(args.labeled_dataset_pkl_path, "*"))
        unlabeled = [
            p
            for p in unlabeled
            if not any(
                get_filename_no_extension(p) in p_labeled for p_labeled in labeled
            )
        ]
        if not unlabeled:
            break
        else:
            p = unlabeled[0]
            parcel_ID = get_filename_no_extension(p)

        with open(p, "rb") as pfile:
            p_data = pickle.load(pfile)
            # for pseudolabeling, use only complete clouds
            p_data = {
                id: info
                for id, info in p_data.items()
                if info["N_points_in_cloud"] > 2000
            }

        dataset = tnt.dataset.ListDataset(
            [pp_id for pp_id in p_data.keys()],
            functools.partial(
                cloud_loader_from_pickle,
                dataset=p_data,
                args=args,
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        for batch_id, (
            clouds_batch,
            centers_batch,
            pseudoplot_ID_batch,
            _,
        ) in enumerate(
            tqdm(dataloader, desc=f"Inference on {parcel_ID}", total=len(dataloader))
        ):
            pred_pointwise, pred_pointwise_b = PCC.run(model, clouds_batch)

            # if infer_task == "selftrain":
            pred_pl, _, _ = project_to_2d(
                pred_pointwise, clouds_batch, pred_pointwise_b, PCC, args
            )
            for pp_ID, pred in zip(pseudoplot_ID_batch, pred_pl.cpu().detach().numpy()):
                p_data[pp_ID].update({"coverages": pred})

        output_pkl = p.replace("_pickled_unlabeled", "_pickled_labeled")
        with open(output_pkl, "wb") as pfile:
            pickle.dump(p_data, pfile)


if __name__ == "__main__":
    main()
