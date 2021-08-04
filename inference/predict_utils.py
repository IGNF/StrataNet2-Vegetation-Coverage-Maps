# Dependency imports
import os
import glob
import pickle
from math import cos, pi, ceil
import numpy as np
import torch
import numpy.ma as ma  # masked array
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import getsizeof

from osgeo import gdal
import rasterio
from rasterio.merge import merge
import rasterio.features
from rasterio.transform import Affine
from shapely import geometry
import shapefile
from shapely.geometry.point import Point
import torch.nn as nn

# We import from other files
from utils.utils import (
    create_dir,
    get_filename_no_extension,
    get_files_of_type_in_folder,
)
from utils.load_data import (
    get_plot_center,
    load_las_file,
    pre_transform,
)
from data_loader.loader import get_val_dataset
from data_loader.loader import rescale_cloud
from inference.geotiff_raster import (
    get_geotransform,
    add_weights_band_to_rasters,
    save_rasters_to_geotiff_file,
    FINAL_RASTER_BANDNAMES,
)
from model.project_to_2d import project_to_2d_rasters
from inference.prepare_utils import define_plot_id

sns.set()
np.random.seed(42)

import logging

logger = logging.getLogger(__name__)


def load_dataset(filename):
    """Load pickled dataset."""
    with open(filename, "rb") as pfile:
        dataset = pickle.load(pfile)
    return dataset


def filter_dataset(dataset, is_pseudo_labelling):
    """Filter dataset depending on task (inference or pseudo_labelling)"""
    if is_pseudo_labelling:
        MIN_POINTS_NB_FOR_PSEUDO_LABELLING = 2000
        return {
            plot_id: cloud_data
            for plot_id, cloud_data in dataset.items()
            if cloud_data["N_points_in_cloud"] > MIN_POINTS_NB_FOR_PSEUDO_LABELLING
        }
    return dataset


def create_dataloader(dataset, args):
    """Create a dataloader from dataset."""
    listdataset = get_val_dataset(dataset, args)
    dataloader = torch.utils.data.DataLoader(
        listdataset,
        batch_size=args.batch_size,
        num_workers=2,
    )
    return dataloader


def define_plot_geotiff_output_path(output_folder, parcel_id, plot_id, plot_center):
    """Define path name for intermediary plot raster during inference."""
    output_path = os.path.join(
        output_folder,
        f"{parcel_id}/{plot_id}.tif",
    )
    return output_path


def create_geotiff_raster(
    coverages_pointwise: torch.Tensor,  # (n_points, n_class)
    cloud: torch.Tensor,  # (n_features, n_points)
    plot_center,
    output_path,
    args,
):
    """ """
    rasters = project_to_2d_rasters(cloud, coverages_pointwise, args)
    rasters = add_weights_band_to_rasters(rasters, args)

    geo = get_geotransform(
        plot_center,
        args,
    )

    save_rasters_to_geotiff_file(
        output_path=output_path,
        width=args.diam_pix,
        height=args.diam_pix,
        data_array=rasters,
        geotransformation=geo,
    )


def get_parcel_info_and_predictions(tif, records):
    """From a prediction tif given by  its path and the records obtained from a shapefile,
    get the parcel metadata as well as the predictions : coverage and admissibility
    """
    mosaic = rasterio.open(tif).read()

    # Vb, Vmoy, Vh, Vmoy_hard
    band_means = np.nanmean(mosaic[:5], axis=(1, 2))

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
        "pred_veg_b": band_means[FINAL_RASTER_BANDNAMES.index("VegetationBasse")],
        "pred_veg_moy": band_means[
            FINAL_RASTER_BANDNAMES.index("VegetationIntermediaire")
        ],
        "pred_veg_h": band_means[FINAL_RASTER_BANDNAMES.index("VegetationHaute")],
        "admissibility": band_means[FINAL_RASTER_BANDNAMES.index("Admissibilite")],
    }
    metadata.update(infered_values)
    return metadata


def make_parcel_predictions_csv(parcel_shapefile_path, final_tiffs_folder):
    shp = shapefile.Reader(parcel_shapefile_path)
    records = {rec.ID: rec for rec in shp.records()}
    predictions_tif = get_files_of_type_in_folder(final_tiffs_folder, "tif")
    assert len(predictions_tif) < 10 ** 4

    infos = []
    for tif_filename in predictions_tif:
        info = get_parcel_info_and_predictions(tif_filename, records)
        infos.append(info)

    # export to a csv
    df_inference = pd.DataFrame(infos)
    csv_path = os.path.join(final_tiffs_folder, "PCC_inference_all_parcels.csv")
    df_inference.to_csv(csv_path, index=False)
    return df_inference, csv_path
