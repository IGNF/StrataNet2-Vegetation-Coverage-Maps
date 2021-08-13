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
    get_filename_from_plot_name,
)
from data_loader.loader import get_val_dataset
from data_loader.loader import rescale_cloud
from inference.geotiff_raster import (
    get_geotransform,
    add_weights_band_to_rasters,
    save_rasters_to_geotiff_file,
    FINAL_RASTER_BANDNAMES,
    SHP_FIELDS_NAME_DICT,
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


def get_shapefile_records_dict(shp):
    """Get the shapefile records as a ID:records dict."""
    return {rec.ID: rec for rec in shp.records()}


def get_parcel_predicted_values(tif_filename):
    """
    From a prediction tif_filename given, get the parcel-level averaged predicted coverage and admissibility.
    Returns mock predictions filled with -1 if tif_filename is None.
    """
    predictions = {}
    if tif_filename is not None:
        mosaic = rasterio.open(tif_filename).read()
        band_means = np.nanmean(mosaic[:5], axis=(1, 2))
        for shp_field, channel_name in SHP_FIELDS_NAME_DICT.items():
            predictions.update(
                {
                    shp_field: band_means[FINAL_RASTER_BANDNAMES.index(channel_name)],
                }
            )
    else:
        for shp_field, channel_name in SHP_FIELDS_NAME_DICT.items():
            predictions.update(
                {
                    shp_field: -1,
                }
            )
    return predictions


def update_shapefile_with_predictions(parcel_shapefile_path, output_folder):
    """Add average coverage and admissibility values form .tiff of prediction rasters to shaepfile."""
    tif_filenames = get_files_of_type_in_folder(output_folder, "tif")
    A_HUNDRED_THOUSAND = 10 ** 5

    shp = shapefile.Reader(parcel_shapefile_path)
    output_shp_path = os.path.join(
        output_folder, get_filename_no_extension(parcel_shapefile_path)
    )
    assert len(tif_filenames) < A_HUNDRED_THOUSAND
    if len(tif_filenames) == 0:
        logger.error(f"No prediction tif file found in {output_folder}")

    with shapefile.Writer(output_shp_path) as w:
        w.fields = shp.fields[1:]  # skip first deletion field
        shp_fields = SHP_FIELDS_NAME_DICT.keys()
        for shp_field in shp_fields:
            w.field(shp_field, "F", decimal=10)

        for shaperec in shp.iterShapeRecords():
            parcel_ID = shaperec.record.ID
            tif_filename = get_filename_from_plot_name(tif_filenames, parcel_ID, ".tif")
            predictions = get_parcel_predicted_values(tif_filename)
            for shp_field in shp_fields:
                shaperec.record.append(predictions[shp_field])
            w.record(*shaperec.record)
            w.shape(shaperec.shape)
            if tif_filename is not None:
                a = 1
