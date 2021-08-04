# Dependency imports
import os
from sys import getsizeof
import glob
from math import cos, pi, ceil
import numpy as np
import numpy.ma as ma  # masked array
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import rasterio
import rasterio.features
from rasterio.transform import Affine
from shapely import geometry
import shapefile
from shapely.geometry.point import Point

from model.project_to_2d import project_to_2d_rasters
from utils.utils import create_dir, get_files_of_type_in_folder

np.random.seed(42)

FINAL_RASTER_BANDNAMES = [
    "VegetationBasse",
    "VegetationIntermediaire",
    "VegetationHaute",
    "VegetationIntermediaireDiscretisee",
    "Admissibilite",
    "PonderationPredictions",
]


def get_geotransform(plot_center_xy, args):
    """
    Get geotransform from plot center and plot dims.
    Structure: top_left_x, pix_width_in_meters, _, top_left_y, pix_heighgt_in_meters (neg for north up picture)
    Geotransform reference : https://gdal.org/user/raster_data_model.html
    """

    return [
        plot_center_xy[0] - args.diam_meters // 2,  # xmin
        args.diam_meters / args.diam_pix,
        0,
        plot_center_xy[1] + args.diam_meters // 2,  # ymax
        0,
        -args.diam_meters / args.diam_pix,
        # negative b/c in geographic raster coordinates (0,0) is at top left
    ]


def save_rasters_to_geotiff_file(
    output_path, width, height, data_array, geotransformation
):
    """
    Create a tiff file from stacked rasters (and their weights during inference)
    Note: for training plots, the xy localization may be approximative since the geotransformation has
    its corner at -10, -10 of the *mean point* of the cloud.
    """
    # We set Lambert 93 projection
    nb_channels = len(data_array)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    proj = srs.ExportToWkt()

    driver_tiff = gdal.GetDriverByName("GTiff")
    create_dir(os.path.dirname(output_path))
    dst_ds = driver_tiff.Create(
        output_path, width, height, nb_channels, gdal.GDT_Float32
    )
    dst_ds.SetGeoTransform(geotransformation)
    dst_ds.SetProjection(proj)
    if nb_channels == 1:
        outband = dst_ds.GetRasterBand(1)
        outband.WriteArray(data_array)
        outband.SetNoDataValue(np.nan)
        outband = None
    else:
        for ch in range(nb_channels):
            outband = dst_ds.GetRasterBand(ch + 1)
            outband.WriteArray(data_array[ch])
            # nodataval is needed for the first band only
            if ch == 0:
                outband.SetNoDataValue(np.nan)
            outband = None
    # write to file
    dst_ds.FlushCache()
    dst_ds = None


def add_weights_band_to_rasters(img_to_write, args):
    """
    Add weights rasters to stacked score rasters (n_rasters, width, height).
    Weights are between 0 and 1 and have linear diminution with distance to center of plot.
    """

    nb_channels = len(img_to_write)
    x = (np.arange(-args.diam_pix // 2, args.diam_pix // 2, 1) + 0.5) / args.diam_pix
    y = (np.arange(-args.diam_pix // 2, args.diam_pix // 2, 1) + 0.5) / args.diam_pix
    xx, yy = np.meshgrid(x, y, sparse=True)
    r = np.sqrt(xx ** 2 + yy ** 2)
    image_weights = 1.5 - r  # 1.5 to avoid null weights
    image_weights[r > 0.5] = np.nan  # 0.5 = "half of the square"

    # add one weight canal for each score channel
    for _ in range(nb_channels):
        img_to_write = np.concatenate([img_to_write, [image_weights]], 0)
    return img_to_write


def insert_hard_med_veg_raster_band(mosaic):
    """
    We classify pixels into medium veg or non medium veg, creating a fourth canal
    We use a threshold for which coverage_hard is the closest to coverage_soft.
    In case of global diffuse medium vegetation, this can overestimate areas with innaccessible medium vegetation,
    but has little consequences since they would be scattered on the surface.
    Return shape : (nb_canals, 32, 32) where canals are (Vb, Vm_soft, Vh, Vm_hard).
    """

    image_med_veg = mosaic[1]
    mask = ma.masked_invalid(image_med_veg).mask

    target_coverage = np.nanmean(image_med_veg)
    lin = np.linspace(0, 1, 10001)
    delta = np.ones_like(lin)
    for idx, threshold in enumerate(lin):
        image_med_veg_hard = 1.0 * (image_med_veg > threshold)
        image_med_veg_hard[mask] = np.nan
        delta[idx] = abs(target_coverage - np.nanmean(image_med_veg_hard))
    threshold = lin[np.argmin(delta)]
    image_med_veg_hard = 1.0 * (image_med_veg > threshold)
    image_med_veg_hard[mask] = np.nan

    mosaic = np.insert(mosaic, 3, image_med_veg_hard, axis=0)

    return mosaic


def insert_admissibility_raster(src_mosaic):
    """
    Return first bands are now : (Vb, Vm_soft, Vh, Vm_hard, admissibility)
    Ref:
    - https://gis.stackexchange.com/a/131080/184486
    - https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.geometry_mask
    """
    # Get data
    mosaic = src_mosaic.copy()
    veg_b = mosaic[0]
    veg_moy_soft = mosaic[1]
    veg_moy_hard = mosaic[3]
    mask = np.isnan(veg_moy_hard)

    # Eliminate zones < 5 pixels.
    veg_moy_hard_sieve = rasterio.features.sieve(
        veg_moy_hard.astype(np.int16), 5, mask=mask
    )
    # Set hard veg outside of parcel to avoid border effects.
    veg_moy_hard_sieve[mask] = 1
    # Use min is to keep small patches of zeros surround by ones (but not ones surrounder by zero).
    veg_moy_hard_sieve = np.nanmin(
        [[veg_moy_hard], [veg_moy_hard_sieve]], axis=0
    ).squeeze()

    # Vectorize + negative buffer of 1.5m -for shapes with value == 1 (i.e. with medium vegetation)
    BUFFER_WIDTH_METERS = -1.5
    poly = [
        geometry.shape(polygon).buffer(BUFFER_WIDTH_METERS)
        for polygon, value in rasterio.features.shapes(veg_moy_hard_sieve, mask=None)
        if value == 1
    ]
    poly = [s for s in poly if not s.is_empty]

    # Create an inaccessibility array mask (i.e. True for pixels whose center is in shapes)
    identity_transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    inaccessibility_mask = rasterio.features.geometry_mask(
        poly, veg_moy_hard.shape, identity_transform, invert=True
    )

    # Gather results
    admissibility = np.max([[veg_b], [veg_moy_soft]], axis=0).squeeze()
    admissibility[inaccessibility_mask] = 0
    admissibility[mask] = np.nan

    mosaic = np.insert(mosaic, 4, admissibility, axis=0)

    return mosaic


def merge_geotiff_rasters(output_folder_path, intermediate_tiff_folder_path):
    """
    Create a weighted average form a folder of tif files with channels [C1, C2, ..., Cn, W1, W2, ..., Wn].
    Outputed tif has same nb of canals, with wreightd average from C1 to Cn and sum of weights on W1 to Wn.
    Returns a message to log.
    """
    tiff_filenames = get_files_of_type_in_folder(intermediate_tiff_folder_path, ".tif")
    src_files_to_mosaic = [rasterio.open(filename) for filename in tiff_filenames]
    if len(src_files_to_mosaic) == 0:
        return f"Nothing in {intermediate_tiff_folder_path}. Cannot merge."

    mosaic, out_trans = rasterio.merge.merge(
        src_files_to_mosaic, method=_weighted_average_of_rasters
    )
    mosaic = finalize_merged_raster(mosaic)

    # save
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": len(mosaic),
            "transform": out_trans,
        }
    )

    with rasterio.open(output_folder_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        for idx in range(len(FINAL_RASTER_BANDNAMES)):
            dest.set_band_description(1 + idx, FINAL_RASTER_BANDNAMES[idx])

    return f"Saved merged raster prediction to {output_folder_path}"


def finalize_merged_raster(mosaic):
    """
    From the mosaic containing predictions (Vb, Vm, Vh), we:
    - Keep only one weight layer instead of three initially
    - Insert hard (0/1) raster for medium vegetation.
    - Insert admissibility raster
    - Replace np.nan with 0 in pixels which have at least one predicted value (for one of the coverage)
    to have appropriate coverage calculations.
    """
    mosaic = mosaic[: (3 + 1)]  # 3 * pred + 1 weights
    mosaic = insert_hard_med_veg_raster_band(mosaic)

    no_predicted_value = np.nansum(np.isnan(mosaic[:3]), axis=0) == 3
    mosaic = np.nan_to_num(mosaic, nan=0.0, posinf=None, neginf=None)
    mosaic[:, no_predicted_value] = np.nan

    mosaic = insert_admissibility_raster(mosaic)

    return mosaic


def _weighted_average_of_rasters(
    old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None
):
    """
    This function is used in rasterio.merge directly.
    Input data is composed of rasters with C * 2 bands, where C is the number of score.
    A weighted sum is performed on both scores bands [0:C] and weights [C:] using weights.
    One then needs to divide scores by the values of weights.
    """

    nb_scores_channels = int(len(old_data) / 2)
    unweighted_weights_bands = np.zeros_like(old_data[:nb_scores_channels, :, :])
    for band_idx in range(nb_scores_channels):  # for each score band
        w_idx = nb_scores_channels + band_idx

        # scale the score with weights, ignoring nodata in scores
        old_data[band_idx] = (
            old_data[band_idx]
            * old_data[w_idx]
            * (1 - old_nodata[band_idx])  # contrib is zero if nodata
        )
        new_data[band_idx] = (
            new_data[band_idx]
            * new_data[w_idx]
            * (1 - new_nodata[band_idx])  # contrib is zero if nodata
        )

        # sum weights
        w_idx = nb_scores_channels + band_idx
        w1 = old_data[w_idx] * (1 - old_nodata[band_idx])
        w2 = new_data[w_idx] * (1 - new_nodata[band_idx])
        unweighted_weights_bands[band_idx] = np.nansum(
            np.concatenate([[w1], [w2]]), axis=0
        )
        both_nodata = old_nodata[band_idx] * new_nodata[band_idx]
        unweighted_weights_bands[band_idx][both_nodata] = np.nan

    # set back to NoDataValue just in case we modified values where we should not
    old_data[old_nodata] = np.nan
    new_data[new_nodata] = np.nan

    # we sum weighted scores, and weights. Set back to nan as nansum generates 0 if no data.
    both_nodata = old_nodata * new_nodata
    out_data = np.nansum([old_data, new_data], axis=0)
    out_data[both_nodata] = np.nan

    # we average scores, using unweighted weights
    out_data[:nb_scores_channels] = (
        out_data[:nb_scores_channels] / unweighted_weights_bands
    )
    # # we do not average weights !

    # we have to update the content of the input argument
    old_data[:] = out_data[:]
