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
from rasterio.merge import merge
import rasterio.features
from rasterio.transform import Affine
from shapely import geometry
import shapefile
from shapely.geometry.point import Point

from inference.infer_utils import (
    infer_and_project_on_rasters,
    stack_the_rasters_and_get_their_geotransformation,
)

from utils.utils import create_dir

np.random.seed(42)


def create_geotiff_raster(
    args,
    pred_pointwise,
    plot_points_tensor,  # (N_feats, N_points) cloud 2D tensor
    plot_center,
    plot_name,
):
    """ """
    # we do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
    image_low_veg, image_med_veg, image_high_veg = infer_and_project_on_rasters(
        plot_points_tensor, pred_pointwise, args
    )

    # We normalize back x,y values to get the geotransform that position the raster on a map
    img_to_write, geo = stack_the_rasters_and_get_their_geotransformation(
        plot_center,
        args,
        image_low_veg,
        image_med_veg,
        image_high_veg,
    )

    # add the weights band for each band
    img_to_write = add_weights_band_to_rasters(img_to_write, args)

    # define save paths
    tiff_folder_path = os.path.join(
        args.stats_path,
        f"img/rasters/{plot_name}/",
    )
    create_dir(tiff_folder_path)
    tiff_file_path = os.path.join(
        tiff_folder_path,
        f"predictions_{plot_name}_X{plot_center[0]:.0f}_Y{plot_center[1]:.0f}.tif",
    )

    nb_channels = len(img_to_write)
    save_rasters_to_geotiff_file(
        nb_channels=nb_channels,
        new_tiff_name=tiff_file_path,
        width=args.diam_pix,
        height=args.diam_pix,
        datatype=gdal.GDT_Float32,
        data_array=img_to_write,
        geotransformation=geo,
    )


def save_rasters_to_geotiff_file(
    nb_channels, new_tiff_name, width, height, datatype, data_array, geotransformation
):
    """
    Create a tiff file from stacked rasters (and their weights during inference)
    Note: for training plots, the xy localization may be approximative since the geotransformation has
    its corner at -10, -10 of the *mean point* of the cloud.
    """

    # We set Lambert 93 projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    proj = srs.ExportToWkt()
    # We create a datasource
    driver_tiff = gdal.GetDriverByName("GTiff")
    dst_ds = driver_tiff.Create(new_tiff_name, width, height, nb_channels, datatype)
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


def merge_geotiff_rasters(args, plot_name):
    """
    Create a weighted average form a folder of tif files with channels [C1, C2, ..., Cn, W1, W2, ..., Wn].
    Outputed tif has same nb of canals, with wreightd average from C1 to Cn and sum of weights on W1 to Wn.
    Returns a message to log.
    """
    tiff_folder_path = os.path.join(
        args.stats_path,
        f"img/rasters/{plot_name}/",
    )
    dem_fps = glob.glob(os.path.join(tiff_folder_path, "*tif"))
    src_files_to_mosaic = []
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    if src_files_to_mosaic:
        mosaic, out_trans = merge(
            src_files_to_mosaic, method=_weighted_average_of_rasters
        )
        # hard raster wera also averaged and need to be set to 0 or 1
        mosaic = finalize_merged_raster(mosaic)

        # save
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "count": len(mosaic),
                "transform": out_trans,
                #         "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs ",
            }
        )
        out_fp = os.path.join(
            tiff_folder_path, f"prediction_raster_parcel_{plot_name}.tif"
        )
        descriptions = [
            "VegetationBasse",
            "VegetationIntermediaire",
            "VegetationHaute",
            "VegetationIntermediaireDiscretisee",
            "Admissibilite",
            "PonderationPredictions",
        ]
        with rasterio.open(out_fp, "w", **out_meta) as dest:
            dest.write(mosaic)
            for idx in range(len(descriptions)):
                dest.set_band_description(1 + idx, descriptions[idx])
        return f"Saved merged raster prediction to {out_fp}"
    else:
        return f"No predictions for {plot_name}. Cannot merge."


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
