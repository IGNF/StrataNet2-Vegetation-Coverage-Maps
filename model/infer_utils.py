# Dependency imports
import os
import glob
from math import cos, pi, ceil
import numpy.ma as ma  # masked array
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import getsizeof
import rasterio
from rasterio.merge import merge


# We import from other files
from utils.useful_functions import create_dir, get_filename_no_extension
from utils.create_final_images import *
from data_loader.loader import *
from model.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from utils.load_las_data import (
    load_and_clean_single_las,
    transform_features_of_plot_cloud,
)

sns.set()

np.random.seed(42)


def divide_parcel_las_and_get_disk_centers(
    args, las_filename, save_fig_of_division=True
):
    """
    Identify centers of plots whose squares cover at least partially every pixel of the parcel
    We consider the square included in a plot with r=10m. Formula for width of
    the square is  W = 2 * (cos(45°) * r) since max radius in square equals r as well.
    We add an overlap of s*0.625 i.e. a pixel in currently produced plots of size 32 pix = 10
    :param las_folder: path
    :param las_filenae: "004000715-5-18.las" like string
    :returns:
        centers_nparray: a nparray of centers coordinates
        points_nparray: a nparray of full cloud coordinates
    Note: outputs are not normalized
    """

    points_nparray, xy_centers = load_and_clean_single_las(las_filename)
    size_MB = getsizeof(round(getsizeof(points_nparray) / 1024 / 1024, 2))
    print(f"Size of LAS file is {size_MB}MB")

    x_las, y_las = points_nparray[:, 0], points_nparray[:, 1]

    # DEBUG
    # # subsample = False
    # if subsample:
    #     subsampling = 500
    #     subset = np.random.choice(points_nparray.shape[0],size=subsampling, replace=False)
    #     x_las = x_las[subset]
    #     y_las = y_las[subset]

    x_min = x_las.min()
    y_min = y_las.min()
    x_max = x_las.max()
    y_max = y_las.max()

    # Get or calculate dimensions of disk and max square in said disk
    plot_radius_meters = 10  # This is hardcoded, but should not change at any time.
    cos_of_45_degrees = cos(pi / 4)
    within_circle_square_width_meters = 2 * cos_of_45_degrees * plot_radius_meters
    plot_diameter_in_pixels = args.diam_pix  # 32 by default
    plot_diameter_in_meters = 2 * plot_radius_meters
    s = 1  # size of overlap in pixels
    square_xy_overlap = (
        s * plot_diameter_in_meters / plot_diameter_in_pixels
    )  # 0.625 by default
    movement_in_meters = within_circle_square_width_meters - square_xy_overlap

    print(
        f"Square dimensions are {within_circle_square_width_meters:.2f}m*{within_circle_square_width_meters:.2f}m"
        + f"but we move {movement_in_meters:.2f}m at a time to have {square_xy_overlap:.2f}m of overlap"
    )

    x_range_of_parcel_in_movements = ceil((x_max - x_min) / (movement_in_meters)) + 1
    y_range_of_parcel_in_movements = ceil((y_max - y_min) / (movement_in_meters)) + 1

    start_x = x_min + movement_in_meters / 4
    start_y = y_min + movement_in_meters / 4
    grid_pixel_xy_centers = [[start_x, start_y]]

    for i_dx in range(x_range_of_parcel_in_movements):
        current_x = start_x + i_dx * movement_in_meters  # move along x axis
        for i_dy in range(y_range_of_parcel_in_movements):
            current_y = start_y + i_dy * movement_in_meters  # move along y axis
            new_plot_center = [current_x, current_y]
            grid_pixel_xy_centers.append(new_plot_center)

    # visualization
    if save_fig_of_division:
        # we need to normalize coordinates points for easier visualization
        save_image_of_parcel_division_into_plots(
            args,
            las_filename,
            x_las,
            y_las,
            x_min,
            y_min,
            x_max,
            y_max,
            within_circle_square_width_meters,
            s,
            square_xy_overlap,
            grid_pixel_xy_centers,
        )

    return grid_pixel_xy_centers, points_nparray


def save_image_of_parcel_division_into_plots(
    args,
    las_filename,
    x_las,
    y_las,
    x_min,
    y_min,
    x_max,
    y_max,
    within_circle_square_width_meters,
    s,
    square_xy_overlap,
    grid_pixel_xy_centers,
):
    """
    Visualize and save to PNG file the division of a large parcel into many disk subplots.
    """
    las_id = get_filename_no_extension(las_filename)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_min_c = x_min - x_center
    x_max_c = x_max - x_center
    y_min_c = y_min - y_center
    y_max_c = y_max - y_center

    # xy to dataframe for visualization
    coordinates = np.array(np.stack([x_las - x_center, y_las - y_center], axis=1))
    coordinates = pd.DataFrame(data=coordinates)
    coordinates.columns = ["x_n", "y_n"]

    sampling_size_for_kde = (
        10000  # fixed size which could lead to poor kde in large parcels.
    )
    if len(coordinates) > sampling_size_for_kde:
        coordinates = coordinates.sample(n=sampling_size_for_kde, replace=False)

    # centers to dataframe for visualization
    centers = np.array(grid_pixel_xy_centers - np.array([x_center, y_center]))
    centers = pd.DataFrame(data=centers)
    centers.columns = ["x_n", "y_n"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"aspect": "equal"})
    ax.grid(False)
    ax.set_aspect("equal")  # Not working right now
    plt.xlim(x_min_c - 5, x_max_c + 5)
    plt.ylim(y_min_c - 5, y_max_c + 5)
    plt.ylabel("y_n", rotation=0)
    plt.title(
        f'Cutting in r=10m plots for parcel "{las_id}"'
        + f"\n Contained squares: W={within_circle_square_width_meters:.2f}m with overlap={square_xy_overlap:.2f}m (i.e. {s}pix)"
    )
    # plot kde of parcel
    fig.tight_layout()
    sns.kdeplot(
        data=coordinates,
        x="x_n",
        y="y_n",
        fill=True,
        alpha=0.5,
        color="g",
        clip=[[x_min_c, x_max_c], [y_min_c, y_max_c]],
    )  # thresh=0.2

    # plot disks and squares
    for _, (x, y) in centers.iterrows():
        a_circle = plt.Circle(
            (x, y), 10, fill=True, alpha=0.1, edgecolor="white", linewidth=1
        )
        ax.add_patch(a_circle)
        a_circle = plt.Circle((x, y), 10, fill=False, edgecolor="white", linewidth=0.3)
        ax.add_patch(a_circle)

    sns.scatterplot(data=centers, x="x_n", y="y_n", s=5)

    # plot boundaries of parcel
    plt.axhline(
        y=y_min_c,
        xmin=x_min_c,
        xmax=x_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    plt.axhline(
        y=y_max_c,
        xmin=x_min_c,
        xmax=x_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    plt.axvline(
        x=x_min_c,
        ymin=y_min_c,
        ymax=y_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    plt.axvline(
        x=x_max_c,
        ymin=y_min_c,
        ymax=y_max_c,
        color="black",
        alpha=0.6,
        linestyle="-",
    )
    # fig.show()

    cutting_plot_save_folder_path = os.path.join(args.stats_path, f"img/cuttings/")
    create_dir(cutting_plot_save_folder_path)
    cutting_plot_save_path = os.path.join(
        cutting_plot_save_folder_path, f"cut_{las_id}.png"
    )

    plt.savefig(cutting_plot_save_path, dpi=200)
    plt.clf()
    plt.close("all")


def extract_points_within_disk(points_nparray, center, radius=10):
    """From a (2, N) np.array with x, y as first features, extract points within radius
    from the center = (x_center, y_center)"""
    xy = points_nparray[:, :2]  # (N, 2)
    contained_points = points_nparray[
        ((xy - center) ** 2).sum(axis=1) <= (radius * radius)
    ]  # (N, f)

    return contained_points


def get_and_prepare_cloud_around_center(parcel_points_nparray, plot_center, args):
    plots_point_nparray = extract_points_within_disk(parcel_points_nparray, plot_center)

    if plots_point_nparray.shape[0] == 0:
        return None

    # TODO: for clarity: make operations on the same axes instead of transposing inbetween
    plots_point_nparray = transform_features_of_plot_cloud(
        plots_point_nparray, args.znorm_radius_in_meters
    )
    plots_point_nparray = plots_point_nparray.transpose()
    plots_point_nparray = rescale_cloud_data(plots_point_nparray, plot_center, args)

    # add a batch dim before trying out dataloader
    # TODO: remove this useless batch dim (or use a DataLoader...)
    plots_point_nparray = np.expand_dims(plots_point_nparray, axis=0)
    plot_points_tensor = torch.from_numpy(plots_point_nparray)
    return plot_points_tensor


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


def add_hard_med_veg_raster_band(mosaic):
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


def merge_geotiff_rasters(args, plot_name):
    """
    Create a weighted average form a folder of tif files with channels [C1, C2, ..., Cn, W1, W2, ..., Wn].
    Outputs has same nb of canals, with wreightd average from C1 to Cn and sum of weights on W1 to Wn.
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

    mosaic, out_trans = merge(src_files_to_mosaic, method=_weighted_average_of_rasters)

    # hard raster wera also averaged and need to be set to 0 or 1
    mosaic = finalize_merged_raster(mosaic)

    # save
    out_meta = src.meta.copy()
    print(out_meta)
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
    out_fp = os.path.join(tiff_folder_path, f"prediction_raster_parcel_{plot_name}.tif")
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)
    print(f"Saved merged raster prediction to {out_fp}")


def finalize_merged_raster(mosaic):
    """Create hard (0/1) raster from the averaged prediction of medium vegetation.
    We then set np.nan where there is no predicted value at all, even for weights, after
    filling the np.nan with 0 within the parcel to have appropriate coverage calculations...
    """
    mosaic = add_hard_med_veg_raster_band(mosaic)

    no_predicted_value = np.nansum(np.isnan(mosaic[:3]), axis=0) == 3
    mosaic = np.nan_to_num(mosaic, nan=0.0, posinf=None, neginf=None)
    mosaic[:, no_predicted_value] = np.nan

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
