import os, sys

repo_absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_absolute_path)

import shutil
from argparse import ArgumentParser
import shapefile
import pandas as pd
from tqdm import tqdm

from utils.useful_functions import (
    get_filename_no_extension,
    get_files_of_type_in_folder,
)

parser = ArgumentParser(description="model")  # Byte-compiled / optimized / DLL files
parser.add_argument(
    "--parcel_shapefile_path",
    default=None,
    type=str,
    help="Path to shapefile (with .shp extension)",
)
parser.add_argument(
    "--las_folder_path",
    default=None,
    type=str,
    help="Path to folder of las files to copy from.",
)
preparation_args, _ = parser.parse_known_args()


def get_dataframe_from_shapefile(shapefile_path):
    """
    From a .dbd of a shapefile, create a pd.DataFrame.
    Column NOM is the ID of the entities of the shapefile.
    """
    sf = shapefile.Reader(shapefile_path)
    records = {rec.ID: rec.as_dict() for rec in sf.records()}
    df = (
        pd.DataFrame(records).transpose().reset_index().rename(columns={"index": "NOM"})
    )
    return df


def gather_las_data_from_shapefile_selection(
    las_folder_path, shapefile_path, output_folder
):
    """
    Helper function to take the elements of a shapefile, find associated LAS data, and copy it
    to an output folder.
    """
    parcel_names_in_shapefile = get_dataframe_from_shapefile(
        preparation_args.parcel_shapefile_path
    ).NOM.values
    las_files = [f for f in get_files_of_type_in_folder(las_folder_path, ".las")]
    las_files_subset = [
        f
        for f in las_files
        if get_filename_no_extension(f) in parcel_names_in_shapefile
    ]
    for original_filename in tqdm(las_files_subset):
        copy_filename = os.path.join(
            output_folder, get_filename_no_extension(original_filename) + ".las"
        )
        if not os.path.isfile(copy_filename):
            shutil.copy(original_filename, copy_filename)


def main():
    # Turn the shapefile into a csv for easier exploration.
    df = get_dataframe_from_shapefile(preparation_args.parcel_shapefile_path)
    df.to_csv(
        preparation_args.parcel_shapefile_path.replace(".shp", ".csv"), index=False
    )
    # List entities in shapefile and move corresponding las files from their storage folder to the folder containing the shapefile
    gather_las_data_from_shapefile_selection(
        preparation_args.las_folder_path,
        preparation_args.parcel_shapefile_path,
        os.path.dirname(preparation_args.parcel_shapefile_path),
    )


if __name__ == "__main__":
    main()
