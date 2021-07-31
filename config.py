from argparse import ArgumentParser
import os
from utils.utils import get_args_from_prev_config

parser = ArgumentParser(description="mode")
parser.add_argument(
    "--mode",
    default="PROD",
    type=str,
    help="DEV or PROD mode - DEV is a quick debug mode",
)
mode = parser.parse_known_args()[0].mode
parser = ArgumentParser(description="model")  # Byte-compiled / optimized / DLL files

# fmt: off

# System Parameters
repo_absolute_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(repo_absolute_path, "data/")
parser.add_argument('--mode', default=mode, type=str, help="DEV or PROD mode - DEV is a quick debug mode")
parser.add_argument('--cuda', default=None, type=int, help="Whether we use cuda (0 or 1 to specify device) or not (None)")
parser.add_argument('--path', default=repo_absolute_path, type=str, help="Repo absolute path directory")
parser.add_argument('--data_path', default=data_path, type=str, help="Path to /repo_root/data/ folder.")

# Data
parser.add_argument('--las_plots_folder_path', default=os.path.join(data_path, "placettes_dataset/las_classes/"), type=str, help="Path to folder with plot las files.")
parser.add_argument('--gt_file_path', default=os.path.join(data_path, "placettes_dataset/placettes_metadata.csv"), type=str, help="Path to ground truth file. Put in dataset folder.")
parser.add_argument('--las_parcels_folder_path', default=os.path.join(data_path, "SubsetParcelle_v0/"), type=str, help="Path to folder with parcels las files (ends with a /).")
parser.add_argument('--parcel_shapefile_path', default=os.path.join(data_path, "SubsetParcelle_v0/Parcelle_jeutest_v0.shp"), type=str, help="Path to shapefile of parcels.")

# Experiment parameters
PLOT_NAME_TO_VISUALIZE_DURING_TRAINING = {"Releve_Lidar_F68", # Vm = 100% -> Vm vs Vb distinction
                                            "2021_POINT_OBS66", # Vb = 50%, well separated -> use of NIR?
                                            "2021_POINT_OBS7",  #Vm = 25%, Vh = 25% -> Vm vs. Vh point localization along z
                                            "POINT_OBS106", # Vb = 50%, Vh=90% -> Vb under vegetation
                                        }
parser.add_argument('--plot_name_to_visualize_during_training', default=PLOT_NAME_TO_VISUALIZE_DURING_TRAINING,  help="A few plot name to track during learning")
parser.add_argument('--plot_geotiff_file', default=False,  action="store_true", help="Set to False to output SVG article format and GeoTIFF at last epoch.")
parser.add_argument("--comet_name", default="", type=str, help="Add this tag to the XP, to indicate its goal")
parser.add_argument('--offline_experiment', default=False,  action="store_true", help="Use for an offline Comet exoperiment.")
parser.add_argument("--full_model_training", default=False,  action="store_true", help="Use to do a full training after cross-validation")
parser.add_argument('--disabled', default=False, action="store_true", help="Wether we disable Comet for this run.")

# SSL
parser.add_argument("--use_PT_model", default=False,  action="store_true", help="Set to True to load finetune model PT_model_id in main.py.")
parser.add_argument('--PT_model_id', default="2021-07-21_13h23m57s", type=str, help="Identifier of experiment to load saved model traine on pseudo-labels (e.g. yyyy-mm-dd_XhXmXs).")

# Inference parameters
parser.add_argument("--use_prev_config", default=None, type=str, help="Identifier of a previous run from which to copy parameters from (e.g. yyyy-mm-dd_XhXmXs).")
parser.add_argument('--inference_model_id', default="2021-07-16_14h13m48s", type=str, help="Identifier of experiment to load saved model with torch.load (e.g. yyyy-mm-dd_XhXmXs).")
parser.add_argument('--resume_last_job', default=False,  action="store_true", help="Use (1) or do not use (0) the folder of the last experiment.")


# Herafter are the args that are reused when use_prev_config is set to a previous experiment id.
# Model Parameters 
parser.add_argument('--n_class', default=4, type=int,
                    help="Size of the model output vector. In our case 4 - different vegetation coverage types")
FEATURE_NAMES = [
    "x",
    "y",
    "z_flat",
    "red",
    "green",
    "blue",
    "near_infrared",
    "intensity",
    "return_num",
    "num_returns",
]
parser.add_argument('--input_feats', default=FEATURE_NAMES, type=str, help="Point features that we keep. in this code, we keep them all. permuting those letters will break everything. To be modified")
parser.add_argument('--subsample_size', default=10000, type=int, help="Subsample cloud size")
parser.add_argument('--diam_meters', default=20, type=int, help="Diameters of the plots.")
parser.add_argument('--diam_pix', default=20, type=int, help="Size of the output stratum raster (its diameter in pixels)")
parser.add_argument('--m', default=0.10, type=float, help="Loss regularization. The weight of the negative loglikelihood loss in the total loss")
parser.add_argument('--e', default=0.2 / 5, type=float, help="Loss regularization for entropy of pointwise scores of coverage. The weight of the entropy loss in the total loss.")
parser.add_argument('--znorm_radius_in_meters', default=1.5, type=float, help='Radius for KNN normalization of altitude.')
parser.add_argument('--z_max', default=24.24, type=float, help="Max (normalized) altitude of points in plots, based on labeled plots.")

# Network Parameters
parser.add_argument('--MLP_1', default=[32, 64], type=list)
parser.add_argument('--MLP_2', default=[128, 256], type=list)
parser.add_argument('--MLP_3', default=[64, 32], type=list)
parser.add_argument('--drop', default=0.4, type=float, help="Probability value of the Dropout layer")
 
# Optimization Parameters
parser.add_argument('--folds', default=5, type=int, help="Number of folds for cross validation model training")
parser.add_argument('--wd', default=0.001, type=float, help="Weight decay for the optimizer")
parser.add_argument('--batch_size', default=20, type=int, help="Size of the training batch")

# fmt: on
args, _ = parser.parse_known_args()
args.n_input_feats = len(args.input_feats)
print(args.disabled)
if args.use_prev_config is not None:
    args = get_args_from_prev_config(args, args.use_prev_config)

print(f"Arguments imported in {mode} mode.")
