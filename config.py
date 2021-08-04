from argparse import ArgumentParser
import numpy as np
import os

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
parser.add_argument('--plots_pickled_dataset_path', default=os.path.join(data_path, f"placettes_dataset/prepared/plots_dataset_{mode}.pkl"), type=str, help="Path to folder with plot las files.")
parser.add_argument('--gt_file_path', default=os.path.join(data_path, "placettes_dataset/placettes_metadata.csv"), type=str, help="Path to ground truth file. Put in dataset folder.")
parser.add_argument('--las_parcels_folder_path', default=os.path.join(data_path, "parcelles_dataset_20m/"), type=str, help="Path to folder with subfolders /input containing .las files.")
parser.add_argument('--parcel_shapefile_path', default=os.path.join(data_path, "parcelles_dataset_20m/input/Parcellaire_2020_zone_expe_BOP_SPL_SPH_J6P_PPH_CAE_CEE_ADM.shp"), type=str, help="Path to shapefile of parcels.")

# Experiment parameters
PLOT_NAME_TO_VISUALIZE_DURING_TRAINING = {"Releve_Lidar_F68", # Vm = 100% -> Vm vs Vb distinction
                                            "2021_POINT_OBS66", # Vb = 50%, well separated -> use of NIR?
                                            "2021_POINT_OBS7",  #Vm = 25%, Vh = 25% -> Vm vs. Vh point localization along z
                                            "POINT_OBS106", # Vb = 50%, Vh=90% -> Vb under vegetation
                                        }
parser.add_argument('--plot_name_to_visualize_during_training', default=PLOT_NAME_TO_VISUALIZE_DURING_TRAINING,  help="A few plot name to track during learning")
parser.add_argument('--plot_geotiff_file', default=False,  action="store_true", help="Set to False to output SVG article format and GeoTIFF at last epoch.")
parser.add_argument("--log_embeddings", default=False, action="store_true", help="False to avoid logging embeddings")
parser.add_argument("--comet_name", default="", type=str, help="Add this tag to the XP, to indicate its goal")
parser.add_argument('--offline_experiment', default=False,  action="store_true", help="Use for an offline Comet exoperiment.")
parser.add_argument("--log_confusion_matrix_frequency", default=10 if not mode=="DEV" else 1, help="Frequency (in  epoch) to log confusion matrixes to comet.")
parser.add_argument('--disabled', default=False, action="store_true", help="Wether we disable Comet for this run.")
# Prediction mode
parser.add_argument('--PT_model_id', default="", type=str, help="Identifier of experiment to load saved model traine on pseudo-labels (e.g. yyyy-mm-dd_XhXmXs).")
parser.add_argument('--inference_model_id', default="", type=str, help="Identifier of experiment to load saved model with torch.load (e.g. yyyy-mm-dd_XhXmXs).")

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
parser.add_argument('--drop', default=0.4, type=float, help="Probability value of the Dropout layer")
parser.add_argument('--ratio1', default=0.25, type=float, help="Ratio of centroid of first PointNet2 layer")
parser.add_argument('--r1', default=np.sqrt(2.0), type=float, help="Radius of first PointNet2 layer")
parser.add_argument('--ratio2', default=0.25, type=float, help="Ratio of centroid of second PointNet2 layer")
parser.add_argument('--r2', default=np.sqrt(8.0), type=float, help="Radius of second PointNet2 layer")
 
# Optimization Parameters
parser.add_argument('--folds', default=5, type=int, help="Number of folds for cross validation model training")
parser.add_argument('--wd', default=0.001, type=float, help="Weight decay for the optimizer")
parser.add_argument('--batch_size', default=20, type=int, help="Size of the training batch")

# Training parameters
parser.add_argument('--n_epoch', default=200 if not mode=="DEV" else 2, type=int, help="Number of training epochs")
parser.add_argument('--n_epoch_test', default=5 if not mode=="DEV" else 1, type=int, help="We evaluate every -th epoch, and every epoch after epoch_to_start_early_stop")
parser.add_argument('--epoch_to_start_early_stop', default=45 if not mode=="DEV" else 1, type=int, help="Epoch from which to start early stopping process, after ups and down of training.")
parser.add_argument('--use_early_stopping', default=False, action="store_true", help="Wether we early stop model based on val data.")
parser.add_argument('--patience_in_epochs', default=30 if not mode=="DEV" else 1, type=int, help="Epoch to wait for improvement of MAE_loss before early stopping. Set to np.inf to disable ES.")
parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('--step_size', default=1, type=int,
                    help="After this number of steps we decrease learning rate. (Period of learning rate decay)")
parser.add_argument('--lr_decay', default=0.985, type=float,
                    help="We multiply learning rate by this value after certain number of steps (see --step_size). (Multiplicative factor of learning rate decay)")

# fmt: on
args, _ = parser.parse_known_args()
args.n_input_feats = len(args.input_feats)

print(f"MODE: {mode}")
