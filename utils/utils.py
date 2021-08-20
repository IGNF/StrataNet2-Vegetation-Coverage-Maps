import os
import glob
import sys
import time
from comet_ml import Experiment, OfflineExperiment
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)


def create_a_logger(args):
    file_handler = logging.FileHandler(args.stats_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[file_handler, stream_handler],
        format="%(asctime)s:%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger


def launch_comet_experiment(args):
    """Launch a new named Comet.ml experiment, logging parameters along the way."""
    if args.offline_experiment:
        experiment = OfflineExperiment(
            project_name="lidar_pac",
            offline_directory=os.path.join(args.path, "experiments/"),
            auto_log_co2=True,
        )
    else:
        experiment = Experiment(
            project_name="lidar_pac",
            auto_log_co2=True,
            disabled=args.disabled,
        )
    experiment.log_parameters(vars(args))
    if args.comet_name:
        experiment.add_tags([args.mode])
        experiment.set_name(args.comet_name)
    else:
        experiment.add_tag(args.mode)
    args.experiment = experiment
    return experiment


def setup_experiment_folder(args, task="learning"):

    results_path = os.path.join(args.path, f"experiments/")
    results_path = os.path.join(results_path, f"{task}/{args.mode}")
    args.results_path = results_path

    current_time = time.time()
    run_name = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))

    args.stats_path = os.path.join(results_path, run_name) + "/"
    create_dir(args.stats_path)
    logger.info(f"Results folder is {args.stats_path}")

    args.stats_file = os.path.join(args.stats_path, "stats.txt")


def update_namespace_with_another_namespace(args, args_local):
    """Update first Namespace with args of second Namespace. This creates a novel object."""

    args_dict = vars(args).copy()
    args_local_dict = vars(args_local).copy()

    args_dict.update(args_local_dict)
    updated_args = Namespace(**args_dict)

    return updated_args


def create_dir(dir_name):
    """Create a new folder if does not exists"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_files_of_type_in_folder(folder_path, extension):
    """Get a list of all files in folder, with a specific particular extension.
    Folder path is a string path, and extension is something like: '.las', '.tif', etc."""
    filenames = os.listdir(folder_path)
    filenames = [
        os.path.join(folder_path, l) for l in filenames if l.lower().endswith(extension)
    ]
    return filenames


def fast_scandir(dirname):
    """List all subfolders (abs paths). See https://stackoverflow.com/a/40347279/8086033"""
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


def get_subfolder_in_folder_by_name(path, subfolder_name):
    """Find the path to a subfolder in path, whose name matches subfolder_name."""
    subfolders = fast_scandir(path)
    subfolders = [
        f
        for f in subfolders
        if (subfolder_name in f) and (f.split("/")[-1] == subfolder_name)
    ]
    return subfolders[0]


def get_filename_no_extension(filename):
    """path/to/filename.extension -> filename"""
    basename = os.path.basename(filename)
    return os.path.splitext(basename)[0]


def get_unprocessed_files(input_datasets_folder, output_datasets_folder):
    """
    Get all filenames from input_datasets_folder that have no matching file (sufix-agnostic) in output_datasets_folder.
    Ignore files in subfolders.
    """
    unlabeled = get_all_files_in_folder(input_datasets_folder)
    labeled = get_all_files_in_folder(output_datasets_folder)

    unlabeled = [
        p
        for p in unlabeled
        if not any(get_filename_no_extension(p) in p_labeled for p_labeled in labeled)
    ]
    return unlabeled


def get_all_files_in_folder(folder):
    """Get filenames from folder, ignoring subfolders."""
    files = glob.glob(os.path.join(folder, "*"), recursive=False)
    files = [l for l in files if os.path.isfile(l)]
    return files


# TODO: find a more appropriate place
def get_trained_model_path_from_experiment(path, experiment_id):
    path_experiments = os.path.join(path, "experiments/")
    experiment_folder = get_subfolder_in_folder_by_name(path_experiments, experiment_id)
    models = get_files_of_type_in_folder(experiment_folder, ".pt")
    try:
        model_path = [m for m in models if "full" in m][0]
    except:
        model_path = [m for m in models if "fold_n=1" in m][0]
    return model_path


def format_float_as_percentage(value):
    """Format float value as a percentage string."""
    return f"{value:.0%}"
