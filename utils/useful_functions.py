import os
import time
from argparse import Namespace


def get_args_from_prev_config(args, experiment_id):
    """args is a Namespace created in config.py, and has a str argument args.use_prev_config"""
    prev_config_folder = get_subfolder_path(
        args.path, experiment_id
    )  # is unique AND exists
    prev_config_path = os.path.join(prev_config_folder, "stats.txt")
    with open(prev_config_path) as f:
        l = f.readline()
        old_dict = vars(eval(l)).copy()
    # Ignore System args
    args_to_copy = [
        "n_class",
        "input_feats",
        "subsample_size",
        "diam_meters",
        "diam_pix",
        "m",
        "norm_ground",
        "ent",
        "e",
        "adm",
        "nb_stratum",
        "ECM_ite_max",
        "NR_ite_max",
        "znorm_radius_in_meters",
        "z_max",
        "MLP_1",
        "MLP_2",
        "MLP_3",
        "drop",
        "soft",
        "folds",
        "wd",
        "lr",
        "step_size",
        "lr_decay",
        "n_epoch",
        "n_epoch_test",
        "batch_size",
    ]
    old_dict = {a: b for a, b in old_dict.items() if a in args_to_copy}
    # Namespace -> dict -> update -> Namespace
    new_dict = vars(args).copy()
    new_dict.update(old_dict)
    args = Namespace(**new_dict)

    return args


# Print stats to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, "a") as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()


# Path and lookup helper functions


def create_dir(dir_name):
    """Create a new folder if does not exists"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_new_experiment_folder(args, infer_mode=False, resume_last_job=False):

    # We write results to different folders depending on the chosen parameters
    results_path = os.path.join(args.path, f"experiments/")

    if infer_mode:
        results_path = os.path.join(results_path, f"inference/{args.mode}")
    else:
        results_path = os.path.join(results_path, f"learning/{args.mode}")
    args.results_path = results_path

    if resume_last_job:
        # Use last job
        prev_jobs = os.listdir(results_path)
        run_name = sorted(prev_jobs)[-1]
    else:
        # Define a new job by date and time
        start_time = time.time()
        print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
        run_name = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))

    stats_path = os.path.join(results_path, run_name) + "/"
    create_dir(stats_path)
    args.stats_path = stats_path
    print(f"Results folder: {stats_path} (with resume_last_job = {resume_last_job})")

    stats_file = os.path.join(stats_path, "stats.txt")
    args.stats_file = stats_file

    if infer_mode:
        times_file = os.path.join(stats_path, "infer_times.csv")
        args.times_file = times_file


def get_files_of_type_in_folder(folder_path, extension):
    """Get a list of all files in folder with particular extension.
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


def get_subfolder_path(path, subfolder_name):
    """Find the path to subfolder of path whose name is experiment_id."""
    subfolders = fast_scandir(path)
    prev_config_folder = [
        f
        for f in subfolders
        if (subfolder_name in f) and (f.split("/")[-1] == subfolder_name)
    ]
    return prev_config_folder[0]


def get_trained_model_path_from_experiment(path, experiment_id, use_full_model=True):
    path_experiments = os.path.join(path, "experiments/")
    experiment_folder = get_subfolder_path(path_experiments, experiment_id)
    models = get_files_of_type_in_folder(experiment_folder, ".pt")
    if use_full_model:
        model_path = [m for m in models if "full" in m][0]
    else:
        model_path = [m for m in models if "fold_1" in m][0]
    return model_path


def get_filename_no_extension(filename):
    "path/to/filename.extension -> filename"
    return os.path.splitext(filename.split("/")[-1])[-2]
