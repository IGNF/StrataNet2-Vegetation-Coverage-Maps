import pickle
from utils.utils import create_dir
import matplotlib.pyplot as plt
import numpy as np
import os

from KDEpy import FFTKDE
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

SUBSAMPLE_SIZE = 5 * 10 ** 5


def sample_z_from_dataset(dataset, subsample_size=SUBSAMPLE_SIZE):
    """Get sample of z values from plot dataset, as a 1D np.array"""
    all_z = [c_data["cloud"][2] for c_data in dataset.values()]
    all_z = np.concatenate(all_z)
    np.random.shuffle(all_z)
    return all_z[:subsample_size]


def get_fitted_kde_mixture_from_z_arr(z_arr, args):
    """From array of z values dataset, get a fitted KDE mixture"""
    logger.info(f"Fitting Mixture KDE on N={len(z_arr)} z values.")
    kde_mixture = KdeMixture(z_arr, args)
    return kde_mixture


def get_fitted_kde_mixture_from_dataset(dataset, args):
    """From plot dataset, get a fitted KDE mixture"""
    z_arr = sample_z_from_dataset(dataset)
    return get_fitted_kde_mixture_from_z_arr(z_arr, args)


class KdeMixture:
    def __init__(self, z, args):
        self.kde1 = None
        self.kde2 = None
        self.kde3 = None

        self.f1 = None
        self.f2 = None
        self.f3 = None

        self.fit(z)
        self.add_args_and_plot(args)

    def fit(self, z):
        """Fit three KDE for strata Vb, Vm and Vh, with strong priors abotu ranges of each strata given by weights."""
        z = self.get_sym_sorted_z(z)

        self.init_w1 = np.vectorize(lambda x: 1 if abs(x) < 0.5 else 0.05)(z)
        self.init_w2 = np.vectorize(lambda x: 1 if 0.5 < abs(x) < 1.5 else 0.05)(z)
        self.init_w3 = np.vectorize(
            lambda x: 1 if 1.5 < abs(x) else 0.5 if 0.5 < abs(x) else 0.05
        )(z)

        self.kde1 = FFTKDE(bw=0.1).fit(z, self.init_w1)
        self.kde2 = FFTKDE(bw=0.1).fit(z, self.init_w2)
        self.kde3 = FFTKDE(bw=0.1).fit(z, self.init_w3)

        X, y1, y2, y3 = self.evaluate_kdes()
        self.f1 = interp1d(X, y1, kind="linear", assume_sorted=False)
        self.f2 = interp1d(X, y2, kind="linear", assume_sorted=False)
        self.f3 = interp1d(X, y3, kind="linear", assume_sorted=False)
        logger.info("Fitted three KDEs and their corresponding linear interpolations.")

    def predict(self, z):
        """Get probabilities for strata Vb, Vm and Vh using learnt KDEs."""
        p1 = self.f1(z)
        p2 = self.f2(z)
        p3 = self.f3(z)
        return p1, p2, p3

    @staticmethod
    def get_sym_sorted_z(z):
        """Get a symétrical distribution around 0 to avoidborder effects. Sort to fit KDE."""
        z_sym = np.concatenate([-z, z])
        z_sym = np.sort(z_sym)
        return z_sym

    def add_args_and_plot(self, args):
        self.args = args
        self.plot_kde_mixture(3)
        self.plot_kde_mixture(25)

    def evaluate_kdes(self):
        X, y1 = self.kde1.evaluate(5 * 10 ** 3)
        y2 = self.kde2.evaluate(X)
        y3 = self.kde3.evaluate(X)
        y1 = y1 * self.init_w1.sum()
        y2 = y2 * self.init_w2.sum()
        y3 = y3 * self.init_w3.sum()
        max_all = np.max([y1.max(), y2.max(), y3.max()])
        y1 = y1 / max_all
        y2 = y2 / max_all
        y3 = y3 / max_all
        return X, y1, y2, y3

    def plot_kde_mixture(self, x_lim):
        X, y1, y2, y3 = self.evaluate_kdes()
        fig, (ax) = plt.subplots(1, 1, figsize=np.array([3, 0.8]) * 5)
        ax.set_title(f"Mixture of KDE for vb, Vm, Vh")
        ax.plot(X, y1, label="Vb", color="green")
        ax.plot(X, y2, label="Vm", color="blue")
        ax.plot(X, y3, label="Vh", color="black")
        ax.set_xlim([0, x_lim])
        ax.set_ylim([0, 2])
        ax.legend()
        plt.tight_layout()
        savepath = os.path.join(
            self.args.stats_path, f"img/kde_mixture/kde_mixture_x_lim={x_lim}.png"
        )
        create_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=100)
        self.args.experiment.log_image(savepath)
