# %%
import torch
import torch.nn as nn
import os
from utils.utils import get_trained_model_path_from_experiment

# /!\ Bias=False if followed by BatchNorm !


class PointNet(nn.Module):
    """
    The PointNet network for semantic segmentation
    """

    def __init__(self, MLP_1, MLP_2, MLP_3, args):
        """
        initialization function
        MLP_1, LMP2 and MLP3 = int array, size of the layers of multi-layer perceptrons
        for example MLP1 = [32,64]
        n_class = int,  the number of class
        input_feat = int, number of input feature
        subsample_size = int, number of points to which the tiles are subsampled

        """

        super(
            PointNet, self
        ).__init__()  # necessary for all classes extending the module class
        self.cuda_device = args.cuda
        self.subsample_size = args.subsample_size
        self.n_class = args.n_class
        self.drop = args.drop
        self.input_feat = args.n_input_feats
        try:
            self.set_patience_attributes(args)
        except AttributeError:
            pass

        # to create the correct number of layers
        m1 = MLP_1[-1] * 2  # size of the first embeding F1
        m2 = MLP_2[-1] * 2  # size of the second embeding F2

        # build MLP_1: input [input_feat x n] -> f1 [m1 x n]
        modules = []
        for i in range(len(MLP_1)):  # loop over the layer of MLP1
            modules.append(
                nn.Conv1d(
                    in_channels=MLP_1[i - 1] if i > 0 else self.input_feat,
                    out_channels=MLP_1[i],
                    kernel_size=1,
                    bias=False,
                )
            )
            modules.append(nn.BatchNorm1d(MLP_1[i]))
            modules.append(nn.ReLU(True))
        self.MLP_1 = nn.Sequential(*modules)

        # build MLP_2: f1 [m1 x n] -> f2 [m2 x n]
        modules = []
        for i in range(len(MLP_2)):
            modules.append(
                nn.Conv1d(
                    in_channels=MLP_2[i - 1] if i > 0 else m1,
                    out_channels=MLP_2[i],
                    kernel_size=1,
                    bias=False,
                )
            )
            modules.append(nn.BatchNorm1d(MLP_2[i]))
            modules.append(nn.ReLU(True))
        self.MLP_2 = nn.Sequential(*modules)

        # build MLP_3: f1 [(m1 + m2) x n] -> output [k x n]
        modules = []
        for i in range(len(MLP_3)):
            modules.append(
                nn.Conv1d(
                    in_channels=MLP_3[i - 1] if i > 0 else m2,
                    out_channels=MLP_3[i],
                    kernel_size=1,
                    bias=False,
                )
            )
            modules.append(nn.BatchNorm1d(MLP_3[i]))
            modules.append(nn.ReLU(True))

        # note: the last layer do not have normalization nor activation
        modules.append(nn.Dropout(p=self.drop))
        modules.append(nn.Conv1d(MLP_3[-1], self.n_class + 1, 1))
        self.MLP_3 = nn.Sequential(*modules)

        self.maxpool = nn.MaxPool1d(self.subsample_size)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.last_G_tensor = None

        if self.cuda_device is not None:
            self = self.cuda(self.cuda_device)

    def forward(self, input):
        """Get coverage scores and class probabilities from points cloud."""
        if self.cuda_device is not None:
            input = input.cuda(self.cuda_device)
        f1 = self.MLP_1(input)
        G1 = self.maxpool(f1)
        G1f2 = torch.cat((G1.repeat(1, 1, self.subsample_size), f1), 1)

        f2 = self.MLP_2(G1f2)
        G2 = self.maxpool(f2)
        self.last_G_tensor = G2
        G2f2 = torch.cat((G2.repeat(1, 1, self.subsample_size), f2), 1)

        scores_pointwise = self.MLP_3(G2f2)

        proba_pointwise, density_pointwise = torch.split(
            scores_pointwise, [4, 1], dim=1
        )
        proba_pointwise = self.softmax(proba_pointwise)
        density_pointwise = self.sigmoid(density_pointwise)

        coverages_pointwise = torch.mul(proba_pointwise, density_pointwise)
        return coverages_pointwise, proba_pointwise

    def set_patience_attributes(self, args):
        """Reset patience. Useful when we load a pretrained model."""
        self.stopped_early = False
        self.best_metric_value = 10 ** 6
        self.best_metric_epoch = 1
        self.patience_in_epochs = args.patience_in_epochs

    def stop_early(self, val_metric, epoch, args):
        """Save best model state until now, based on a validation metric to minimize, if no improvement over n epochs."""
        if val_metric < self.best_metric_value:
            self.best_metric_value = val_metric
            self.best_metric_epoch = epoch
            self.save_state(args)
        else:
            if epoch < args.epoch_to_start_early_stop:
                return False
            if epoch >= self.best_metric_epoch + self.patience_in_epochs:
                self.stopped_early = True
                return True
        return False

    def save_state(self, args):
        """Save model state in stats_path."""
        checkpoint = {
            "best_metric_epoch": self.best_metric_epoch,
            "state_dict": self.state_dict(),
            "best_metric_value": self.best_metric_value,
        }

        crossvalidating = args.current_fold_id > 0
        save_path = os.path.join(
            args.stats_path,
            f"PCC_model_{'fold_n='+str(args.current_fold_id) if crossvalidating else 'full'}.pt",
        )
        torch.save(checkpoint, save_path)

    def load_state(self, save_path):
        """Load model state from a path."""
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["state_dict"])
        self.best_metric_epoch = checkpoint["best_metric_epoch"]
        self.best_metric_value = checkpoint["best_metric_value"]
        return self

    def load_best_state(self, args):
        """Load best model state from early stopping checkpoint. Does not load the optimizer state."""
        crossvalidating = args.current_fold_id > 0
        save_path = os.path.join(
            args.stats_path,
            f"PCC_model_{'fold_n='+str(args.current_fold_id) if crossvalidating else 'full'}.pt",
        )
        self.load_state(save_path)
        return self
