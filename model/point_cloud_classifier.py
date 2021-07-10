import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# TODO: add an internal timer that will monitor the different operations :


class PointCloudClassifier:
    """
    The main point cloud classifier Class
    deal with subsampling the tiles to a fixed number of points
    and interpolating to the original clouds
    """

    def __init__(self, args):
        self.subsample_size = (
            args.subsample_size
        )  # number of points to subsample each point cloud in the batches
        self.n_input_feats = len(args.input_feats)
        self.n_class = args.n_class  # number of classes in the output
        self.is_cuda = args.cuda  # wether to use GPU acceleration

    def run(self, model, clouds):
        """
        INPUT:
        model = the neural network
        clouds = batch of point clouds [n_batch, n_feat, n_points_i]
        OUTPUT:
        pred = [sum_i n_points_i, n_class] float tensor : prediction for each element of the
             batch in a single tensor

        """

        # batch_size = len(clouds)
        # will contain the prediction for all clouds in the batch
        # prediction_batch = torch.zeros((self.n_class, 0))

        # batch_data contain all the clouds in the batch subsampled to self.subsample_size points
        # sampled_clouds = torch.Tensor(
        #     batch_size, self.n_input_feats, self.subsample_size
        # )
        # if self.is_cuda:
        #     # sampled_clouds = sampled_clouds.cuda()
        #     prediction_batch = prediction_batch.cuda()

        # build batches of the same size
        # for i_batch in range(batch_size):
        #     # load the elements in the batch one by one and subsample/ oversample them
        #     # to a size of self.subsample_size points

        #     cloud = clouds[i_batch][: int(self.n_input_feats), :]
        #     n_points = cloud.shape[1]  # number of points in the considered cloud
        #     if n_points > self.subsample_size:
        #         selected_points = np.random.choice(
        #             n_points, self.subsample_size, replace=False
        #         )
        #     else:
        #         selected_points = np.random.choice(
        #             n_points, self.subsample_size, replace=True
        #         )
        #     cloud = cloud[
        #         :, selected_points
        #     ]  # reduce the current cloud to the selected points

        #     sampled_clouds[
        #         i_batch, :, :
        #     ] = cloud.clone()  # place current sampled cloud in sampled_clouds

        point_prediction = model(clouds)  # classify the batch of sampled clouds
        # assert point_prediction.shape == torch.Size(
        #     [batch_size, self.n_class, self.subsample_size]
        # )

        # # interpolation to original point clouds
        # prediction_batches = []
        # for i_batch in range(batch_size):
        #     # get the original point clouds positions
        #     cloud = clouds[i_batch]
        #     # and the corresponding sampled batch (only xyz position)
        #     sampled_cloud = sampled_clouds[i_batch, :3, :]
        #     n_points = cloud.shape[1]
        #     knn = NearestNeighbors(1, algorithm="kd_tree").fit(
        #         sampled_cloud.cpu().permute(1, 0)
        #     )
        #     # select for each point in the original point cloud the closest point in sampled_cloud
        #     _, closest_point = knn.kneighbors(cloud[:3, :].permute(1, 0).cpu())
        #     closest_point = closest_point.squeeze()
        #     prediction_cloud = point_prediction[i_batch, :, closest_point]
        #     prediction_batch = torch.cat((prediction_batch, prediction_cloud), 1)
        #     prediction_batches.append(prediction_cloud)
        prediction_batches = [
            prediction_cloud for prediction_cloud in point_prediction
        ]  # [(c,subsample_size) x b]
        pointwise_prediction = torch.cat(
            [cloud.permute(1, 0) for cloud in prediction_batches]
        )  # (4, b*subsample_size)
        return pointwise_prediction, prediction_batches
