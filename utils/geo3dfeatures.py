"""Geometry features from https://github.com/Oslandia/geo3dfeatures/blob/master/geo3dfeatures/features.py

List of functions which extracts 1D, 2D or 3D geometry features from point
clouds.

See:
- Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet,
2015. Semantic point cloud interpretation based on optimal neighborhoods,
relevant features and efficient classifiers. ISPRS Journal of
Photogrammetry and Remote Sensing, vol 105, pp 286-304.
- Brodu, N., Lague D., 2011. 3D Terrestrial lidar data classification of
complex natural scenes using a multi-scale dimensionality criterion:
applications in geomorphology. arXiV:1107.0550v3.


"""

import math

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial import cKDTree as KDTree

BIN_BUF = 1e-3  # Value that allows to consider max x/y values in bin building


def compute_tree(point_cloud, leaf_size):
    """Compute the KDTree structure
    Parameters
    ----------
    point_cloud : numpy.array
    leaf_size : int
    Returns
    -------
    sklearn.neighbors.KDTree (or scipy.spatial.KDTree)
    """
    return KDTree(point_cloud, leaf_size)


def request_tree(point, kd_tree, nb_neighbors=None, radius=None):
    """Extract a point neighborhood by the way of a KDTree method
    At least one parameter amongst 'nb_neighbors' and 'radius' much be
    valid. 'nb_neighbors' is considered first.
    Parameters
    ----------
    point : numpy.array
        Coordinates of the reference point (x, y, z)
    tree : scipy.spatial.KDTree
        Tree representation of the point cloud
    nb_neighbors : int
        Number of neighboring points to consider
    radius : float
        Radius that defines the neighboring ball around a given point
    Returns
    -------
    tuple
        Neighborhood, decomposed as a mean distance between the reference point
    and its neighbors and an array of neighbor indices within the point
    cloud. Get `nb_neighborhood + 1` in order to have the reference point and
    its k neighbors. If the neighborhood is recovered through radius, distances
    are not computed.
    """
    if nb_neighbors is not None:
        return kd_tree.query(point, k=nb_neighbors + 1)
    if radius is not None:
        return None, kd_tree.query_ball_point(point, r=radius)  # query_radius ?
    raise ValueError(
        "Kd-tree request error: nb_neighbors and radius can't be both None."
    )


def fit_pca(point_cloud):
    """Fit a PCA model on a 3D point cloud characterized by x-, y- and
    z-coordinates
    Parameters
    ----------
    point_cloud : numpy.array
        Array of (x, y, z) points
    Returns
    -------
    sklearn.decomposition.PCA
        Principle Component Analysis model fitted to the input data
    """
    return PCA().fit(point_cloud)


def max_normalize(a):
    """Compute and normalize values in "a". The normalized values are comprised
    between 0 and 1, 1 being the values of the larger value.

    If there is only one value in a, return an array of 0 (otherwise the
    normalization will produce a NaN).

    Parameters
    ----------
    a : numpy.array

    Returns
    -------
    numpy.array
    """
    if len(np.unique(a)) == 1:
        return pd.Series(np.zeros_like(a))
    else:
        return (a - np.min(a)) / (np.max(a) - np.min(a))


def sum_normalize(a):
    """Compute and normalize values in "a". The normalized values are comprised
    between 0 and 1, and the sum of normalized values equals to 1.

    Parameters
    ----------
    a : numpy.array

    Returns
    -------
    numpy.array
    """
    return a / np.sum(a)


def triangle_variance_space(eigenvalues):
    """Compute barycentric coordinates of a point within the explained variance
    space, knowing the PCA eigenvalues

    See Brodu, N., Lague D., 2011. 3D Terrestrial lidar data classification of
    complex natural scenes using a multi-scale dimensionality criterion:
    applications in geomorphology. arXiV:1107.0550v3.

    Extract of C++ code by N. Brodu:

        // Use barycentric coordinates : a for 1D, b for 2D and c for 3D
        // Formula on wikipedia page for barycentric coordinates
        // using directly the triangle in %variance space, they simplify a lot
        //FloatType c = 1 - a - b; // they sum to 1
        a = svalues[0] - svalues[1];
        b = 2 * svalues[0] + 4 * svalues[1] - 2;

    See the wikipedia page
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system\
    #Conversion_between_barycentric_and_Cartesian_coordinates

    If you project the three normalized eigenvalues on a 2D plane, i.e. (λ1,
    λ2), you get the triangle with these following coordinates:

    (1/3, 1/3), (1/2, 1/2), (1, 0)

    Thus you can build the T matrix and get the barycentric coordinates with
    the T^{-1}(r - r_3) formula.

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    list
        First two barycentric coordinates in the variance space (triangle)
    """
    alpha = eigenvalues[0] - eigenvalues[1]
    beta = 2 * eigenvalues[0] + 4 * eigenvalues[1] - 2
    return [alpha, beta]


def curvature_change(eigenvalues):
    """Compute the curvature change in the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Curvature change
    """
    return eigenvalues[2]


def linearity(eigenvalues):
    """Compute the linearity of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Linearity
    """
    return (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]


def planarity(eigenvalues):
    """Compute the planarity of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Planarity
    """
    return (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]


def scattering(eigenvalues):
    """Compute the scattering of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Scattering
    """
    return eigenvalues[2] / eigenvalues[0]


def omnivariance(eigenvalues):
    """Compute the omnivariance of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Omnivariance
    """
    return np.prod(eigenvalues) ** (1 / 3)


def anisotropy(eigenvalues):
    """Compute the anisotropy of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Anisotropy
    """
    return (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]


def eigenentropy(eigenvalues):
    """Compute the eigenentropy

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Entropy of the dataset
    """
    nonnull_eig = eigenvalues[eigenvalues > 0]
    return -1 * np.sum(nonnull_eig * np.log(nonnull_eig))


def val_sum(a):
    """Compute the sum of items in "a"

    Parameters
    ----------
    a : numpy.array
        Data

    Returns
    -------
    float
        Sum of "a" items
    """
    return np.sum(a)


def eigenvalue_ratio_2D(eigenvalues):
    """Compute the 2D eigenvalue ratio

    Parameters
    ----------
    eigenvalues : numpy.array
        Array of eigenvalues; must be (1, 2)-shaped
    """
    return eigenvalues[1] / eigenvalues[0]


def val_range(a):
    """Compute the range between minimal and maximal values of "a"

    Parameters
    ----------
    a : numpy.array
        Data

    Returns
    -------
    float
        Range of "a"
    """
    return np.ptp(a)


def std_deviation(a):
    """Compute the standard deviation of values contained in "a"

    Parameters
    ----------
    a : numpy.array
        Data

    Returns
    -------
    float
        Standard deviation of "a"
    """
    return np.std(a)


def radius_3D(distances):
    """Compute the max distance between a point and its neighbors, assuming
    that "distances" contains the euclidian distances

    Parameters
    ----------
    distances : numpy.array

    Returns
    -------
        3D radius associated to the point of interest
    """
    return np.max(distances)


def radius_2D(point, neighbors_2D):
    """Compute the max distance between 'point' and its neighbors, measured as
    the squared euclidian distance between 'point' and the furthest neighbor

    Parameters
    ----------
    point : numpy.array
        (x, y) coordinates of the point of interest
    2D_neighbors : numpy.array
        Set of 2D neighboring points

    Returns
    -------
    float
        2D radius associated to "point"
    """
    return np.power(((point - neighbors_2D) ** 2).sum(axis=1), 0.5).max()


def density_3D(radius, nb_neighbors):
    """Compute the density in a 3D space, as the ratio between point quantity
    and 3D volume

    Parameters
    ----------
    radius : float
        Radius of the sphere of interest around the considered point
    nb_neighbors : int
        Number of points in the neighborhood

    Returns
    -------
    float
        3D point density in the considered volume
    """
    if radius == 0:
        return None
    return (nb_neighbors + 1) / ((4 / 3) * math.pi * radius ** 3)


def density_2D(radius, nb_neighbors):
    """Compute the density in a 2D space, as the ratio between point quantity
    and 2D area

    Parameters
    ----------
    radius : float
        Radius of the area of interest around the considered point
    nb_neighbors : int
        Number of points in the neighborhood

    Returns
    -------
    float
        2D point density in the considered area
    """
    if radius == 0:
        return None
    return (nb_neighbors + 1) / (math.pi * radius ** 2)


def verticality_coefficient(pca):
    """Verticality score aiming at evaluating how vertical a 3D-point cloud is,
    by considering its decomposition through a PCA.

    See:
    - Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet,
    2015. Semantic point cloud interpretation based on optimal neighborhoods,
    relevant features and efficient classifiers. ISPRS Journal of
    Photogrammetry and Remote Sensing, vol 105, pp 286-304.
    - Jerome Demantke, Bruno Vallet, Nicolas Paparotidis, 2012. Streamed
    Vertical Rectangle Detection in Terrestrial Laser Scans for Facade Database
    Productions. In ISPRS Annals of the Photogrammetry, Remote Sensing and
    Spatial Information Sciences, Volume 1-3, pp99-104.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Principle Component Analysis output; must have a `.components_`
    attribute that contains decomposition axes in 3D space

    Returns
    -------
    float
        Verticality coefficient, as defined in (Demantke et al, 2012) and
    (Weinmann et al, 2015)
    """
    decomposition_axes = pca.components_
    assert decomposition_axes.shape[1] == 3
    return 1 - abs(decomposition_axes[2, 2])
