"""
Modified code from https://github.com/nwojke/deep_sort
"""

import copy

import numpy as np


def _pdist_l2(a, b):
    """Compute pair-wise squared l2 distances between points in `a` and `b`."""
    try:
        a, b = np.asarray(a), np.asarray(b)
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))

        a = a[:, :, 0]
        b = b[:, :, 0]
        a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
        r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))
    except:
        import pdb;
        pdb.set_trace()
    return r2


def _pdist_mean(a, b):
    """Compute pair-wise squared l2 distances between points in `a` and `b`."""
    a = np.mean(a, axis=0, keepdims=True)
    a, b = np.asarray(a), np.asarray(b)

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)

    # Compute the dot product between a and b and their norms to get squared L2 distances
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]

    # Find the minimum distance for each point in a
    r2 = np.clip(r2, 0., float(np.inf))

    return r2


def mahalanobis_distance(mean_a, cov_a, mean_b, cov_b):
    cov_ab = 0.5 * (cov_a + cov_b)
    diff = mean_a - mean_b
    inv_cov_ab = np.linalg.inv(cov_ab)
    mahalanobis_dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov_ab), diff))
    return mahalanobis_dist


def _pdist(cfg, a, b, dims, phalp_tracker):
    a_appe, a_loca, a_pose, a_uv = [], [], [], []
    for i_ in range(len(a)):
        a_appe.append(a[i_][0])
        a_loca.append(a[i_][1])

    b_appe, b_loca, = b[0], b[1]
    a_appe, b_appe = np.asarray(a_appe), copy.deepcopy(np.asarray(b_appe))
    a_loca, b_loca = np.asarray(a_loca), copy.deepcopy(np.asarray(b_loca))

    if ('L' in cfg.distance_type):
        loc_distance = np.sqrt(_pdist_l2(a_loca, b_loca)) * 10
    if cfg.use_pred:
        c_x = a_loca[:, 3]
        c_y = a_loca[:, 4]
        c_z = a_loca[:, 5]
        c_xy = np.sqrt((c_x ** 2 + c_y ** 2 + c_z ** 2))
        c_xy = np.tile(c_xy, (1, len(b_appe)))

    r_texture = np.zeros((len(a_appe), len(b_appe)))

    if ('A' in cfg.distance_type):
        track_appe = a_appe / 10 if cfg.model != 'clip' else a_appe
        detect_appe = b_appe / 10 if cfg.model != 'clip' else b_appe
        r_texture = _pdist_mean(track_appe, detect_appe) if cfg.aggregation == 'mean' else _pdist_l2(track_appe,
                                                                                                     detect_appe)
    if (cfg.distance_type == "A0"): return cfg.beta_0 * r_texture
    if (cfg.distance_type == "L0"): return cfg.beta_1 * loc_distance

    betas = [cfg.beta_0, cfg.beta_1]

    xy_cxy_distance = loc_distance / betas[1]

    ruv2 = (1 + r_texture * betas[0]) * (1 / betas[1]) * np.exp(xy_cxy_distance)
    return ruv2



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _nn_euclidean_distance_min(cfg, x, y, dims, phalp_tracker):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray./
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """

    distances_a = _pdist(cfg, x, y, dims, phalp_tracker)
    return np.maximum(0.0, distances_a.min(axis=0))


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, cfg, matching_threshold, budget=None):

        self.cfg = cfg
        self._metric = _nn_euclidean_distance_min
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, appe_features, loca_features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for appe_feature, loca_feature, target in zip(appe_features, loca_features, targets):
            self.samples.setdefault(target, []).append([appe_feature, loca_feature])
            if self.budget is not None:
                if self.budget == 'first':
                    self.samples[target] = [self.samples[target][0]]
                else:
                    self.samples[target] = self.samples[target][-self.budget:]

        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, detection_features, targets, dims=None, phalp_tracker=None):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix_a = np.zeros((len(targets), len(detection_features[0])))
        for i, target in enumerate(targets):
            cost_matrix_a[i, :] = self._metric(self.cfg, self.samples[target], detection_features, dims,
                                               phalp_tracker)
        return cost_matrix_a
