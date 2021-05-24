# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    var = np.var(a, axis = 0) # 计算矩阵每一列的方差
    w = 1./var
    exp_w = np.exp(w)
    softmax_w = exp_w / np.sum(exp_w)
    w_matrix = np.diag(softmax_w)
    return 1. - np.dot(np.dot(a, w_matrix),b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """

    # mean = np.mean(x,axis=0)
    # error = x - mean
    # error_norm = np.linalg.norm(error, axis=1)
    # error_max = max(error_norm)
    # w = error_norm/error_max
    # w_matrix = np.diag(w)
    # #
    # new_x = np.dot(w_matrix,x)

    distances = _cosine_distance(x, y)

    # new_distances = np.dot(w_matrix,distances)

    return distances.min(axis=0)


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

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.orignial_samples = {}
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
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
        for feature, target in zip(features, targets):
            self.orignial_samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.orignial_samples[target] = self.orignial_samples[target][-self.budget:]
        self.orignial_samples = {k: self.orignial_samples[k] for k in active_targets}


        ################# every 3 frames take mean #############
        # for k,v in self.orignial_samples.items():
        #     self.samples[k]=[]
        #     if len(v)<=3:
        #         self.samples[k].append(np.mean(v,axis=0))
        #     else :
        #         for i in range(len(v)-2):
        #             self.samples[k].append(np.mean(v[i:i+3],axis=0))
        #     if self.budget is not None:
        #         self.samples[k] = self.samples[k][-self.budget:]
        # self.samples = {k: self.samples[k] for k in active_targets}

        # ################# every 5 frames take mean #############
        # for k,v in self.orignial_samples.items():
        #     self.samples[k]=[]
        #     if len(v)<=5:
        #         self.samples[k].append(np.mean(v,axis=0))
        #     else :
        #         for i in range(len(v)-4):
        #             self.samples[k].append(np.mean(v[i:i+5],axis=0))
        #     if self.budget is not None:
        #         self.samples[k] = self.samples[k][-self.budget:]
        # self.samples = {k: self.samples[k] for k in active_targets}

        ################# every 3 frames take median #############
        # for k,v in self.orignial_samples.items():
        #     self.samples[k]=[]
        #     if len(v)<=3:
        #         self.samples[k].append(np.median(v,axis=0))
        #     else :
        #         for i in range(len(v)-2):
        #             self.samples[k].append(np.median(v[i:i+3],axis=0))
        #     if self.budget is not None:
        #         self.samples[k] = self.samples[k][-self.budget:]
        # self.samples = {k: self.samples[k] for k in active_targets}

        ################ every 5 frames take median #############
        # for k,v in self.orignial_samples.items():
        #     self.samples[k]=[]
        #     if len(v)<=5:
        #         self.samples[k].append(np.median(v,axis=0))
        #     else :
        #         for i in range(len(v)-4):
        #             self.samples[k].append(np.median(v[i:i+5],axis=0))
        #     if self.budget is not None:
        #         self.samples[k] = self.samples[k][-self.budget:]
        # self.samples = {k: self.samples[k] for k in active_targets}


        ######################################### weight #########################
        # for k,v in self.orignial_samples.items():
        #     v_copy=v[:]
        #     feature_mean = np.mean(v_copy,axis=0)
        #     for i in range(len(v_copy)):
        #         diff = v_copy[i] - feature_mean
        #         diff_norm = np.linalg.norm(diff)
        #         if diff_norm > 0.6:
        #             v_copy[i]=v_copy[i]*100000
        #     self.samples[k]=v_copy
        #     if self.budget is not None:
        #         self.samples[k] = self.samples[k][-self.budget:]
        # self.samples = {k: self.samples[k] for k in active_targets}





    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M=128.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.orignial_samples[target], features)
        return cost_matrix
