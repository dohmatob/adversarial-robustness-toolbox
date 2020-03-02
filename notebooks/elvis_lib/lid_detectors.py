import numpy as np

from skelarn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope

from ._base_detector import _BaseDetector


class LIDDetector(_BaseDetector):
    def __init__(self, classifier_model, layer_names=None, n_neighbors=None,
                 contamination=0., n_components=None, random_state=None):
        super(LIDDetector, self).__init__(classifier_model=classifier_model,
                                          layer_names=layer_names,
                                          n_components=n_components,
                                          random_state=random_state)
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def _fit_knns(self, activations):
        self.knns_ = {}
        for layer_name, layer_activations in activations.items():
            if self.n_components is None:
                k = len(layer_activations)
            else:
                k = self.n_components
            knn = NearestNeighbors(n_neighbors=k, n_jobs=1,
                                   algorithm='ball_tree')
            print("Fitting %s for layer '%s'..." % (knn, layer_name))
            self.knns_[layer_name] = knn.fit(layer_activations)
        self.lids_ = self._compute_lids(activations, remove_zero_dist=True)
        return self

    def fit(self, X):
        _BaseDetector.fit(self, X)
        activations = self.get_activations(X)
        self._fit_knns(activations)

        # fit a multivariate Gaussian to the lids of all the layers
        lids = np.transpose(list(self.lids_.values()))
        ee = EllipticEnvelope(contamination=self.contamination,
                              random_state=self.random_state)
        print("Fitting %s to LIDs..." % ee)
        ee.fit(lids)
        self.confidence_ellipsoid_ = ee
        return self

    def _check_is_fitted(self):
        return check_is_fitted(self, attributes=["knns_", "lids_"])

    def _compute_lids(self, activations, remove_zero_dist=False):
        """
        Compute the Local Intrinsic Dimension (LID) of each sample in a batch
        """
        lids = {}
        for layer_name, layer_activations in activations.items():
            print("Computing LIDs for layer '%s'..." % layer_name)
            dist, _ = self.knns_[layer_name].kneighbors(layer_activations)

            # handle zeros
            if remove_zero_dist:
                zero_mask = dist == 0.
                if zero_mask.sum():
                    dist[zero_mask] = dist[~zero_mask].min()

            lids[layer_name] = np.reciprocal(
                np.log(dist[:, -1][:, None] / dist).mean(axis=-1))
        return lids

    def predict(self, X):
        self._check_is_fitted()
        activations = self.get_activations(X)
        lids = np.transpose(list(self._compute_lids(activations).values()))
        return self.confidence_ellipsoid_.predict(lids)


def intrinsic_dim_sample_wise(X, k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation

    Input parameters:
    X    - data
    k    - number of nearest neighbours (Default = 5)
    dist - matrix of distances to the k nearest neighbors of each point
    (Optional)

    Returns:
    dimensionality estimation for the k
    """
    if dist is None:
        knn = NearestNeighbors(n_neighbors=k+1, n_jobs=1,
                               algorithm='ball_tree').fit(X)
        dist, _ = knn.kneighbors(X)
    dist = dist[:, 1:(k+1)]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample


def intrinsic_dim_scale_interval(X, k1=10, k2=20, dist=None, knn=None):
    """
    Returns range of Levina-Bickel dimensionality estimation for
    k = k1..k2, k1 < k2

    Input parameters:
    X    - data
    k1   - minimal number of nearest neighbours (Default = 10)
    k2   - maximal number of nearest neighbours (Default = 20)

    Returns:
    list of Levina-Bickel dimensionality estimation for k = k1...k2
    """
    if dist is None:
        if knn is None:
            knn = NearestNeighbors(n_neighbors=k2+1, n_jobs=1,
                                   algorithm='ball_tree').fit(X)
        dist, _ = knn.kneighbors(X)

    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k, dist=dist)
        intdim_k.append(m)
    return np.asanyarray(intdim_k), knn


class LocalIntrinsicDimensionMLE(BaseEstimator):
    def __init__(self, k1=10, k2=20, preprocessor=None):
        super(LocalIntrinsicDimensionMLE, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.preprocessor = preprocessor

    def fit(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor(X)
        intrinsic_dims, self.knn_ = intrinsic_dim_scale_interval(
            X, k1=self.k1, k2=self.k2)
        self.intrinsic_dims_ = intrinsic_dims.mean(axis=0)
        return self

    def predict(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor(X)
        intrinsic_dims, _ = intrinsic_dim_scale_interval(X, knn=self.knn_,
                                                         k1=self.k1,
                                                         k2=self.k2)
        return intrinsic_dims.mean(axis=0)


class LIDDetectorBickel(LIDDetector):
    def __init__(self, classifier_model, layer_names=None, k1=10, k2=20,
                 contamination=0.):
        super(LIDDetectorBickel, self).__init__(classifier_model,
                                                layer_names=layer_names,
                                                contamination=contamination)
        self.k1 = k1
        self.k2 = k2

    def _check_is_fitted(self):
        return check_is_fitted(self, attributes=["liders_"])

    def _fit_knns(self, activations):
        self.liders_ = {}
        self.lids_ = {}
        for layer_name, layer_activations in activations.items():
            lider = LocalIntrinsicDimensionMLE(k1=self.k1, k2=self.k2)
            print("Fitting %s for layer %s..." % (lider, layer_name))
            lider.fit(layer_activations)
            self.liders_[layer_name] = lider
            self.lids_[layer_name] = lider.intrinsic_dims_
        return self

    def _compute_lids(self, activations, remove_zero_dist=False):
        lids = {}
        for layer_name, layer_activations in activations.items():
            print("Computing LIDs for layer %s..." % layer_name)
            lids[layer_name] = self.liders_[layer_name].predict(
                layer_activations)
        return lids


class LocalIntrinsicDimensionMLE2NN(LocalIntrinsicDimensionMLE):
    def __init__(self, confidence=.95, **kwargs):
        super(LocalIntrinsicDimensionMLE2NN, self).__init__(**kwargs)
        self.confidence = confidence

    def _compute_mus(self, X):
        dist, _ = self.knn_.kneighbors(X)
        assert dist.shape[1] == 3
        dist = dist[:, 1:]
        return dist[:, 1] / dist[:, 0]  # ratio of distance from 2nd nearest to
                                        # nearest point

    def _preprocess(self, X):
        if self.preprocessor is None:
            return X
        return self.preprocessor(X)

    def _check_is_fitted(self):
        return check_is_fitted(self, attributes=["intrinsic_dim_"])

    def fit(self, X):
        X = self._preprocess(X)
        self.knn_ = NearestNeighbors(n_neighbors=2+1, n_jobs=1,
                                     algorithm='ball_tree').fit(X)
        self.mus_ = self._compute_mus(X)
        self.mus_.sort()
        self.Fmus_ = np.arange(1, len(X) + 1) / float(len(X))
        self.xx_ = np.log(self.mus_)
        self.yy_ = -np.log(1 - self.Fmus_)
        lr = LinearRegression(fit_intercept=False)
        self.intrinsic_dim_ = lr.fit(self.xx_[:-1][:, None],
                                     self.yy_[:-1]).coef_[0]
        return self

    def predict(self, X):
        raise NotImplementedError


class LIDDetector2NN(LIDDetector):
    def fit(self, X):
        X = self._get_activations(X)
        self.lid2nn_ = LocalIntrinsicDimensionMLE2NN().fit(X)
        self.intrinsic_dim_ = self.lid2nn_.intrinsic_dim_
        self.xx_ = self.lid2nn_.xx_
        self.yy_ = self.lid2nn_.yy_

        self.cutoff_ = 1.5 * self.intrinsic_dim_

        if False:
            self.knn_ = NearestNeighbors(n_neighbors=self.k, n_jobs=1,
                                         algorithm='ball_tree').fit(X)
        else:
            self.knn_ = self.lid2nn_.knn_
        return self
