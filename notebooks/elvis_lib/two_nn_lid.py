import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

from lid_detectors import LocalIntrinsicDimensionMLE, LIDDetector


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
