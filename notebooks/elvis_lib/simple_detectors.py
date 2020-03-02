import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegressionCV

from ._base_detector import _BaseDetector


class RawActivationsDetector(_BaseDetector):
    def __init__(self, classifier_model, layer_names=None, n_components=20,
                 random_state=None):
        super(RawActivationsDetector, self).__init__(classifier_model,
                                                     layer_names=layer_names,
                                                     n_components=n_components,
                                                     random_state=random_state)

    def transform(self, X):
        """
        Get dimensionality-reduced intermediate activations of host classifier.
        """
        activations = self.get_activations(X)
        return np.concatenate(list(activations.values()), axis=1)

    def fit(self, X, y=None):
        """
        y: ndarray, optional (default None)
            If provided, we will, the detector will be a logistic-regression
            model (i.e supervised). Otherwise, the detector will be a one-class
            SVM (i.e unsupervided).
        """
        _BaseDetector.fit(self, X)
        codes = self.transform(X)

        if y is None:
            self.final = OneClassSVM()
        else:
            self.final = LogisticRegressionCV(random_state=self.random_state,
                                              cv=3)
        print("Fitting %s.." % self.final)
        self.final.fit(codes, y)
        return self

    def predict(self, X):
        codes = self.transform(X)
        return self.final.predict(codes)
