"""
Base API for detectors
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

import keras


class _BaseDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier_model, layer_names=None, n_components=None,
                 random_state=None):
        # super(_BaseDetector, self).__init__()
        self.classifier_model = classifier_model
        self.layer_names = layer_names
        self.n_components = n_components
        self.random_state = random_state
        self._check_params()

    def _check_params(self):
        if not isinstance(self.classifier_model, keras.models.Model):
            raise NotImplementedError("We only work with keras models!")
        if self.layer_names is None:
            raise ValueError(
                "`layer_names` is None. You need to provide a list of layer names")

    def _build_activation_model(self):
        print("Building activation model...")
        outputs = []
        self.layer_names_ = []
        for layer in self.classifier_model.layers:
            for layer_name in self.layer_names:
                if layer_name.lower() == layer.name.lower():
                    outputs.append(layer.output)
                    self.layer_names_.append(layer.name)
                    break
        self.activation_model_= keras.models.Model(
            input=self.classifier_model.input, output=outputs)
        return self

    def get_activations(self, X):
        activations = self.activation_model_.predict(X)
        if isinstance(activations, np.ndarray):
            # XXX fix keras wahala
            activations = [activations]
        assert len(activations) == len(self.layer_names_)
        shape = len(X), -1
        activations = dict((layer_name, layer_activations.reshape(shape))
                           for layer_name, layer_activations in zip(
                                   self.layer_names_, activations))
        return self._apply_pcas_maybe(activations)

    def _fit_pcas_maybe(self, activations):
        if self.n_components is not None:
            self.pcas_ = {}
            for layer_name, layer_activations in activations.items():
                pca = PCA(n_components=self.n_components,
                          random_state=self.random_state)
                print("Fitting %s to layer '%s'..." % (pca, layer_name))
                self.pcas_[layer_name] = pca.fit(layer_activations)
        return self

    def _apply_pcas_maybe(self, activations):
        if hasattr(self, "pcas_"):
            return dict(
                (layer_name,
                 self.pcas_[layer_name].transform(layer_activations))
                for layer_name, layer_activations in activations.items())
        else:
            return activations

    def fit(self, X):
        self._build_activation_model()
        activations = self.get_activations(X)
        self._fit_pcas_maybe(activations)
        return self

    def predict(self, X):
        raise NotImplementedError
