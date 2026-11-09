"""Principal component analysis of a table of numbers (rows: samples, columns: features), in numpy.

It gives what scikit-learn's PCA gave the app, under the same names and with the same signs, without loading
scikit-learn (about a second when a project opens):

    pca = PCA().fit(X)                  every component
    pca = PCA(n_components=2).fit(X)    the first two
    pca.explained_variance_             each component's variance (divided by samples - 1)
    pca.explained_variance_ratio_       each one's share of the total variance
    pca.components_                     the directions, one per row
    pca.transform(Y)                    Y in those directions (centered on X's mean)
    pca.n_features_in_                  X's number of columns

The sign of each direction is fixed as scikit-learn fixes it: its largest entry (in size) is positive.
Used for advice and pictures only (the recommended latent dimension, the PCA table, the 2D latent views),
never by the optimization.
"""
import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        largest = np.argmax(np.abs(vt), axis=1)
        vt = vt * np.sign(vt[np.arange(len(vt)), largest])[:, None]
        variance = s ** 2 / max(X.shape[0] - 1, 1)
        total = variance.sum()
        k = len(variance) if self.n_components is None else self.n_components
        self.components_ = vt[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / total if total > 0 else np.zeros(k)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
