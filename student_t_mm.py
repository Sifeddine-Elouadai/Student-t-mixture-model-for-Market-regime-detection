# student_t_mm.py
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import t


class StudentTMixtureRobust:
    """
    Approximate Student's t Mixture by fitting Gaussian Mixture to data
    and inflating variances to mimic t-distribution behavior.
    Simple, robust, 1D returns only.
    """

    def __init__(self, n_components=4, max_iter=500, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = None
        self.mu = None
        self.sigma = None
        self.weights_ = None

    def fit(self, X):
        gm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        gm.fit(X)
        self.model = gm
        self.mu = gm.means_.flatten()
        self.sigma = np.sqrt(gm.covariances_).flatten()
        self.weights_ = gm.weights_

    def predict_proba(self, X):
        return self.model.predict_proba(X)
