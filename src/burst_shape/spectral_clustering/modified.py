import warnings

import numpy as np
from sklearn.base import _fit_context
from sklearn.cluster import SpectralClustering
from sklearn.cluster._kmeans import k_means
from sklearn.cluster._spectral import cluster_qr, discretize
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state


class SpectralClusteringModified(SpectralClustering):
    def __init__(
        self,
        n_clusters=8,
        *,
        eigen_solver=None,
        n_components=None,
        n_components_max=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol="auto",
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            n_components=n_components,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.n_components_max = n_components_max
        if n_components_max is None:
            self.n_components_max = self.n_components
            raise UserWarning(
                "The point of using this class is to set n_components_max"
            )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # construct affinity matrix
        self.compute_maps(X)

        # compute labels
        self.compute_labels()

        return self

    def compute_maps(self, X):
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64,
            ensure_min_samples=2,
        )
        allow_squared = self.affinity in [
            "precomputed",
            "precomputed_nearest_neighbors",
        ]
        if X.shape[0] == X.shape[1] and not allow_squared:
            warnings.warn(
                "The spectral clustering API has changed. ``fit``"
                "now constructs an affinity matrix from data. To use"
                " a custom affinity matrix, "
                "set ``affinity=precomputed``."
            )

        if self.affinity == "nearest_neighbors":
            connectivity = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True, n_jobs=self.n_jobs
            )
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed_nearest_neighbors":
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric="precomputed"
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed":
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params["gamma"] = self.gamma
                params["degree"] = self.degree
                params["coef0"] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(
                X, metric=self.affinity, filter_params=True, **params
            )

        random_state = check_random_state(self.random_state)
        # We now obtain the real valued solution matrix to the
        # relaxed Ncut problem, solving the eigenvalue problem
        # L_sym x = lambda x  and recovering u = D^-1/2 x.
        # The first eigenvector is constant only for fully connected graphs
        # and should be kept for spectral clustering (drop_first = False)
        # See spectral_embedding documentation.
        self.maps_ = spectral_embedding(
            self.affinity_matrix_,
            n_components=self.n_components_max,
            eigen_solver=self.eigen_solver,
            random_state=random_state,
            eigen_tol=self.eigen_tol,
            drop_first=False,
        )
        return self

    def compute_labels(self):
        if self.verbose:
            print(f"Computing label assignment using {self.assign_labels}")

        random_state = check_random_state(self.random_state)
        self.labels_ = {}
        for n_cluster in self.n_clusters:
            self.labels_[n_cluster] = self._compute_labels(n_cluster)
        return self

    def _compute_labels(self, n_clusters):
        random_state = check_random_state(self.random_state)
        n_components = n_clusters if self.n_components is None else self.n_components
        maps = self.maps_[:, :n_components]
        if self.assign_labels == "kmeans":
            _, labels, _ = k_means(
                maps,
                n_clusters,
                random_state=random_state,
                n_init=self.n_init,
                verbose=self.verbose,
            )
        elif self.assign_labels == "cluster_qr":
            labels = cluster_qr(maps)
        else:
            labels = discretize(maps, random_state=random_state)

        return labels
