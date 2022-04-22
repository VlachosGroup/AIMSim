"""Operation for clustering molecules"""
import sklearn.exceptions
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids as SklearnExtraKMedoids


class Cluster:
    """
    Wrapper class for different clustering algorithms.
    Attributes:
        clustering_method (str):
            Label for the specific algorithm used.
            'kmedoids':
                for the K-Medoids algorithm [1]. This method is useful
                when the molecular descriptors are continuous / Euclidean
                since it relies on the existence of a sensible medoid.
            'complete_linkage', 'complete':
                Complete linkage agglomerative hierarchical clustering [2].
            'average_linkage', 'average':
                average linkage agglomerative hierarchical clustering [2].
            'single_linkage', 'single':
                single linkage agglomerative hierarchical clustering [2].
            'ward':
                for Ward's algorithm [2]. This method is useful for
                Euclidean descriptors.
        n_clusters (int):
            Number of clusters.
        model_ (sklearn.cluster.AgglomerativeClustering or sklearn_extra.cluster.KMedoids):
            The clustering estimator.
        labels_ (np.ndarray of shape (n_samples,)):
            cluster labels of the training set samples.

    Methods:
        fit(X): Fit the estimator.
        predict(X): Get prediction from the estimator.
        get_labels (): Get cluster labels of the training set samples.

    References:
        [1] Hastie, T., Tibshirani R. and Friedman J.,
            The Elements of statistical Learning: Data Mining, Inference,
            and Prediction, 2nd Ed., Springer Series in Statistics (2009).
        [2] Murtagh, F. and Contreras, P., Algorithms for hierarchical
            clustering: an overview. WIREs Data Mining Knowl Discov
            (2011). https://doi.org/10.1002/widm.53

    """
    def __init__(self, n_clusters, clustering_method, **kwargs):
        """
        Constructor for the Cluster class.
        Args:
            n_clusters (int): Number of clusters.
            clustering_method(str): Label for the specific algorithm used.
                Supported methods are:
                'kmedoids' for the K-Medoids algorithm [1]. This method is
                    useful when the molecular descriptors are continuous
                    / Euclidean since it relies on the existence of a
                    sensible medoid.
                'complete_linkage', 'complete' for complete linkage
                    agglomerative hierarchical clustering [2].
                'average_linkage', 'average' for average linkage agglomerative
                    hierarchical clustering [2].
                'single_linkage', 'single' for single linkage agglomerative
                    hierarchical clustering [2].
                'ward' for Ward's algorithm [2]. This method is useful for
                    Euclidean descriptors.
            kwargs (dict): Keyword arguments. These are passed to the
                estimators. Refer to the following documentation page for
                kmedoids: https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
                agglomerative hierarchical clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

        References:
        [1] Hastie, T., Tibshirani R. and Friedman J.,
            The Elements of statistical Learning: Data Mining, Inference,
            and Prediction, 2nd Ed., Springer Series in Statistics (2009).
        [2] Murtagh, F. and Contreras, P., Algorithms for hierarchical
            clustering: an overview. WIREs Data Mining Knowl Discov
            (2011). https://doi.org/10.1002/widm.53
        """
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        if self.clustering_method == "kmedoids":
            self.model_ = self._get_kmedoids_model_(**kwargs)
        elif clustering_method in ["complete_linkage", "complete"]:
            self.model_ = self._get_linkage_model(linkage_method="complete",
                                                  **kwargs)
        elif clustering_method in ["average", "average_linkage"]:
            self.model_ = self._get_linkage_model(linkage_method="average",
                                                  **kwargs)
        elif clustering_method in ["single", "single_linkage"]:
            self.model_ = self._get_linkage_model(linkage_method="single",
                                                  **kwargs)
        elif clustering_method == "ward":
            self.model_ = self._get_linkage_model(linkage_method="ward",
                                                  **kwargs)
        else:
            raise ValueError(f"{clustering_method} not implemented")

    def _get_kmedoids_model_(self, **kwargs):
        """
        Initialize a k-medoids model.

        Args:
        kwargs (dict): Keyword arguments. These are passed to the
                estimators. Refer to the following documentation page for
                kmedoids:
                [https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html]

        """
        _ = kwargs.pop('metric', None)
        return SklearnExtraKMedoids(
            n_clusters=self.n_clusters,
            metric="precomputed",
            **kwargs
        )

    def _get_linkage_model(self, linkage_method, **kwargs):
        _ = kwargs.pop('affinity', None)
        return AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            linkage=linkage_method,
            **kwargs
        )

    def fit(self, X):
        """Fit the estimator.
        Args:
            X (np.ndarray or list):  Distance matrix.
        """
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        return self

    def predict(self, X):
        """Get predictions from the estimator.
        Args:
            X (np.ndarray or list):  samples to predict on.
        Raises:
            sklearn.exceptions.NotFittedError if estimator is not fitted.
        """
        try:
            return self.model_.predict(X)
        except sklearn.exceptions.NotFittedError as e:
            raise e

    def get_labels(self):
        """
        Get cluster labels of the training set samples.
        Returns:
            np.ndarray of shape (n_samples,)): Returns self.labels_,
                cluster labels of the training set samples.
        """
        if not self._is_fitted():
            raise sklearn.exceptions.NotFittedError
        return self.labels_

    def __str__(self):
        """
        String representation of the class.
        Returns:
            (str): Returns the self.clustering_method attribute.
        """
        return self.clustering_method

    def _is_fitted(self):
        """
        Returns:
            (bool): True is self.labels_ exists.
        """
        return hasattr(self, "labels_")
