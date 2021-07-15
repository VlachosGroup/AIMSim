import sklearn.exceptions
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids as SklearnExtraKMedoids


class Cluster:
    def __init__(self, n_clusters, clustering_method, **kwargs):
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        if self.clustering_method == 'kmedoids':
            self.model_ = self._get_kmedoids_model_(**kwargs)
        elif clustering_method == 'complete_linkage':
            self.model_ = self._get_linkage_model(linkage_method='complete',
                                                  **kwargs)
        else:
            raise ValueError('{clustering_method} not implemented')
    
    def _get_kmedoids_model_(self, **kwargs):
        max_iter = kwargs.get('max_iter', 300)
        return SklearnExtraKMedoids(n_clusters=self.n_clusters, 
                                    metric='precomputed', 
                                    max_iter=max_iter)

    def _get_linkage_model(self, linkage_method, **kwargs):
        return AgglomerativeClustering(n_clusters=self.n_clusters,
                                       affinity='precomputed',
                                       linkage=linkage_method,
                                       **kwargs)

    def fit(self, X):
        """
        Parameters
        ----------
        X: np.ndarray or list
            Distance matrix.

        """
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        return self

    def predict(self, X):
        try:
            return self.model_.predict(X)
        except sklearn.exceptions.NotFittedError as e:
            raise e

    def get_labels(self):
        if not self._is_fitted():
            raise sklearn.exceptions.NotFittedError
        return self.labels_

    def __str__(self):
        return self.clustering_method 
    
    def _is_fitted(self):
        return hasattr(self, 'labels_')
