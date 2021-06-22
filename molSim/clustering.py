from abc import ABC, abstractmethod

import sklearn.exceptions
from sklearn_extra.cluster import KMedoids as SklearnExtraKMedoids


class Cluster(ABC):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model_ = None
        self.labels_ = None
        self.interia_ = None

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def get_cluster_labels(self):
        pass

    @abstractmethod
    def get_cluster_intertia(self):
        pass

    @abstractmethod
    def get_cluster_centers(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class KMedoids(Cluster):
    def __init__(self, n_clusters, max_iter=300):
        super().__init__(n_clusters)
        self.model_ = SklearnExtraKMedoids(n_clusters=n_clusters, 
                                           metric='precomputed', 
                                           max_iter=max_iter)
        
    def fit(self, X):
        """
        Parameters
        ----------
        X: np.ndarray or list
            Distance matrix.

        """
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.interia_ = self.model_.interia_
    
    def predict(self, X):
        try:
            return self.model_.predict(X)
        except sklearn.exceptions.NotFittedError as e:
            raise e
    
    def get_cluster_labels(self):
        if not self._is_fitted():
            raise sklearn.exceptions.NotFittedError
        return self.labels_
    
    def get_cluster_intertia(self):
        if not self._is_fitted():
            raise sklearn.exceptions.NotFittedError
        return self.interia_
    
    def _is_fitted(self):
        return bool(self.labels_)

    


    
