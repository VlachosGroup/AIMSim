"""This module contains methods to find similarities between molecules.
Notes
-----
Input to all methods is an rdkit fingerprint or numpy array.

Supported descriptor datatypes
------------------------------
'numpy': np.array
'rdkit': rdkit.DataStructs.cDataStructs.ExplicitBitVect

Supported Metrics
----------------
'tanimoto': Jaccard Coefficient/ Tanimoto Similarity
        0 (not similar at all) to 1 (identical)
'negative_l0': Negative L0 norm of |x1 - x2|
'negative_l1': Negative L1 norm of |x1 - x2|
'negative_l2': Negative L2 norm of |x1 - x2|
'dice': Dice similarity
'cosine': Cosine similarity

"""
import numpy as np
from rdkit import DataStructs
from scipy.spatial.distance import cosine as scipy_cosine


class SimilarityMeasure:
    def __init__(self, metric):
        if metric.lower() in ['negative_l0']:
            self.metric = 'negative_l0'
            self.type_ = 'continuous'
            self.to_distance = lambda x: -x

        elif metric.lower() in ['negative_l1', 'negative_manhattan']:
            self.metric = 'negative_l1'
            self.type_ = 'continuous'
            self.to_distance = lambda x: -x

        elif metric.lower() in ['negative_l2', 'negative_euclidean']:
            self.metric = 'negative_l2'
            self.type_ = 'continuous'
            self.to_distance = lambda x: -x

        elif metric.lower() in ['cosine']:
            self.metric = 'cosine'
            self.type_ = 'continuous'
            # angular distance
            self.to_distance = lambda x: np.arccos(x) / np.pi

        elif metric.lower() in ['dice']:
            self.metric = 'dice'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['jaccard', 'tanimoto']:
            self.metric = 'tanimoto'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        else:
            raise ValueError(f'Similarity metric: {metric} is not implemented')

    def __call__(self, mol1_descriptor, mol2_descriptor):
        """

        Parameters
        ----------
        mol1_descriptor: Descriptor object
        mol2_descriptor: Descriptor object

        Returns
        -------
        similarity_: float
            Similarity value

        """
        similarity_ = None
        if self.metric == 'negative_l0':
            similarity_ = -np.linalg.norm(mol1_descriptor.to_numpy()
                                          - mol2_descriptor.to_numpy(),
                                          ord=0)
        elif self.metric == 'negative_l1':
            similarity_ = -np.linalg.norm(mol1_descriptor.to_numpy()
                                          - mol2_descriptor.to_numpy(),
                                          ord=1)
        elif self.metric == 'negative_l2':
            similarity_ = -np.linalg.norm(mol1_descriptor.to_numpy()
                                          - mol2_descriptor.to_numpy(),
                                          ord=2)
        elif self.metric == 'dice':
            try:
                similarity_ = DataStructs.DiceSimilarity(
                    mol1_descriptor.to_rdkit(),
                    mol2_descriptor.to_rdkit())
            except ValueError as e:
                raise ValueError(
                    'Dice similarity is only useful for bit strings '
                    'generated from fingerprints. Consider using '
                    'other similarity measures for arbitrary vectors.')

        elif self.metric == 'tanimoto':
            try:
                similarity_ = DataStructs.TanimotoSimilarity(
                    mol1_descriptor.to_rdkit(),
                    mol2_descriptor.to_rdkit())
            except ValueError as e:
                raise ValueError(
                    'Tanimoto similarity is only useful for bit strings '
                    'generated from fingerprints. Consider using '
                    'other similarity measures for arbitrary vectors.')

        elif self.metric == 'cosine':
            if mol1_descriptor.rdkit_ and mol2_descriptor.rdkit_:
                similarity_ = DataStructs.CosineSimilarity(
                    mol1_descriptor.rdkit_,
                    mol2_descriptor.rdkit_)
            else:
                similarity_ = scipy_cosine(mol1_descriptor.to_numpy(),
                                           mol2_descriptor.to_numpy())

        return similarity_

    @staticmethod
    def get_supported_metrics():
        """Return a list of strings for the currently implemented similarity measures, aka metrics.

        Returns:
            List: List of strings.
        """
        return ['negative_l0', 'negative_l1', 'negative_manhattan',
                'negative_l2', 'negative_euclidean', 'dice', 'jaccard', 'tanimoto', 'cosine']
