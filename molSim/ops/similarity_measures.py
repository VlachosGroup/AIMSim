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
'neg_l0': Negative L0 norm of |x1 - x2|
'neg_l1': Negative L1 norm of |x1 - x2|
'neg_l2': Negative L2 norm of |x1 - x2|
'dice': Dice similarity
'cosine': Cosine similarity

"""
import numpy as np
from rdkit import DataStructs
import scipy.spatial.distance.cosine as scipy_cosine


class SimilarityMeasure:
    def __init__(self, metric):
        if metric.lower() in ['negative_l0']:
            self.metric = 'negative_l0'
        elif metric.lower() in ['negative_l1', 'negative_manhattan']:
            self.metric = 'negative_l1'
        elif metric.lower() in ['negative_l2', 'negative_euclidean']:
            self.metric = 'negative_l2'
        elif metric.lower() in ['dice']:
            self.metric = 'dice'
        elif metric.lower() in ['jaccard', 'tanimoto']:
            self.metric = 'tanimoto'
        elif metric.lower() in ['cosine']:
            self.metric = 'cosine'
        else:
            raise NotImplementedError

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
        similarity_ = np.nan
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
            if (mol1_descriptor.datatype == 'numpy'
                    or mol2_descriptor.datatype == 'numpy'):
                raise ValueError(
                    'Dice similarity is only useful for bit strings.'
                    'Consider using the rdkit bitstring data structure.')
            else:
                similarity_ = DataStructs.DiceSimilarity(
                                              mol1_descriptor.to_rdkit(),
                                              mol2_descriptor.to_rdkit())
        elif self.metric == 'tanimoto':
            if (mol1_descriptor.datatype == 'numpy'
                    or mol2_descriptor.datatype == 'numpy'):
                raise ValueError(
                    'Tanimoto similarity is only useful for bit strings.'
                    'Consider using the rdkit bitstring data structure.')
            else:
                similarity_ = DataStructs.TanimotoSimilarity(
                                              mol1_descriptor.to_rdkit(),
                                              mol2_descriptor.to_rdkit())
        elif self.metric == 'cosine':
            if (mol1_descriptor.datatype == 'rdkit'
                    and mol2_descriptor.datatype == 'rdkit'):
                similarity_ = DataStructs.CosineSimilarity(
                                              mol1_descriptor.to_rdkit(),
                                              mol2_descriptor.to_rdkit())
            else:
                similarity_ = scipy_cosine(mol1_descriptor.to_numpy(),
                                           mol2_descriptor.to_numpy())

        return similarity_


