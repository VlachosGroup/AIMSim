"""This module contains methods to find similarities between .
Notes
-----
Input to all methods is an rdkit fingerprint or numpy array.

Supported descriptor datatypes
------------------------------
'numpy': np.array
'rdkit': rdkit.DataStructs.cDataStructs.ExplicitBitVect

"""
import numpy as np
from rdkit import DataStructs


def get_l_similarity(mol1_descriptor, mol2_descriptor, order):
    """Get L-similarity of arbitrary order.
    This is defined -(L-order norm).

    Parameters
    ---------
    mol1_descriptor: np.ndarray
        Descriptor Representation for molecule 1.
    mol2_descriptor: np.ndarray
        Descriptor Representation for molecule 2.
    order: int
        Order of the norm used to calculate similarity.
        E.g. order of 2 corresponds to L2 similarity etc.

    Returns
    ------
    float
        Similarity between the two molecules.

    Notes
    -----
    Ensure that all molecular descriptors are numpy.

    """
    return -np.linalg.norm(mol1_descriptor - mol2_descriptor, ord=order)


def get_tanimoto_similarity(mol1_descriptor, mol2_descriptor, descriptor_dtype):
    """Get tanimoto similarity between two molecular descriptors.

    Parameters
    ----------
    mol1_descriptor: np.ndarray
        Descriptor Representation for molecule 1.
    mol2_descriptor: np.ndarray
        Descriptor Representation for molecule 2.
    descriptor_dtype: str
        Label indicating data type for the fingerprint.
        Check file docstring for list of available types.

    Returns
    -------
    float
    """
    if descriptor_dtype == 'rdkit':
        return DataStructs.TanimotoSimilarity(mol1_descriptor, mol2_descriptor)
    elif descriptor_dtype == 'numpy':
        print('Tanimoto similarity is only useful for bitstrings.'
              'Consider using the rdkit bitstring data structure.'
              'Returning None')
        return None


def get_supported_measures():
    """Returns a list of labels for the similarity_measures
    that are supported currently

    Returns
    ------
    list(str)
        Supported similarity measures.
    """
    return ['tanimoto', 'neg_l0', 'neg_l1', 'neg_l2']

