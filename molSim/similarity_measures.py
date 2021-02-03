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

"""
import numpy as np
from rdkit import DataStructs


def get_l_similarity(mol1_descriptor, mol2_descriptor, order):
    """Get L-similarity of arbitrary order.
    This is defined -(L-order norm).

    Parameters
    ---------
    mol1_descriptor: Descriptor object
        Descriptor Representation for molecule 1.
    mol2_descriptor: Descriptor object
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
    All molecular descriptors need to be of datatype numpy.

    """
    if (mol1_descriptor.datatype == 'numpy'
            and mol2_descriptor.datatype == 'numpy'):
        return -np.linalg.norm(mol1_descriptor.value - mol2_descriptor.value,
                               ord=order)


def get_tanimoto_similarity(mol1_descriptor, mol2_descriptor):
    """Get tanimoto similarity between two molecular descriptors.

    Parameters
    ----------
    mol1_descriptor: Descriptor object
        Descriptor Representation for molecule 1.
    mol2_descriptor: Descriptor object
        Descriptor Representation for molecule 2.

    Returns
    -------
    float

    """
    if (mol1_descriptor.datatype == 'rdkit'
            and mol2_descriptor.datatype == 'rdkit'):
        return DataStructs.TanimotoSimilarity(mol1_descriptor.value,
                                              mol2_descriptor.value)
    elif (mol1_descriptor.datatype == 'numpy'
          or mol2_descriptor.datatype == 'numpy'):
        raise ValueError('Tanimoto similarity is only useful for bitstrings.'
                         'Consider using the rdkit bitstring data structure')


def get_dice_similarity(mol1_descriptor, mol2_descriptor):
    """Get dice similarity between two molecular descriptors.

    Parameters
    ----------
    mol1_descriptor: Descriptor object
        Descriptor Representation for molecule 1.
    mol2_descriptor: Descriptor object
        Descriptor Representation for molecule 2.

    Returns
    -------
    float

    """
    if (mol1_descriptor.datatype == 'rdkit'
            and mol2_descriptor.datatype == 'rdkit'):
        return DataStructs.DiceSimilarity(mol1_descriptor.value,
                                          mol2_descriptor.value)
    elif (mol1_descriptor.datatype == 'numpy'
          or mol2_descriptor.datatype == 'numpy'):
        raise ValueError('Dice similarity is only useful for bitstrings.'
                         'Consider using the rdkit bitstring data structure.')


def get_supported_measures():
    """Returns a list of labels for the similarity_measures
    that are supported currently

    Returns
    ------
    list(str)
        Supported similarity measures.
    """
    return ['tanimoto',
            'neg_l0',
            'neg_l1',
            'neg_l2',
            'dice',
            'neg_manhattan',
            'neg_hamming',
            'neg_euclidean']


