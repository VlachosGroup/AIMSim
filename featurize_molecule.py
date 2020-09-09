"""This module contains methods to featurize molecules.
Notes
-----
Input to all methods is a Molecule object.
Output of all methods is supplied with the method.

Supported output type
--------------------
'numpy': np.array
'rdkit': rdkit.DataStructs.cDataStructs.ExplicitBitVect
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem import AllChem

from chemical_datatypes import Molecule


def get_morgan_fingerprint(molecule, required_output_dtype,
                           radius=3, n_bits=1024):
    """Generate a morgan fingerprint.

    Parameters
    ----------
    molecule: Molecule
        Molecule object to be featurized.
    required_output_dtype: str
        Label indicating output type for the fingerprint.
        Check file docstring for list of available types.
    radius: int
        Radius of fingerprint, 3 corresponds to diameter (ECFP)6. Default 3.
    n_bits: int
        Number of bits to use if Morgan Fingerprint wanted as
        a bit vector. If set to None, Morgan fingerprint returned
        as count. Default is 1024.

    Returns
    -------
    np.array

    """

    fingerprint_bits = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius, nBits=n_bits)

    return _manage_output_datatype(fingerprint_bits,
                                   required_output_dtype=required_output_dtype)


def get_rdkit_topological_fingerprint(molecule, required_output_dtype,
                                      min_path=1, max_path=7):
    """Generate a topological fingerprint.

    Parameters
    ----------
    molecule: Molecule
        Molecule object to be featurized.
    required_output_dtype: str
        Label indicating output type for the fingerprint.
        Check file docstring for list of available types.
    min_path: int
        Minimum path used to generate the topological fingerprint.
        Default is 1.
    max_path: int
        Maximum path used to generate the topological fingerprint.
        Default is 7.

    Returns
    -------
    np.array
    """
    fingerprint_bits = rdmolops.RDKFingerprint(
        molecule, minPath=min_path, maxPath=max_path)

    return _manage_output_datatype(fingerprint_bits,
                                   required_output_dtype=required_output_dtype)


def _convert_to_numpy(fingerprint_object, in_dtype):
    """Convert arbitrary fingerprints of type in_dtype to numpy arrays.

    Parameters
    ----------
    fingerprint_object: Object
        Object of arbitrary fingerprint class.
    in_dtype: str
        Datatype of the fingerprint class.
        Current accepted formats:
        'rdkit': rdkit.DataStructs.cDataStructs.UIntSparseIntVec

    Returns
    -------
    np.array

    """
    if in_dtype == 'rdkit':
        np_fingerprint = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint_object,
                                        np_fingerprint)
        return np_fingerprint


def _manage_output_datatype(fingerprint, required_output_dtype):
    """
    Manage the output datatype of the fingerprint by ensuring that it is of
    the desired type.
    Parameters
    ----------
    fingerprint: Object
        Arbitrary
    required_output_dtype: str
        Required output datatype of the fingerprint class.
        Current accepted formats:
        'rdkit': rdkit.DataStructs.cDataStructs.UIntSparseIntVec

    Returns
    -------
    Object of required class.

    """
    implemented_classes = {
        'rdkit': DataStructs.cDataStructs.ExplicitBitVect,
        'numpy': np.ndarray
    }

    if required_output_dtype == 'rdkit':
        if isinstance(fingerprint,
                      implemented_classes['rdkit']):
            return fingerprint
        elif isinstance(fingerprint, implemented_classes['numpy']):
            pass # IMPLEMENT

    elif required_output_dtype == 'numpy':
        if isinstance(fingerprint,
                      implemented_classes['rdkit']):
            return _convert_to_numpy(fingerprint, in_dtype='rdkit')
        elif isinstance(fingerprint, implemented_classes['numpy']):
            return fingerprint
