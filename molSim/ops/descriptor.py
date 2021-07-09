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
from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem import AllChem

from ..exceptions import NotInitializedError


class Descriptor:
    """Class for descriptors.

    Attributes
    ----------
    label_: str
        Label used to denote the type of descriptor being used.
    numpy_: np.ndarray
        Value of the descriptor in the numpy format.
    rdkit_: rdkit.DataStructs.cDataStructs.UIntSparseIntVec
        Value of the descriptor in the rdkit format.
        
    """
    def __init__(self, label=None, value=None):
        if value is None:
            self.label_ = label
            self.numpy_ = None
        else:
            self.label_ = 'arbitrary'
            self.numpy_ = np.array(value)
        self.rdkit_ = None

    def to_numpy(self):
        """Convert arbitrary fingerprints of type in_dtype to numpy arrays.

        Returns
        -------
        np.array

        """
        self._check_init()
        if self.numpy_ is None:
            self.numpy_ = np.zeros((0,), dtype=np.int8)
            self.numpy_ = DataStructs.ConvertToNumpyArray(self.rdkit_,
                                                          self.numpy_)
        return self.numpy_

    def to_rdkit(self):
        self._check_init()
        if self.rdkit_ is None:
            raise ValueError('Attempting to convert arbitrary numpy array '
                             'to rdkit bit vector is not supported')
        return self.rdkit_

    def _check_init(self):
        if self.numpy_ or self.rdkit_:
            return True
        raise NotInitializedError('Descriptor value not generated. '
                                  'Use make_fingerprint() to initialize it.')

    def _set_morgan_fingerprint(self,
                                molecule_graph,
                                radius=3,
                                n_bits=1024):
        """Set the descriptor to a morgan fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.
        radius: int
            Radius of fingerprint, 3 corresponds to diameter (ECFP)6.
            Default 3.
        n_bits: int
            Number of bits to use if Morgan Fingerprint wanted as
            a bit vector. If set to None, Morgan fingerprint returned
            as count. Default is 1024.

        """

        self.rdkit_ = AllChem.GetMorganFingerprintAsBitVect(molecule_graph,
                                                            radius,
                                                            nBits=n_bits)
        self.label_ = 'morgan_fingerprint'

    def _set_rdkit_topological_fingerprint(self,
                                           molecule_graph,
                                           min_path=1,
                                           max_path=7):
        """Set the descriptor to a topological fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.
        min_path: int
            Minimum path used to generate the topological fingerprint.
            Default is 1.
        max_path: int
            Maximum path used to generate the topological fingerprint.
            Default is 7.

        """
        self.rdkit_ = rdmolops.RDKFingerprint(molecule_graph,
                                              minPath=min_path,
                                              maxPath=max_path)
        self.label_ = 'topological_fingerprint'

    def make_fingerprint(self,
                         molecule_graph,
                         fingerprint_type,
                         **kwargs):
        """Make fingerprint of a molecule based on a graph representation.
        Set the state of the descriptor to this fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            The graph object used to make a fingerprint
        fingerprint_type: str
            label for the type of fingerprint.
            Invokes get_supported_descriptors()['fingerprints']
            for list of supported fingerprints.
        kwargs: dict
            Keyword arguments used to modify parameters of fingerprint.

        """

        if fingerprint_type == 'morgan_fingerprint':
            self._set_morgan_fingerprint(molecule_graph=molecule_graph,
                                         **kwargs)
        elif fingerprint_type == 'topological_fingerprint':
            self._set_rdkit_topological_fingerprint(
                                        molecule_graph=molecule_graph,
                                        **kwargs)
        else:
            raise ValueError(f'{fingerprint_type} not supported')

    def set_manually(self, arbitrary_descriptor_val):
        """
        Set the descriptor value manually based on user specified value.
        Parameters
        ----------
        arbitrary_descriptor_val: np.ndarray or list
            Vectorized representation of descriptor values.

        """
        self.label_ = 'arbitrary'
        self.numpy_ = np.array(arbitrary_descriptor_val)






