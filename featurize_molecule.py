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


def get_supported_datatypes():
    """Returns a list of labels for supported datatypes.

    Returns
    -------
    List(str)
        List of supported datatype labels.
    """
    return ['rdkit', 'numpy']


class Descriptor:
    """Class for descriptors.

    Attributes
    ----------
    label: str
        Label used to denote the type of descriptor being used.
    value: str
        Value of the descriptor.
    datatype: str
        String label denoting the datatype of the descriptor value.
    """
    def __init__(self, label=None, value=None, datatype=None):
        self.value = value
        # can only be initialized by numpy array arbitrary value
        self.label = 'arbitrary' if value is not None else label
        self.datatype = 'numpy' if value is not None else datatype

    def _convert_to_numpy(self):
        """Convert arbitrary fingerprints of type in_dtype to numpy arrays.

        Parameters
        ----------
        in_dtype: str
            Datatype of the fingerprint class.
            Current accepted formats:
            'rdkit': rdkit.DataStructs.cDataStructs.UIntSparseIntVec

        Returns
        -------
        np.array

        """
        if self.datatype == 'rdkit':
            np_fingerprint = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(self.value,
                                            np_fingerprint)
            self.value = np_fingerprint
        self.datatype = 'numpy'

    def _coerce_output_datatype(self, output_datatype):
        """
        Manage the output datatype of the fingerprint by ensuring that it is of
        the desired type.

        Parameters
        ----------
        output_datatype: str
            Required output datatype of the fingerprint class.
            Current accepted formats:
            'rdkit': rdkit.DataStructs.cDataStructs.UIntSparseIntVec

        """
        if output_datatype == self.datatype:
            return
        if output_datatype == 'numpy':
            self._convert_to_numpy()

    def _set_morgan_fingerprint(self,
                                molecule_graph,
                                output_datatype,
                                radius=3,
                                n_bits=1024):
        """Set the descriptor to a morgan fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.
        output_datatype: str
            Label indicating the required format for the fingerprint.
        radius: int
            Radius of fingerprint, 3 corresponds to diameter (ECFP)6.
            Default 3.
        n_bits: int
            Number of bits to use if Morgan Fingerprint wanted as
            a bit vector. If set to None, Morgan fingerprint returned
            as count. Default is 1024.

        """

        self.value = AllChem.GetMorganFingerprintAsBitVect(   # rdkit bitvector
            molecule_graph, radius, nBits=n_bits)
        self.label = 'morgan_fingerprint'
        self.datatype = 'rdkit'
        self._coerce_output_datatype(output_datatype=output_datatype)

    def _set_rdkit_topological_fingerprint(self,
                                           molecule_graph,
                                           output_datatype,
                                           min_path=1,
                                           max_path=7):
        """Set the descriptor to a topological fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.
        output_datatype: str
            Label indicating the required format for the fingerprint.
        min_path: int
            Minimum path used to generate the topological fingerprint.
            Default is 1.
        max_path: int
            Maximum path used to generate the topological fingerprint.
            Default is 7.

        """
        self.value = rdmolops.RDKFingerprint(molecule_graph,  # rdkit bitvector
                                             minPath=min_path,
                                             maxPath=max_path)
        self.label = 'topological_fingerprint'
        self.datatype = 'rdkit'
        self._coerce_output_datatype(output_datatype=output_datatype)

    def _get_supported_fingerprints(self):
        """Returns a list of labels for the molecular_fingerprints
        that are supported currently

        Returns
        -------
        List(str)
            List of labels for fingerprints currently supported.

        """
        return ['topological_fingerprint', 'morgan_fingerprint']

    @staticmethod
    def get_supported_descriptors():
        """Returns a list od descriptors supported by this class

        Returns
        -------
        List(str)
            List of labels for descriptors currently supported.
        """
        supported_descriptors = []
        supported_descriptors.extend(Descriptor().
                                     _get_supported_fingerprints())
        return supported_descriptors

    def make_fingerprint(self,
                         molecule_graph,
                         fingerprint_type,
                         fingerprint_datatype,
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
        fingerprint_datatype: str
            Type of fingerprint to take.
        kwargs: dict
            Keyword arguments used to modify parameters of fingerprint.

        """

        if fingerprint_type == 'morgan_fingerprint':
            self._set_morgan_fingerprint(molecule_graph=molecule_graph,
                                         output_datatype=fingerprint_datatype,
                                         **kwargs)
        elif fingerprint_type == 'topological':
            self._set_rdkit_topological_fingerprint(
                                        molecule_graph=molecule_graph,
                                        output_datatype=fingerprint_datatype,
                                        **kwargs)
        else:
            raise ValueError(f'{fingerprint_type} not supported')






