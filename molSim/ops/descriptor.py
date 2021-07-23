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
from mordred import Calculator, descriptors

from ..exceptions import NotInitializedError, MordredCalculatorError


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

    def __init__(self, value=None):
        if value is not None:
            self.set_manually(arbitrary_descriptor_val=value)

    def to_numpy(self):
        """Convert arbitrary fingerprints of type in_dtype to numpy arrays.

        Returns
        -------
        np.array

        """
        if self.check_init() is False:
            raise NotInitializedError('Descriptor value not generated. Use '
                                      'make_fingerprint() to initialize it.')
        if not hasattr(self, 'numpy_'):
            self.numpy_ = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(self.rdkit_, self.numpy_)
        return self.numpy_

    def to_rdkit(self):
        if self.check_init() is False:
            raise NotInitializedError('Descriptor value not generated. Use '
                                      'make_fingerprint() to initialize it.')
        if not hasattr(self, 'rdkit_'):
            raise ValueError('Attempting to convert arbitrary numpy array '
                             'to rdkit bit vector is not supported')
        return self.rdkit_

    def check_init(self):
        if hasattr(self, 'numpy_') or hasattr(self, 'rdkit_'):
            return True
        return False

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

    def _set_mordred_descriptor(self, molecule_graph, descriptor):
        """Set the value of numpy_ to the descriptor as indicated by descriptor.

        Args:
            molecule_graph (RDKit object): Graph of the molecule of interest.
            descriptor (string): Name of descriptor, as implemented in Mordred.

        Raises:
            MordredCalculatorError: If Morded is unable to calculate a value
                for the indicated descriptor, this exception will be raised.
        """
        try:
            calc = Calculator(descriptors, ignore_3D=False)
            res = calc(molecule_graph)
            res.drop_missing()
            self.numpy_ = np.array(res[descriptor])
            self.label_ = descriptor
        except KeyError:
            raise MordredCalculatorError(
                """Mordred descriptor calculator unable to calculate descriptor \"{}\",
                ensure correct name is used (https://mordred-descriptor.github.io/documentation/master/descriptors.html).""" .format(
                    descriptor))

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
        elif fingerprint_type.split(":")[0] == 'mordred':
            self._set_mordred_descriptor(
                molecule_graph=molecule_graph,
                descriptor=fingerprint_type.split(":")[1],
                ** kwargs)
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
    
    def get_label(self):
        if not self.check_init():
            raise NotInitializedError
        else:
            return self.label_
    
    def is_fingerprint(self):
        return 'fingerprint' in self.get_label()
