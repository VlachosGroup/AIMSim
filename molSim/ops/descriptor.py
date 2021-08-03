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
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from mordred import Calculator, descriptors

from ..exceptions import *


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
            raise NotInitializedError(
                "Descriptor value not generated. Use "
                "make_fingerprint() to initialize it."
            )
        if not hasattr(self, "numpy_"):
            self.numpy_ = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(self.rdkit_, self.numpy_)
        return self.numpy_

    def to_rdkit(self):
        if self.check_init() is False:
            raise NotInitializedError(
                "Descriptor value not generated. Use "
                "make_fingerprint() to initialize it."
            )
        if not hasattr(self, "rdkit_"):
            raise ValueError(
                "Attempting to convert arbitrary numpy array "
                "to rdkit bit vector is not supported"
            )
        return self.rdkit_

    def check_init(self):
        if hasattr(self, "numpy_") or hasattr(self, "rdkit_"):
            return True
        return False

    def _set_morgan_fingerprint(self, molecule_graph, radius, n_bits, **kwargs):
        """Set the descriptor to a morgan fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.
        radius: int
            Radius of fingerprint, 3 corresponds to diameter (ECFP)6.
        n_bits: int
            Number of bits to use if Morgan Fingerprint wanted as
            a bit vector. If set to None, Morgan fingerprint returned
            as count.

        """
        self.rdkit_ = AllChem.GetMorganFingerprintAsBitVect(
            molecule_graph, radius, nBits=n_bits
        )
        self.label_ = "morgan_fingerprint"
        self.params_ = {"radius": radius, "n_bits": n_bits}

    def _set_rdkit_topological_fingerprint(
        self, molecule_graph, min_path, max_path, **kwargs
    ):
        """Set the descriptor to a topological fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.
        min_path: int
            Minimum path used to generate the topological fingerprint.
        max_path: int
            Maximum path used to generate the topological fingerprint.

        """
        if molecule_graph.GetNumAtoms() <= min_path:
            raise InvalidConfigurationError(
                f"# atoms in molecule: "
                f"{molecule_graph.GetNumAtoms()}, "
                f"min_path: {min_path}. "
                f"For topological fingerprint, "
                f"the number of atoms has to be "
                f"greater than the minimum path "
                f"used for fingerprint."
            )
        self.rdkit_ = rdmolops.RDKFingerprint(
            molecule_graph, minPath=min_path, maxPath=max_path
        )
        self.label_ = "topological_fingerprint"
        self.params_ = {"min_path": min_path, "max_path": max_path}

    def _set_mordred_descriptor(self, molecule_graph, descriptor, **kwargs):
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
                ensure correct name is used (https://mordred-descriptor.github.io/documentation/master/descriptors.html).""".format(
                    descriptor
                )
            )

    def _set_daylight_fingerprint(self, molecule_graph, **kwargs):
        """Set the descriptor to a daylight fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.

        """
        fp = FingerprintMols.FingerprintMol(molecule_graph)
        txt = fp.ToBinary()
        fp2 = DataStructs.ExplicitBitVect(txt)
        self.numpy_ = np.array(fp2.GetOnBits())
        self.label_ = "daylight_fingerprint"
        self.params_ = {}

    def _set_maccs_keys(self, molecule_graph, **kwargs):
        """Set the descriptor to MACCS keys.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.

        """
        self.rdkit_ = MACCSkeys.GenMACCSKeys(molecule_graph)
        self.label_ = "maccs_keys"
        self.params_ = {}

    def _set_atom_pair_fingerprint(self, molecule_graph, **kwargs):
        """Set the descriptor to an atom-pair fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.

        """
        self.rdkit_ = Pairs.GetAtomPairFingerprintAsBitVect(molecule_graph)
        self.label_ = "atom-pair_fingerprint"
        self.params_ = {}

    def _set_torsion_fingerprint(self, molecule_graph, **kwargs):
        """Set the descriptor to a torsion fingerprint.

        Parameters
        ----------
        molecule_graph: RDKIT object
            Graph of molecule to be fingerprinted.

        """
        self.rdkit_ = Torsions.GetTopologicalTorsionFingerprintAsIntVect(molecule_graph)
        self.label_ = "torsion_fingerprint"
        self.params_ = {}

    def make_fingerprint(
        self, molecule_graph, fingerprint_type, fingerprint_params=None
    ):
        """Make fingerprint of a molecule based on a graph representation.
        Set the state of the descriptor to this fingerprint.

        Args:
            molecule_graph (RDKIT object): The graph object used to make a
                fingerprint.
            fingerprint_type (str): label for the type of fingerprint.
                Invokes get_supported_descriptors()['fingerprints']
                for list of supported fingerprints.
            fingerprint_params (dict): Keyword arguments used to modify
                parameters of fingerprint. Default is None.
        """
        if fingerprint_params is None:
            fingerprint_params = {}
        if fingerprint_type == "morgan_fingerprint":
            morgan_params = {"radius": 3, "n_bits": 1024}
            morgan_params.update(fingerprint_params)
            self._set_morgan_fingerprint(molecule_graph=molecule_graph, **morgan_params)
        elif fingerprint_type == "topological_fingerprint":
            topological_params = {"min_path": 1, "max_path": 7}
            topological_params.update(fingerprint_params)
            self._set_rdkit_topological_fingerprint(
                molecule_graph=molecule_graph, **topological_params
            )
        elif fingerprint_type == "daylight_fingerprint":
            daylight_params = {}
            daylight_params.update(fingerprint_params)
            self._set_daylight_fingerprint(
                molecule_graph=molecule_graph, **daylight_params
            )
        elif fingerprint_type == "maccs_keys":
            maccs_params = {}
            maccs_params.update(fingerprint_params)
            self._set_maccs_keys(molecule_graph=molecule_graph, **maccs_params)
        elif fingerprint_type == "atom-pair_fingerprint":
            atom_pair_fp_params = {}
            atom_pair_fp_params.update(fingerprint_params)
            self._set_atom_pair_fingerprint(
                molecule_graph=molecule_graph, **atom_pair_fp_params
            )
        elif fingerprint_type == "torsion_fingerprint":
            torsion_params = {}
            torsion_params.update(fingerprint_params)
            self._set_torsion_fingerprint(
                molecule_graph=molecule_graph, **torsion_params
            )
        elif fingerprint_type.split(":")[0] == "mordred":
            mordred_params = {}
            self._set_mordred_descriptor(
                molecule_graph=molecule_graph,
                descriptor=fingerprint_type.split(":")[1],
                **mordred_params,
            )
        else:
            raise ValueError(f"{fingerprint_type} not supported")

    def set_manually(self, arbitrary_descriptor_val):
        """Set the descriptor value manually based on user specified value.

        Args:
            arbitrary_descriptor_val (np.ndarray or list): Vectorized
                representation of descriptor values.
        """
        self.label_ = "arbitrary"
        self.numpy_ = np.array(arbitrary_descriptor_val)

    def get_label(self):
        if not self.check_init():
            raise NotInitializedError
        else:
            return self.label_

    def get_params(self):
        if not self.check_init():
            raise NotInitializedError
        else:
            return self.params_

    def is_fingerprint(self):
        return "fingerprint" in self.get_label()

    @staticmethod
    def get_supported_fprints():
        """Return a list of strings for the currently implemented molecular fingerprints.
        Returns:
            List: List of strings.
        """
        return [
            "morgan_fingerprint",
            "topological_fingerprint",
            # TODO: These should be bit vectors of some kind
            # "daylight_fingerprint",
            # "maccs_keys",
            # "atom-pair_fingerprint",
            "torsion_fingerprint",
        ]
