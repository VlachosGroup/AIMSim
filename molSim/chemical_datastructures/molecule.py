"""Abstraction of a molecule with relevant property manipulation methods."""
import os.path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from molSim.exceptions import *
from molSim.ops.descriptor import Descriptor


class Molecule:
    """Molecular object defined from RDKIT mol object."""

    def __init__(
        self,
        mol_graph=None,
        mol_text=None,
        mol_property_val=None,
        mol_descriptor_val=None,
        mol_src=None,
        mol_smiles=None,
    ):
        """Constructor

        Args:
            mol_graph (RDKIT mol object): Graph-level information of molecule.
                Implemented as an RDKIT mol object. Default is None.
            mol_text (str): Text identifier of the molecule. Default is None.
                Identifiers can be:
                ------------------
                1. Name of the molecule.
                2. SMILES string representing the molecule.
            mol_property_val (float): Some property associated with the molecule.
                This is typically the response being studied. E.g. Boiling point,
                Selectivity etc. Default is None.
            mol_descriptor_val (numpy ndarray): Decriptor value for the molecule.
                Must be numpy array or list. Default is None.
            mol_src (str):
                Source file or SMILES string to load molecule. Acceptable files are
                -> .pdb file
                -> .txt file with SMILE string in first column, first row and
                        (optionally) property in second column, first row.
                Default is None.
                If provided mol_graph is attempted to be loaded from it.
            mol_smiles (str): SMILES string for molecule. If provided, mol_graph
                is loaded from it. If mol_text not set in keyword argument,
                this string is used to set it.
        """
        self.mol_graph = mol_graph
        self.mol_text = mol_text
        self.mol_property_val = mol_property_val
        self.descriptor = (
            Descriptor()
            if mol_descriptor_val is None
            else Descriptor(value=np.array(mol_descriptor_val))
        )
        if mol_src is not None:
            try:
                self._set_molecule_from_file(mol_src)
            except LoadingError as e:
                raise e
        if mol_smiles is not None:
            try:
                self._set_molecule_from_smiles(mol_smiles)
            except LoadingError as e:
                raise e

    def _set_molecule_from_pdb(self, fpath):
        """Set the mol_graph attribute from a PDB file.
        If self.mol_text is not set, it is set to the smiles string.

        Args:
            fpath (str): Path of PDB file.

        Raises:
             LoadingError: If Molecule cannot be loaded from SMILES string.
        """
        try:
            self.mol_graph = Chem.MolFromPDBFile(fpath)
        except Exception:
            raise LoadingError(f'{fpath} could not be loaded')
        if self.mol_graph is None:
            raise LoadingError(f'{fpath} could not be loaded')
        if self.mol_text is None:
            self.mol_text = os.path.basename(fpath).split('.')[0]

    def _set_molecule_from_smiles(self, mol_smiles):
        """Set the mol_graph attribute from smiles string.
        If self.mol_text is not set, it is set to the smiles string.

        Args:
            mol_smiles (str): SMILES string for molecule. If provided,
                mol_graph is loaded from it. If mol_text not set in keyword
                argument, this string is used to set it.

        Raises:
             LoadingError: If Molecule cannot be loaded from SMILES string.
        """
        try:
            self.mol_graph = Chem.MolFromSmiles(mol_smiles)
        except Exception:
            raise LoadingError(f'{mol_smiles} could not be loaded')
        if self.mol_graph is None:
            raise LoadingError(f'{mol_smiles} could not be loaded')
        if self.mol_text is None:
            self.mol_text = mol_smiles

    def _set_molecule_from_file(self, mol_src):
        """Load molecule graph from file.

        Args:
            mol_src (str): Source file or SMILES string to load molecule.
                Acceptable files are
                -> .pdb file
                -> .txt file with SMILE string in first column, first row.

        Raises:
             LoadingError: If Molecule cannot be loaded from source.
        """
        if os.path.isfile(mol_src):
            mol_fname, extension = os.path.basename(mol_src).split(".")
            if extension == "pdb":
                try:
                    self._set_molecule_from_pdb(mol_src)
                except LoadingError as e:
                    raise e
            elif extension == "txt":
                with open(mol_src, "r") as fp:
                    mol_smiles = fp.readline().split()[0]
                try:
                    self._set_molecule_from_smiles(mol_smiles)
                except LoadingError as e:
                    raise e

    def set_descriptor(
        self,
        arbitrary_descriptor_val=None,
        fingerprint_type=None,
        fingerprint_params=None,
    ):
        """Sets molecular descriptor attribute.

        Args:
            arbitrary_descriptor_val (np.array or list): Arbitrary descriptor
                vector. Default is None.
            fingerprint_type (str): String label specifying which fingerprint
                to use. Default is None.
            fingerprint_params (dict): Additional parameters for modifying
                fingerprint defaults. Default is None.
        """
        if arbitrary_descriptor_val is not None:
            self.descriptor.set_manually(arbitrary_descriptor_val)
        elif fingerprint_type is not None:
            if self.mol_graph is None:
                raise ValueError(
                    "Molecular graph not present. "
                    "Fingerprint cannot be calculated."
                )
            self.descriptor.make_fingerprint(
                self.mol_graph,
                fingerprint_type=fingerprint_type,
                fingerprint_params=fingerprint_params,
            )
        else:
            raise ValueError(f"No descriptor vector were passed.")

    def get_descriptor_val(self):
        """Get value of molecule descriptor.

        Returns:
            np.ndarray: value(s) of the descriptor.

        """
        return self.descriptor.to_numpy()

    def match_fingerprint_from(self, reference_mol):
        """
        If target_mol.descriptor is a fingerprint, this method will try
        to calculate the fingerprint of the self molecules.
        If this fails because of the absence of mol_graph atttribute in
        target_molecule, a ValueError is raised.

        Args:
            reference_mol (molSim.ops Molecule): Target molecule. Fingerprint
            of this molecule is used as the reference.

        Raises:
            ValueError
        """
        if reference_mol.descriptor.is_fingerprint():
            try:
                self.set_descriptor(
                    fingerprint_type=reference_mol.descriptor.get_label(),
                    fingerprint_params=reference_mol.descriptor.get_params(),
                )
            except ValueError as e:
                if e.message is None:
                    e.message = ""
                e.message += f" For {self.mol_text}"
                raise e

    def get_similarity_to(self, target_mol, similarity_measure):
        """Get a similarity metric to a target molecule

        Args:
            target_mol (molSim.ops Molecule): Target molecule. Similarity
                score is with respect to this molecule
            similarity_measure (molSim.ops SimilarityMeasure). The similarity
                metric used.

        Returns:
            similarity_score (float): Similarity coefficient by the chosen
                method.

        Raises
        ------
            NotInitializedError
                If target_molecule has uninitialized descriptor. See note.
        """
        try:
            return similarity_measure(self.descriptor, target_mol.descriptor)
        except NotInitializedError as e:
            if e.message is None:
                e.message = ""
            e.message += "Similarity could not be calculated. "
            raise e

    def get_name(self):
        return self.mol_text

    def get_mol_property_val(self):
        return self.mol_property_val

    def draw(self, fpath=None, **kwargs):
        """Draw or molecule graph.

        Args:
            fpath (str): Path of file to store image. If None, image is
                displayed in io. Default is None.
            kwargs (keyword arguments): Arguments to modify plot properties.
        """
        if fpath is None:
            Draw.MolToImage(self.mol_graph, **kwargs).show()
        else:
            Draw.MolToFile(self.mol_graph, fpath, **kwargs)

    @staticmethod
    def is_same(source_molecule, target_molecule):
        """Check if the target_molecule is a duplicate of source_molecule.

        Args:
            source_molecule (molSim.chemical_datastructures Molecule): Source
                molecule to compare.
            target_molecule (molSim.chemical_datastructures Molecule): Target
                molecule to compare.

        Returns:
            bool: True if the molecules are the same.

        """
        return source_molecule.mol_text == target_molecule.mol_text
