"""
Abstraction of a molecule with relevant property manipulation methods.
"""
from glob import glob
import os.path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from molSim.utils.helper_methods import get_feature_datatype
from molSim.ops.descriptor import Descriptor
from molSim.ops.similarity_measures import SimilarityMeasure


class Molecule:
    """Molecular object defined from RDKIT mol object.

    """
    def __init__(self,
                 mol_graph=None,
                 mol_text=None,
                 mol_property_val=None,
                 mol_descriptor_val=None,
                 mol_src=None,
                 mol_smiles=None):
        """Constructor

        Parameters
        ----------
        mol_graph: RDKIT mol object
            Graph-level information of molecule.
            Implemented as an RDKIT mol object. Default is None.
        mol_text: str
            Text identifier of the molecule. Default is None.
            Identifiers can be:
            ------------------
            1. Name of the molecule.
            2. SMILES string representing the molecule.
        mol_property_val: float
            Some property associated with the molecule. This is typically the
            response being studied. E.g. Boiling point, Selectivity etc.
            Default is None.
        mol_descriptor_val: numpy ndarray
            Decriptor value for the molecule. Must be numpy array or list.
            Default is None.
        mol_src: str
            Source file or SMILES string to load molecule. Acceptable files are
              -> .pdb file
              -> .txt file with SMILE string in first column, first row and
                      (optionally) property in second column, first row.
            Default is None.
            If provided mol_graph is attempted to be loaded from it.
        mol_smiles: str
            SMILES string for molecule. If provided, mol_graph is loaded from
            it. If mol_text not set in keyword argument, this string is used
            to set it.

        """
        self.mol_graph = mol_graph
        self.mol_text = mol_text
        self.mol_property_val = mol_property_val
        self.descriptor = Descriptor() if mol_descriptor_val is None \
            else Descriptor(value=np.array(mol_descriptor_val))
        if mol_src is not None:
            self._set_molecule_from_file(mol_src)
            if self.mol_graph is None:
                raise ValueError('Could not load molecule from file source',
                                 mol_src)
        if mol_smiles is not None:
            self._set_molecule_from_smiles(mol_smiles)
            if self.mol_graph is None:
                raise ValueError('Could not load molecule from SMILES string',
                                 mol_smiles)

    def _set_molecule_from_smiles(self, mol_smiles):
        """
        Set the mol_graph attribute from smiles string.
        If self.mol_text is not set, it is set to the smiles string.

        Parameters
        ----------
        mol_smiles: str
        SMILES string for molecule. If provided, mol_graph is loaded from
            it. If mol_text not set in keyword argument, this string is used
            to set it.

        """
        self.mol_graph = Chem.MolFromSmiles(mol_smiles)
        if self.mol_text is None:
            self.mol_text = mol_smiles

    def _set_molecule_from_file(self, mol_src):
        """Load molecule graph from file

        Parameters
        mol_src: str
            Source file or SMILES string to load molecule.
            Acceptable files are
              -> .pdb file
              -> .txt file with SMILE string in first column, first row.

        """
        if os.path.isfile(mol_src):
            mol_fname, extension = os.path.basename(mol_src).split('.')
            if extension == 'pdb':
                # read pdb file
                self.mol_graph = Chem.MolFromPDBFile(mol_src)
                if self.mol_text is None:
                    self.mol_text = mol_fname
            elif extension == 'txt':
                with open(mol_src, "r") as fp:
                    mol_smiles = fp.readline().split()[0]
                self._set_molecule_from_smiles(mol_smiles)

    def set_descriptor(self,
                       arbitrary_descriptor_val=None,
                       fingerprint_type=None):
        """Sets molecular descriptor attribute.

        Parameters
        ----------
        arbitrary_descriptor_val : np.array or list
            Arbitrary descriptor vector. Default is None.
        fingerprint_type : str
            String label specifying which fingerprint to use. Default is None.

        """
        if arbitrary_descriptor_val:
            self.descriptor.set_manually(arbitrary_descriptor_val)
        elif fingerprint_type:
            self.descriptor.make_fingerprint(self.mol_graph,
                                             fingerprint_type=fingerprint_type)
        else:
            raise ValueError(f'No descriptor vector were passed.')

    def get_similarity_to_molecule(self,
                                   target_mol,
                                   similarity_measure):
        """Get a similarity metric to a target molecule

        Parameters
        ----------
        target_mol: Molecule object: Target molecule.
            Similarity score is with respect to this molecule
        similarity_measure: SimilarityMeasure object.
            The similarity metric used.

        Returns
        -------
        similarity_score: float
            Similarity coefficient by the chosen method.

        """
        return similarity_measure(self.descriptor, target_mol.descriptor)

    def compare_to_molecule_set(self, molecule_set):
        """Compare the molecule to a database contained in
        a MoleculeSet object.

        Parameters
        ----------
        molecule_set: MoleculeSet object
            Database of molecules to compare against.

        Returns
        -------
        target_similarity: list
           List of similarity scores of molecules of the database when
           compared to the self molecule.

        Note
        ----
        Excludes the self molecule if it is part of the same database.
        Uses mol_text attribute to achieve this.

        """
        target_similarity = [
            self.get_similarity_to_molecule(
                ref_mol, similarity_measure=molecule_set.similarity_measure,
                molecular_descriptor=molecule_set.molecular_descriptor)
            for ref_mol in molecule_set.molecule_database
            if ref_mol.mol_text != self.mol_text]
        return target_similarity
    
    def get_mol_name(self):
        return self.mol_text

    def get_mol_property_val(self):
        return self.mol_property_val
    
    def draw(self, fpath=None, **kwargs):
        """Draw or molecule graph.

        Parameters
        ----------
        fpath: str
            Path of file to store image. If None, image is displayed in io.
            Default is None.
        kwargs: keyword arguments
            Arguments to modify plot properties.

        """
        if fpath is None:
            Draw.MolToImage(self.mol_graph, **kwargs).show()
        else:
            Draw.MolToFile(self.mol_graph, fpath, **kwargs)