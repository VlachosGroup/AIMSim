from time import time as t
from molecular_similarity import Molecule, Molecules
import rdkit

start = t()
"""
Assert test format:

assert call_to_something(args) == expected_output, \
    output_if_fails

"""
# this will be used as arguments to the various tests
path = r'test_smiles_responses.txt'
# triethylphosphine
testMol = rdkit.Chem.MolFromSmiles('CCP(CC)CC')
# trimethylphosphine
testMol2 = rdkit.Chem.MolFromSmiles('CP(C)C')

# make a molecule
testMolecule = Molecule(
    testMol,
    "test_name",
    13.22)
# tests
assert testMolecule, \
    "Failed to instantiate Molecule object"

assert testMolecule.mol_property == 13.22, \
    "Failed to assign mol_property"

assert testMolecule.name_ == "test_name", \
    "Failed to assign name__"

assert testMolecule._get_morgan_fingerprint(
    ) == rdkit.Chem.AllChem.GetMorganFingerprint(testMol, 3), \
    "Failed to retrieve morgan fingerprint"

assert testMolecule._get_morgan_fingerprint(
    3, 2) == rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(
        testMol, 3, nBits=2), \
    "Failed to retrieve morgan fingerprint"

assert testMolecule._get_rdkit_topological_fingerprint(
    ) == rdkit.Chem.rdmolops.RDKFingerprint(
        testMol, minPath=1, maxPath=7), \
        "Failed to retrieve RDKit topological fingerprint"

testMolmorgan = rdkit.Chem.AllChem.GetMorganFingerprint(testMol, 3)
testMol2morgan = rdkit.Chem.AllChem.GetMorganFingerprint(testMol2, 3)
assert testMolecule.get_similarity(
    'tanimoto_similarity',
    testMolmorgan, testMol2morgan
    ) == rdkit.DataStructs.TanimotoSimilarity(
        testMolmorgan, testMol2morgan), \
    "Failed to calculate tanimoto similarity"

# make a group of molecules using a short list of SMILES strings
testMolecules = Molecules(
    path, 'tanimoto_similarity',
    'rdkit_topological', False)

assert testMolecules, \
    "Failed to instantiate Molecules object"

assert testMolecules.isVerbose is False, \
    "Failed to assign isVerbose"

assert testMolecules.molecular_descriptor == 'rdkit_topological', \
    "Failed to assign molecular_desciptor"

assert testMolecules.similarity_measure == 'tanimoto_similarity', \
    "Failed to assign similarity_measure"

# cannot test contents of list, as each molecule is stored as an
# object with a unique address so instantitating a new molecule
# with Molecule(fails). Writing an __eq__ method may solve this,
# but would be a lot of work for little profit
assert testMolecules.mols, \
    "Failed to build list of molecules"

# triethylphosphine
testMol = rdkit.Chem.MolFromSmiles('CCP(CC)CC')
# trimethylphosphine
testMol2 = rdkit.Chem.MolFromSmiles('CP(C)C')

# make a molecule
testMolecule = Molecule(
    testMol,
    "test_name",
    13.22)

# make another group of molecules using a short list of SMILES strings,
# check again for new values
testMolecules = Molecules(
    path, 'tanimoto_similarity',
    'morgan_fingerprint', True)

assert testMolecules.isVerbose is True, \
    "Failed to assign isVerbose"

assert testMolecules.molecular_descriptor == 'morgan_fingerprint', \
    "Failed to assign molecular_desciptor"

assert testMolecules.similarity_measure == 'tanimoto_similarity', \
    "Failed to assign similarity_measure"

print(f'Execution took {t()-start:.2f} seconds.')
