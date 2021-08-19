#   Copyright 2020 Martin Vogt, Antonio de la Vega de Leon
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial
#  portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#  WITH  THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file has been edited to include only the functions used in the molSim
# repository in which it is now included by Jackson Burns (UDel). In compliance
# with the license above, the code is included with the repository.
from zlib import adler32


from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint
from rdkit.Chem.rdMolDescriptors import GetTopologicalTorsionFingerprint
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import RDKFingerprint

# Functions for calculating different fingerprints
# All fingerprints are returned as lists of features


def rdkit_fingerprint(mol, **kwargs):
    return list(RDKFingerprint(mol, **kwargs).GetOnBits())


def maccs_keys(mol, **kwargs):
    return list(GetMACCSKeysFingerprint(mol).GetOnBits())


def atom_pairs(mol, **kwargs):
    return list(GetAtomPairFingerprint(mol, **kwargs).GetNonzeroElements())


def torsions(mol, **kwargs):
    return list(
        GetTopologicalTorsionFingerprint(
            mol,
            **kwargs,
        ).GetNonzeroElements()
    )


def morgan(mol, **kwargs):
    return list(GetMorganFingerprint(mol, **kwargs).GetNonzeroElements())


def hashed_atom_pairs(mol, **kwargs):
    return list(
        GetHashedAtomPairFingerprint(
            mol,
            **kwargs,
        ).GetNonzeroElements()
    )


def hashed_torsions(mol, **kwargs):
    return list(
        GetHashedTopologicalTorsionFingerprint(
            mol,
            **kwargs,
        ).GetNonzeroElements()
    )


def hashed_morgan(mol, **kwargs):
    return list(
        GetHashedMorganFingerprint(
            mol,
            **kwargs,
        ).GetNonzeroElements()
    )


def avalon(mol, **kwargs):
    return list(GetAvalonFP(mol, **kwargs).GetOnBits())


def generate_fingerprints(mol_suppl, fp, **pars):
    for mol in mol_suppl:
        if mol:
            yield fp(mol, **pars)


def hash_parameter_set(pars):
    s = sorted(pars.items())
    return adler32(str(s).encode("UTF-8"))


def to_key_val_string(pars):
    return " ".join("{}:{}".format(k, v) for k, v in sorted(pars.items()))


fingerprints = {
    "rdkit": rdkit_fingerprint,
    "maccs": maccs_keys,
    "atom_pairs": atom_pairs,
    "torsions": torsions,
    "morgan": morgan,
    "hashed_atom_pairs": hashed_atom_pairs,
    "hashed_torsions": hashed_torsions,
    "hashed_morgan": hashed_morgan,
    "avalon": avalon,
}
