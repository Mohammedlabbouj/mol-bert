from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator as FG

RDLogger.DisableLog("rdApp.warning")


def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    if mol is None:
        return []

    generator = FG.GetMorganGenerator(radius=int(radius))
    additional_output = FG.AdditionalOutput()
    additional_output.CollectBitInfoMap()
    _ = generator.GetSparseCountFingerprint(mol, additionalOutput=additional_output)
    info = additional_output.GetBitInfoMap()

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]

    #     print(mol_atoms)
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element, occurrences in info.items():
        for atom_idx, radius_at in occurrences:
            if atom_idx in dict_atoms and radius_at in dict_atoms[atom_idx]:
                dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)
