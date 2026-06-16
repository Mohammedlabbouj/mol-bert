import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature import mol2alt_sentence
from rdkit import Chem


def load_first_smiles(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            first = line.split()[0]
            if first.lower() == "smiles":
                continue
            return first
    raise ValueError(f"No SMILES found in {path}")


def tokenize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    radius0 = mol2alt_sentence(mol, 0)
    radius1 = mol2alt_sentence(mol, 1)
    if len(radius0) > 0 and len(radius1) >= 2 * len(radius0):
        return [str(radius1[len(radius0) + i]) + str(radius0[i]) for i in range(len(radius0))]
    return [str(token) for token in radius1 if token is not None]


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_vocab(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Show fingerprint tokenization for one ZINC-20 SMILES")
    parser.add_argument("--input_path", type=str, default="data/ZINC-20/zinc_2m_subset.smi")
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default="croups/ident_merge.pickle")
    return parser.parse_args()


def main():
    args = parse_args()
    smiles = args.smiles or load_first_smiles(args.input_path)
    canonical = canonicalize_smiles(smiles)
    vocab = load_vocab(args.vocab_path)
    tokens = tokenize_smiles(canonical or smiles)
    token_ids = [vocab.get(token, vocab.get("unk_index", 1)) for token in tokens]
    unk_id = vocab.get("unk_index", 1)
    unk_count = sum(1 for token_id in token_ids if token_id == unk_id)
    print("SMILES:", smiles)
    print("Canonical:", canonical)
    print("Token count:", len(tokens))
    print("UNK count:", unk_count)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)


if __name__ == "__main__":
    main()
