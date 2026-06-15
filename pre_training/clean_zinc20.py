import argparse
import sys
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def canonicalize_smiles(smiles):
    smiles = "".join(str(smiles).split())
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def clean_zinc20(raw_path, output_path):
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    skipped = 0
    with raw_path.open("r", encoding="utf-8", errors="ignore") as reader, output_path.open("w", encoding="utf-8") as writer:
        for line in tqdm(reader, desc="Cleaning ZINC-20"):
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            first = line.split()[0]
            if first.lower() == "smiles":
                continue
            canonical = canonicalize_smiles(first)
            if canonical is None:
                skipped += 1
                continue
            writer.write(canonical + "\n")
            kept += 1
    return {"total": total, "kept": kept, "skipped": skipped, "output_path": str(output_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Clean ZINC-20 into a canonical SMILES corpus")
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    stats = clean_zinc20(args.raw_data_path, args.output_path)
    print(stats)


if __name__ == "__main__":
    main()

