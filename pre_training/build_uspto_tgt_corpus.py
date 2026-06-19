#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

RDLogger.DisableLog("rdApp.*")


def compact_smiles(smiles):
    return "".join(str(smiles).split())


def canonicalize_smiles(smiles):
    smiles = compact_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_args():
    parser = ArgumentParser(description="Build a USPTO target-molecule corpus for MLM pretraining.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/uspto-480k-clean"),
        help="Directory containing tgt-train.txt and tgt-val.txt.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output .smi file path.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Write only unique canonical SMILES.",
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize molecules before writing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = [args.input_dir / "tgt-train.txt", args.input_dir / "tgt-val.txt"]
    seen = set()
    written = 0
    kept = 0
    dropped = 0

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as out_handle:
        for path in input_files:
            with path.open("r", encoding="utf-8") as handle:
                iterator = handle
                if tqdm is not None:
                    iterator = tqdm(handle, desc=f"reading {path.name}", leave=True)
                for line in iterator:
                    written += 1
                    smiles = compact_smiles(line)
                    if not smiles:
                        dropped += 1
                        continue
                    if args.canonicalize:
                        smiles = canonicalize_smiles(smiles) or ""
                    else:
                        # Keep only valid SMILES even if we do not canonicalize.
                        if Chem.MolFromSmiles(smiles) is None:
                            dropped += 1
                            continue
                    if not smiles:
                        dropped += 1
                        continue
                    if args.dedupe:
                        if smiles in seen:
                            continue
                        seen.add(smiles)
                    out_handle.write(smiles + "\n")
                    kept += 1

    print(
        {
            "input_files": [str(path) for path in input_files],
            "raw_lines": written,
            "kept_molecules": kept,
            "dropped_lines": dropped,
            "deduped": args.dedupe,
            "canonicalized": args.canonicalize,
            "output_path": str(args.output_path),
        }
    )


if __name__ == "__main__":
    main()
