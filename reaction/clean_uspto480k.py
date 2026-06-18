#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from reaction.tokenizers import FingerprintTokenizer


def compact_line(line):
    return "".join(str(line).split())


def is_valid_smiles(smiles):
    smiles = compact_line(smiles)
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return False
    return True


def validate_reaction_side(side):
    side = compact_line(side)
    if not side:
        return True
    for fragment in side.split("."):
        fragment = fragment.strip()
        if not fragment:
            continue
        if not is_valid_smiles(fragment):
            return False
    return True


def source_has_unk(source_tokenizer, source_smiles):
    encoded = source_tokenizer.encode_reaction_context(source_smiles, "")
    unk_count = sum(1 for token_id in encoded if token_id == source_tokenizer.unk_token_id)
    return unk_count, len(encoded)


def clean_pair_file(
    src_path,
    tgt_path,
    output_dir,
    canonicalize_targets=False,
    filter_source_unk=False,
    max_source_unk_ratio=1.0,
    min_source_tokens=1,
    source_tokenizer=None,
):
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_src = output_dir / src_path.name
    out_tgt = output_dir / tgt_path.name
    bad_path = output_dir / f"{src_path.stem}_invalid_rows.json"

    with src_path.open("r", encoding="utf-8") as src_handle, tgt_path.open("r", encoding="utf-8") as tgt_handle:
        src_lines = src_handle.readlines()
        tgt_lines = tgt_handle.readlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Length mismatch for {src_path.name} and {tgt_path.name}: {len(src_lines)} vs {len(tgt_lines)}")

    kept = 0
    dropped = 0
    invalid_rows = []
    source_unk_rows = 0

    iterator = zip(src_lines, tgt_lines)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(src_lines), desc=f"cleaning {src_path.name}", leave=True)

    with out_src.open("w", encoding="utf-8") as src_out, out_tgt.open("w", encoding="utf-8") as tgt_out:
        for line_no, (src_line, tgt_line) in enumerate(iterator, start=1):
            src_clean = compact_line(src_line)
            tgt_clean = compact_line(tgt_line)

            if not src_clean or not tgt_clean:
                dropped += 1
                invalid_rows.append({"line": line_no, "reason": "blank_line"})
                continue

            if not validate_reaction_side(src_clean):
                dropped += 1
                invalid_rows.append({"line": line_no, "reason": "invalid_source_smiles", "source": src_clean, "target": tgt_clean})
                continue

            if not is_valid_smiles(tgt_clean):
                dropped += 1
                invalid_rows.append({"line": line_no, "reason": "invalid_target_smiles", "source": src_clean, "target": tgt_clean})
                continue

            if canonicalize_targets:
                mol = Chem.MolFromSmiles(tgt_clean)
                tgt_clean = Chem.MolToSmiles(mol, canonical=True) if mol is not None else tgt_clean

            if filter_source_unk and source_tokenizer is not None:
                source_unk_count, source_token_count = source_has_unk(source_tokenizer, src_clean)
                source_unk_ratio = source_unk_count / max(source_token_count, 1)
                if source_token_count >= min_source_tokens and source_unk_ratio > max_source_unk_ratio:
                    dropped += 1
                    source_unk_rows += 1
                    invalid_rows.append(
                        {
                            "line": line_no,
                            "reason": "source_vocab_unk",
                            "source": src_clean,
                            "target": tgt_clean,
                            "source_unk_count": source_unk_count,
                            "source_token_count": source_token_count,
                            "source_unk_ratio": source_unk_ratio,
                        }
                    )
                    continue

            src_out.write(src_clean + "\n")
            tgt_out.write(tgt_clean + "\n")
            kept += 1
            if tqdm is None and line_no % 10000 == 0:
                print(f"{src_path.name}: checked {line_no}/{len(src_lines)} | kept {kept} | dropped {dropped}", flush=True)

    with bad_path.open("w", encoding="utf-8") as handle:
        json.dump(invalid_rows, handle, indent=2)

    report = {
        "source_file": src_path.name,
        "target_file": tgt_path.name,
        "input_rows": len(src_lines),
        "kept_rows": kept,
        "dropped_rows": dropped,
        "source_unk_rows": source_unk_rows,
        "invalid_rows_file": bad_path.name,
    }
    return report


def parse_args():
    parser = ArgumentParser(description="Clean and validate USPTO-480k split files.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing src-*.txt and tgt-*.txt")
    parser.add_argument("--output_dir", type=Path, required=True, help="Destination folder for cleaned files")
    parser.add_argument(
        "--fingerprint_vocab_path",
        type=Path,
        default=Path("croups/ident_merge.pickle"),
        help="Fingerprint vocabulary used to detect source-side UNK substructures.",
    )
    parser.add_argument(
        "--filter_source_unk",
        action="store_true",
        help="Drop rows whose source fingerprint tokens contain too many UNK tokens.",
    )
    parser.add_argument(
        "--max_source_unk_ratio",
        type=float,
        default=1.0,
        help="Maximum allowed fraction of UNK source tokens before a row is dropped.",
    )
    parser.add_argument(
        "--min_source_tokens",
        type=int,
        default=1,
        help="Only apply the UNK ratio filter when the source has at least this many tokens.",
    )
    parser.add_argument(
        "--canonicalize_targets",
        action="store_true",
        help="Canonicalize product SMILES in the target files after validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_tokenizer = FingerprintTokenizer(args.fingerprint_vocab_path) if args.filter_source_unk else None
    pairs = [
        ("src-train.txt", "tgt-train.txt"),
        ("src-val.txt", "tgt-val.txt"),
        ("src-test.txt", "tgt-test.txt"),
    ]
    reports = []
    for src_name, tgt_name in pairs:
        report = clean_pair_file(
            args.input_dir / src_name,
            args.input_dir / tgt_name,
            args.output_dir,
            canonicalize_targets=args.canonicalize_targets,
            filter_source_unk=args.filter_source_unk,
            max_source_unk_ratio=args.max_source_unk_ratio,
            min_source_tokens=args.min_source_tokens,
            source_tokenizer=source_tokenizer,
        )
        reports.append(report)
        extra = f", source_unk={report['source_unk_rows']}" if args.filter_source_unk else ""
        print(f"cleaned {src_name} / {tgt_name}: kept {report['kept_rows']} of {report['input_rows']}{extra}")

    summary_path = Path(args.output_dir) / "clean_report.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    print(f"wrote report to {summary_path}")


if __name__ == "__main__":
    main()
