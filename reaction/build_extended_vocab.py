#!/usr/bin/env python3
import json
import pickle
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from reaction.tokenizers import FingerprintTokenizer


def parse_args():
    parser = ArgumentParser(description="Extend a fingerprint vocab with USPTO-missing substructure tokens.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing cleaned USPTO split files (src-*.txt).",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("croups/ident_merge.pickle"),
        help="Original fingerprint vocab pickle.",
    )
    parser.add_argument(
        "--output_vocab_path",
        type=Path,
        required=True,
        help="Path to write the extended vocab pickle.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = FingerprintTokenizer(args.vocab_path)

    missing_counts = Counter()
    total_tokens = 0
    total_missing = 0
    rows = 0

    for split in ("train", "val", "test"):
        src_path = args.input_dir / f"src-{split}.txt"
        with src_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows += 1
                source = "".join(line.split())
                if not source:
                    continue
                for fragment in source.split("."):
                    fragment = fragment.strip()
                    if not fragment:
                        continue
                    tokens = tokenizer._morgan_tokens_for_smiles(fragment)
                    total_tokens += len(tokens)
                    for token in tokens:
                        if token not in tokenizer.vocab:
                            missing_counts[token] += 1
                            total_missing += 1

    vocab = dict(tokenizer.vocab)
    next_id = max(vocab.values()) + 1 if vocab else 0
    added = 0
    for token, _count in sorted(missing_counts.items(), key=lambda item: (-item[1], item[0])):
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
            added += 1

    args.output_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_vocab_path.open("wb") as handle:
        pickle.dump(vocab, handle)

    report = {
        "input_vocab_size": len(tokenizer.vocab),
        "output_vocab_size": len(vocab),
        "added_tokens": added,
        "rows_scanned": rows,
        "total_source_tokens": total_tokens,
        "missing_token_occurrences": total_missing,
        "unique_missing_tokens": len(missing_counts),
        "coverage_before": 1.0 - (total_missing / total_tokens if total_tokens else 0.0),
        "top_missing_tokens": missing_counts.most_common(50),
    }

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
