#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import pickle


def parse_args():
    parser = ArgumentParser(description="Inspect a fingerprint vocabulary pickle.")
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("croups/ident_merge.pickle"),
        help="Path to the fingerprint vocabulary pickle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.vocab_path.open("rb") as handle:
        vocab = pickle.load(handle)

    print(f"path: {args.vocab_path}")
    print(f"type: {type(vocab).__name__}")

    if hasattr(vocab, "__len__"):
        print(f"len: {len(vocab)}")

    if isinstance(vocab, dict):
        values = list(vocab.values())
        print(f"min_id: {min(values) if values else 'n/a'}")
        print(f"max_id: {max(values) if values else 'n/a'}")
        print("sample_items:")
        for key, value in list(vocab.items())[:10]:
            print(f"  {key} -> {value}")


if __name__ == "__main__":
    main()
