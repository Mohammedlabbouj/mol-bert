import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.bert_model import BertConfig
from reaction.dataset import (
    FingerprintReactionDataset,
    canonicalize_smiles,
    load_parallel_reaction_records,
    load_reaction_records,
)
from reaction.model import FingerprintReactionModel
from reaction.tokenizers import SmilesTokenizer
from reaction.train_forward import (
    evaluate_beam,
    evaluate_loss,
    get_source_tokenizer,
    infer_encoder_config,
    make_loader,
)


def _coalesce(value, fallback):
    return fallback if value is None else value


def load_checkpoint_state(path):
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _limit_records(records, max_examples):
    if max_examples is None:
        return records
    max_examples = int(max_examples)
    if max_examples <= 0:
        return records[:0]
    return records[:max_examples]


def _select_records(records, max_examples=None, random_subset=False, seed=42):
    if max_examples is None:
        return list(records)
    max_examples = int(max_examples)
    if max_examples <= 0:
        return []
    if max_examples >= len(records):
        return list(records)
    if random_subset:
        rng = random.Random(seed)
        indices = rng.sample(range(len(records)), max_examples)
        return [records[index] for index in indices]
    return list(records[:max_examples])


def build_datasets(args, target_tokenizer, valid_limit=None, test_limit=None, random_subset=False, seed=42):
    if args.data_dir and not (args.train_path or args.valid_path or args.test_path):
        args.train_path = args.train_path or str(Path(args.data_dir) / "src-train.txt")
        args.valid_path = args.valid_path or str(Path(args.data_dir) / "src-val.txt")
        args.test_path = args.test_path or str(Path(args.data_dir) / "src-test.txt")
        args.train_target_path = args.train_target_path or str(Path(args.data_dir) / "tgt-train.txt")
        args.valid_target_path = args.valid_target_path or str(Path(args.data_dir) / "tgt-val.txt")
        args.test_target_path = args.test_target_path or str(Path(args.data_dir) / "tgt-test.txt")

    if args.train_path and args.valid_path and args.test_path:
        if args.train_target_path and args.valid_target_path and args.test_target_path:
            train_records = load_parallel_reaction_records(args.train_path, args.train_target_path)
            valid_records = load_parallel_reaction_records(args.valid_path, args.valid_target_path)
            test_records = load_parallel_reaction_records(args.test_path, args.test_target_path)
        else:
            train_records = load_reaction_records(
                args.train_path,
                reaction_column=args.reaction_column,
                reactants_column=args.reactants_column,
                reagents_column=args.reagents_column,
                product_column=args.product_column,
            )
            valid_records = load_reaction_records(
                args.valid_path,
                reaction_column=args.reaction_column,
                reactants_column=args.reactants_column,
                reagents_column=args.reagents_column,
                product_column=args.product_column,
            )
            test_records = load_reaction_records(
                args.test_path,
                reaction_column=args.reaction_column,
                reactants_column=args.reactants_column,
                reagents_column=args.reagents_column,
                product_column=args.product_column,
            )
        valid_records = _select_records(valid_records, valid_limit, random_subset=random_subset, seed=seed)
        test_records = _select_records(test_records, test_limit, random_subset=random_subset, seed=seed)
        train_dataset = FingerprintReactionDataset.from_records(
            train_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
        valid_dataset = FingerprintReactionDataset.from_records(
            valid_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
        test_dataset = FingerprintReactionDataset.from_records(
            test_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
        return train_dataset, valid_dataset, test_dataset

    if args.data_path is not None and args.train_size is not None and args.valid_size is not None:
        records = load_reaction_records(
            args.data_path,
            reaction_column=args.reaction_column,
            reactants_column=args.reactants_column,
            reagents_column=args.reagents_column,
            product_column=args.product_column,
        )
        total = len(records)
        train_len = int(total * args.train_size)
        valid_len = int(total * args.valid_size)
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(total, generator=generator).tolist()
        train_indices = permutation[:train_len]
        valid_indices = permutation[train_len:train_len + valid_len]
        test_indices = permutation[train_len + valid_len:]
        train_records = [records[index] for index in train_indices]
        valid_records = [records[index] for index in valid_indices]
        test_records = [records[index] for index in test_indices]
        valid_records = _select_records(valid_records, valid_limit, random_subset=random_subset, seed=seed)
        test_records = _select_records(test_records, test_limit, random_subset=random_subset, seed=seed)
        train_dataset = FingerprintReactionDataset.from_records(
            train_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
        valid_dataset = FingerprintReactionDataset.from_records(
            valid_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
        test_dataset = FingerprintReactionDataset.from_records(
            test_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
        return train_dataset, valid_dataset, test_dataset

    raise ValueError("Provide either train/valid/test files or train_size and valid_size split fractions.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the best forward-reaction checkpoint")
    parser.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--train_target_path", type=str, default=None)
    parser.add_argument("--valid_target_path", type=str, default=None)
    parser.add_argument("--test_target_path", type=str, default=None)
    parser.add_argument("--train_size", type=float, default=None)
    parser.add_argument("--valid_size", type=float, default=None)
    parser.add_argument("--fingerprint_vocab_path", type=str, default="croups/ident_merge.pickle")
    parser.add_argument("--target_tokenizer_path", type=str, default=None)
    parser.add_argument("--reaction_column", type=str, default=None)
    parser.add_argument("--reactants_column", type=str, default=None)
    parser.add_argument("--reagents_column", type=str, default=None)
    parser.add_argument("--product_column", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_source_len", type=int, default=None)
    parser.add_argument("--max_target_len", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--beam_eval_examples", type=int, default=None)
    parser.add_argument("--valid_eval_examples", type=int, default=None, help="Limit validation evaluation to the first N examples.")
    parser.add_argument("--test_eval_examples", type=int, default=None, help="Limit test evaluation to the first N examples.")
    parser.add_argument("--random_eval_subset", action="store_true", help="Randomly sample validation/test evaluation examples instead of taking the first N.")
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation evaluation and only run test evaluation.")
    parser.add_argument("--skip_test", action="store_true", help="Skip test evaluation and only run validation evaluation.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1200)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=6)
    parser.add_argument("--decoder_ff_size", type=int, default=0)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--disable_target_canonicalization", action="store_true", default=None)
    parser.add_argument("--tie_decoder_weights", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("reaction_output") / "best_epoch_eval")
    parser.add_argument("--report_name", type=str, default="evaluation_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint_state(args.checkpoint_path)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    args.batch_size = _coalesce(args.batch_size, checkpoint_args.get("batch_size", 32))
    args.max_source_len = _coalesce(args.max_source_len, checkpoint_args.get("max_source_len", 256))
    args.max_target_len = _coalesce(args.max_target_len, checkpoint_args.get("max_target_len", 256))
    args.beam_size = _coalesce(args.beam_size, checkpoint_args.get("beam_size", 5))
    args.beam_eval_examples = _coalesce(args.beam_eval_examples, checkpoint_args.get("beam_eval_examples", 64))
    args.num_workers = _coalesce(args.num_workers, checkpoint_args.get("num_workers", 0))
    args.device = _coalesce(args.device, "cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_target_canonicalization is None:
        args.disable_target_canonicalization = bool(checkpoint_args.get("disable_target_canonicalization", False))
    if args.tie_decoder_weights is None:
        args.tie_decoder_weights = bool(checkpoint_args.get("tie_decoder_weights", False))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available in this environment. Falling back to CPU.", flush=True)
        args.device = "cpu"

    if "target_tokenizer" in checkpoint:
        target_tokenizer = SmilesTokenizer(token_to_id=checkpoint["target_tokenizer"])
    elif args.target_tokenizer_path:
        target_tokenizer = SmilesTokenizer.load(args.target_tokenizer_path)
    else:
        if not args.data_dir and not (args.train_path and args.valid_path and args.test_path):
            raise ValueError("A checkpoint tokenizer was not found. Please provide --target_tokenizer_path or the dataset splits.")
        temp_records = load_reaction_records(
            args.train_path or str(Path(args.data_dir) / "src-train.txt"),
            reaction_column=args.reaction_column,
            reactants_column=args.reactants_column,
            reagents_column=args.reagents_column,
            product_column=args.product_column,
        )
        target_tokenizer = SmilesTokenizer.build(
            [canonicalize_smiles(record["product"]) or record["product"] for record in temp_records]
        )

    train_dataset, valid_dataset, test_dataset = build_datasets(
        args,
        target_tokenizer,
        valid_limit=args.valid_eval_examples,
        test_limit=args.test_eval_examples,
        random_subset=args.random_eval_subset,
        seed=args.seed,
    )

    source_vocab_size = get_source_tokenizer(train_dataset).vocab_size
    if isinstance(checkpoint, dict) and "encoder_config" in checkpoint:
        encoder_config = BertConfig(**checkpoint["encoder_config"])
    else:
        encoder_config = infer_encoder_config(
            args.checkpoint_path,
            vocab_size=source_vocab_size,
            default_hidden_size=args.hidden_size,
            default_layers=args.num_hidden_layers,
            default_heads=args.num_attention_heads,
            default_ff=args.intermediate_size,
        )
    if encoder_config.vocab_size != source_vocab_size:
        print(
            f"Warning: checkpoint encoder vocab_size={encoder_config.vocab_size} "
            f"but source vocab size from {args.fingerprint_vocab_path} is {source_vocab_size}. "
            "Use the same fingerprint vocab that was used for training."
        )

    target_pad_id = target_tokenizer.pad_id
    target_bos_id = target_tokenizer.bos_id
    target_eos_id = target_tokenizer.eos_id

    model = FingerprintReactionModel(
        encoder_config=encoder_config,
        target_vocab_size=target_tokenizer.vocab_size,
        target_pad_id=target_pad_id,
        target_bos_id=target_bos_id,
        target_eos_id=target_eos_id,
        encoder_checkpoint=None,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff_size=args.decoder_ff_size or None,
        decoder_dropout=args.decoder_dropout,
        max_target_len=args.max_target_len,
        tie_decoder_weights=args.tie_decoder_weights,
    )
    try:
        model = model.to(args.device)
    except RuntimeError as exc:
        if args.device.startswith("cuda"):
            print(f"Falling back to CPU after CUDA init failure: {exc}", flush=True)
            args.device = "cpu"
            model = model.to(args.device)
        else:
            raise

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded checkpoint with missing keys: {missing}")
        print(f"Loaded checkpoint with unexpected keys: {unexpected}")

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pad_source_id=get_source_tokenizer(train_dataset).pad_token_id,
        pad_target_id=target_pad_id,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pad_source_id=get_source_tokenizer(valid_dataset).pad_token_id,
        pad_target_id=target_pad_id,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pad_source_id=get_source_tokenizer(test_dataset).pad_token_id,
        pad_target_id=target_pad_id,
        num_workers=args.num_workers,
    )

    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    report = {
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "checkpoint_epoch": checkpoint_epoch,
        "device": args.device,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "beam_eval_examples": args.beam_eval_examples,
        "valid_eval_examples": args.valid_eval_examples,
        "test_eval_examples": args.test_eval_examples,
        "source_vocab_size": source_vocab_size,
        "target_vocab_size": target_tokenizer.vocab_size,
    }

    print(f"Evaluating checkpoint from epoch {checkpoint_epoch}", flush=True)
    if not args.skip_validation:
        print("Running validation loss...", flush=True)
        valid_loss_metrics = evaluate_loss(model, valid_loader, args.device, desc="valid-loss")
        report["valid_loss"] = valid_loss_metrics["valid_loss"]
        report["valid_source_coverage"] = valid_loss_metrics["source_coverage"]

        if args.beam_eval_examples and args.beam_eval_examples > 0:
            print("Running validation beam metrics...", flush=True)
            report.update(
                {
                    f"valid_{key}": value
                    for key, value in evaluate_beam(
                        model,
                        valid_loader,
                        args.device,
                        beam_size=args.beam_size,
                        max_examples=args.beam_eval_examples,
                    ).items()
                }
            )

    if not args.skip_test:
        print("Running test loss...", flush=True)
        test_loss_metrics = evaluate_loss(model, test_loader, args.device, desc="test-loss")
        report["test_loss"] = test_loss_metrics["valid_loss"]
        report["test_source_coverage"] = test_loss_metrics["source_coverage"]

        if args.beam_eval_examples and args.beam_eval_examples > 0:
            print("Running test beam metrics...", flush=True)
            report.update(
                {
                    f"test_{key}": value
                    for key, value in evaluate_beam(
                        model,
                        test_loader,
                        args.device,
                        beam_size=args.beam_size,
                        max_examples=args.beam_eval_examples,
                    ).items()
                }
            )

    report_path = args.output_dir / args.report_name
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
