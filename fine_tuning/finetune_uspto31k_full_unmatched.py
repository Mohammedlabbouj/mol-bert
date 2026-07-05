import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.bert_model import BertConfig
from reaction.dataset import FingerprintReactionDataset, load_parallel_reaction_records
from reaction.model import FingerprintReactionModel
from reaction.tokenizers import SmilesTokenizer
from reaction.train_forward import (
    evaluate_beam,
    evaluate_loss,
    get_source_tokenizer,
    make_loader,
    make_record_subset_loader,
    train_epoch,
)


def load_json_config(path):
    if path is None:
        return {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config(args):
    config = {}
    config.update(load_json_config(args.config))
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    return config


def _coalesce(value, fallback):
    return fallback if value is None else value


def load_checkpoint_state(path):
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_datasets(args, target_tokenizer):
    if args.data_dir and not (args.train_path or args.valid_path or args.test_path):
        data_dir = Path(args.data_dir)
        split_paths = {
            "train_path": data_dir / "src-train.txt",
            "valid_path": data_dir / "src-val.txt",
            "train_target_path": data_dir / "tgt-train.txt",
            "valid_target_path": data_dir / "tgt-val.txt",
        }
        if not all(path.exists() for path in split_paths.values()):
            raise FileNotFoundError(
                "This fine-tune script no longer does internal random splitting. "
                "Please provide your own train/valid split files, either via "
                "--train_path/--valid_path and matching target paths, or place "
                "src-train.txt, src-val.txt, tgt-train.txt, and tgt-val.txt in the "
                "data directory."
            )
        args.train_path = str(split_paths["train_path"])
        args.valid_path = str(split_paths["valid_path"])
        args.train_target_path = str(split_paths["train_target_path"])
        args.valid_target_path = str(split_paths["valid_target_path"])

    if not (args.train_path and args.valid_path):
        raise ValueError(
            "Please provide explicit train/valid split files. "
            "The script does not split the full USPTO-31K file internally."
        )

    train_records = load_parallel_reaction_records(args.train_path, args.train_target_path)
    valid_records = load_parallel_reaction_records(args.valid_path, args.valid_target_path)
    test_records = []
    if args.test_path and args.test_target_path:
        test_records = load_parallel_reaction_records(args.test_path, args.test_target_path)
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
    test_dataset = None
    if test_records:
        test_dataset = FingerprintReactionDataset.from_records(
            test_records,
            fingerprint_vocab_path=args.fingerprint_vocab_path,
            smiles_tokenizer=target_tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            canonicalize_targets=not args.disable_target_canonicalization,
        )
    return train_dataset, valid_dataset, test_dataset, {
        "total_records": len(train_records) + len(valid_records) + len(test_records),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "test_records": len(test_records),
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "test_path": args.test_path,
        "train_target_path": args.train_target_path,
        "valid_target_path": args.valid_target_path,
        "test_target_path": args.test_target_path,
    }


def build_model_from_checkpoint(checkpoint, args, target_tokenizer):
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    encoder_config_data = checkpoint.get("encoder_config", None) if isinstance(checkpoint, dict) else None
    if encoder_config_data is None:
        raise ValueError("Checkpoint does not contain encoder_config; cannot rebuild the model safely.")
    encoder_config = BertConfig(**encoder_config_data)
    model = FingerprintReactionModel(
        encoder_config=encoder_config,
        target_vocab_size=target_tokenizer.vocab_size,
        target_pad_id=target_tokenizer.pad_id,
        target_bos_id=target_tokenizer.bos_id,
        target_eos_id=target_tokenizer.eos_id,
        encoder_checkpoint=None,
        decoder_layers=checkpoint_args.get("decoder_layers", 4),
        decoder_heads=checkpoint_args.get("decoder_heads", 6),
        decoder_ff_size=checkpoint_args.get("decoder_ff_size", 0) or None,
        decoder_dropout=checkpoint_args.get("decoder_dropout", 0.1),
        max_target_len=checkpoint_args.get("max_target_len", args.max_target_len),
        tie_decoder_weights=checkpoint_args.get("tie_decoder_weights", False),
    )
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded checkpoint with missing keys: {missing}")
        print(f"Loaded checkpoint with unexpected keys: {unexpected}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the Mol-BERT forward model on USPTO-31K unmatched")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing src-full.txt and tgt-full.txt")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Best checkpoint to fine-tune from.")
    parser.add_argument("--fingerprint_vocab_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--train_target_path", type=str, default=None)
    parser.add_argument("--valid_target_path", type=str, default=None)
    parser.add_argument("--test_target_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_encoder", type=float, default=None)
    parser.add_argument("--lr_decoder", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--beam_eval_examples", type=int, default=None)
    parser.add_argument("--valid_eval_examples", type=int, default=None)
    parser.add_argument("--test_eval_examples", type=int, default=None)
    parser.add_argument("--random_eval_subset", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_source_len", type=int, default=None)
    parser.add_argument("--max_target_len", type=int, default=None)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--load_optimizer_state", action="store_true", default=None)
    parser.add_argument("--evaluate_test_each_epoch", action="store_true", default=None)
    parser.add_argument("--disable_target_canonicalization", action="store_true", default=None)
    parser.add_argument("--monitor_metric", type=str, default=None, help="Validation metric to use for early stopping and best-checkpoint saving.")
    parser.add_argument("--monitor_mode", type=str, default=None, help="Either 'min' or 'max'. If omitted, inferred from monitor_metric.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolve_config(args)

    data_dir = config.get("data_dir", "data/uspto_31k_full_unmatched")
    checkpoint_path = config.get("checkpoint_path", "reaction_output/best_checkpoint.pt")
    fingerprint_vocab_path = config.get("fingerprint_vocab_path", "croups/ident_merge_uspto_extended.pickle")
    output_dir = Path(config.get("output_dir", "output/finetuning_uspto31k_full_unmatched"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint = load_checkpoint_state(checkpoint_path)
    if "target_tokenizer" not in checkpoint:
        raise ValueError(
            "The checkpoint does not contain a target tokenizer. "
            "Please use a checkpoint produced by reaction/train_forward.py."
        )
    target_tokenizer = SmilesTokenizer(token_to_id=checkpoint["target_tokenizer"])

    args_namespace = argparse.Namespace(
        data_dir=data_dir,
        fingerprint_vocab_path=fingerprint_vocab_path,
        max_source_len=int(config.get("max_source_len", 256)),
        max_target_len=int(config.get("max_target_len", checkpoint.get("args", {}).get("max_target_len", 256))),
        disable_target_canonicalization=bool(config.get("disable_target_canonicalization", False)),
        train_path=config.get("train_path"),
        valid_path=config.get("valid_path"),
        test_path=config.get("test_path"),
        train_target_path=config.get("train_target_path"),
        valid_target_path=config.get("valid_target_path"),
        test_target_path=config.get("test_target_path"),
        seed=seed,
        valid_eval_examples=config.get("valid_eval_examples", None),
        test_eval_examples=config.get("test_eval_examples", None),
        random_eval_subset=bool(config.get("random_eval_subset", False)),
    )

    train_dataset, valid_dataset, test_dataset, split_report = build_datasets(args_namespace, target_tokenizer)
    encoder_config = BertConfig(**checkpoint["encoder_config"])
    model = build_model_from_checkpoint(checkpoint, args_namespace, target_tokenizer)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available in this environment. Falling back to CPU.", flush=True)
        device = "cpu"
    model = model.to(device)

    batch_size = int(config.get("batch_size", checkpoint.get("args", {}).get("batch_size", 32)))
    num_workers = int(config.get("num_workers", checkpoint.get("args", {}).get("num_workers", 0)))
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pad_source_id=get_source_tokenizer(train_dataset).pad_token_id,
        pad_target_id=target_tokenizer.pad_id,
        num_workers=num_workers,
    )
    valid_eval_examples = config.get("valid_eval_examples", None)
    random_eval_subset = bool(config.get("random_eval_subset", False))

    def build_valid_loader(epoch_seed):
        if valid_eval_examples is not None:
            return make_record_subset_loader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                pad_source_id=get_source_tokenizer(valid_dataset).pad_token_id,
                pad_target_id=target_tokenizer.pad_id,
                num_workers=num_workers,
                max_examples=int(valid_eval_examples),
                random_subset=random_eval_subset,
                seed=epoch_seed,
            )
        return make_loader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pad_source_id=get_source_tokenizer(valid_dataset).pad_token_id,
            pad_target_id=target_tokenizer.pad_id,
            num_workers=num_workers,
        )

    encoder_params = list(model.encoder.parameters())
    decoder_params = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": float(config.get("lr_encoder", 1e-5))},
            {"params": decoder_params, "lr": float(config.get("lr_decoder", 1e-4))},
        ]
    )

    if config.get("load_optimizer_state", False) and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint.")
        except Exception as exc:
            print(f"Could not load optimizer state, starting fresh: {exc}")

    log_path = output_dir / "training_log.csv"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    best_top1_checkpoint_path = output_dir / "best_top1_checkpoint.pt"
    dataset_report_path = output_dir / "dataset_report.json"
    with dataset_report_path.open("w", encoding="utf-8") as handle:
        json.dump(split_report, handle, indent=2)

    rows = []
    best_valid_loss = float("inf")
    best_valid_top1 = float("-inf")
    stale_epochs = 0
    start_epoch = 1
    if "epoch" in checkpoint:
        print(f"Starting fine-tuning from checkpoint epoch {checkpoint['epoch']}", flush=True)

    if log_path.exists():
        try:
            rows = pd.read_csv(log_path).to_dict(orient="records")
        except Exception:
            rows = []

    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 1e-4))
    save_every = int(config.get("save_every", 1))
    beam_size = int(config.get("beam_size", 5))
    beam_eval_examples = int(config.get("beam_eval_examples", valid_eval_examples or 0))
    gradient_clip = float(config.get("gradient_clip", 1.0))
    monitor_metric = str(config.get("monitor_metric", "valid_loss"))
    monitor_mode = config.get("monitor_mode", None)
    if monitor_mode is None:
        monitor_mode = "max" if any(token in monitor_metric.lower() for token in ("top", "acc", "exact", "coverage")) else "min"
    monitor_mode = str(monitor_mode).lower()
    if monitor_mode not in {"min", "max"}:
        raise ValueError("--monitor_mode must be 'min' or 'max'.")
    best_monitor_value = float("-inf") if monitor_mode == "max" else float("inf")

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, device, gradient_clip=gradient_clip)
        current_valid_loader = build_valid_loader(seed + epoch if random_eval_subset else seed)
        print("Running validation loss...", flush=True)
        valid_loss_metrics = evaluate_loss(model, current_valid_loader, device, desc="valid-loss")
        valid_metrics = {
            "valid_loss": valid_loss_metrics["valid_loss"],
            "exact_match": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "top10": 0.0,
        }
        if beam_eval_examples > 0:
            print("Running validation beam metrics...", flush=True)
            valid_metrics.update(
                evaluate_beam(model, current_valid_loader, device, beam_size=beam_size, max_examples=beam_eval_examples)
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["valid_loss"],
            "valid_top1": valid_metrics["top1"],
            "valid_top3": valid_metrics["top3"],
            "valid_top5": valid_metrics["top5"],
            "valid_top10": valid_metrics["top10"],
            "valid_exact_match": valid_metrics["exact_match"],
            "source_coverage": get_source_tokenizer(train_dataset).coverage(),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        print(json.dumps(row, indent=2))

        current_monitor_value = row.get(monitor_metric)
        if current_monitor_value is None:
            raise KeyError(
                f"Configured monitor_metric='{monitor_metric}' is not present in the logged row. "
                f"Available keys: {sorted(row.keys())}"
            )
        if monitor_mode == "max":
            improved = current_monitor_value > (best_monitor_value + min_delta)
        else:
            improved = current_monitor_value < (best_monitor_value - min_delta)
        if improved:
            best_monitor_value = current_monitor_value
            best_valid_loss = valid_metrics["valid_loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "encoder_config": encoder_config.__dict__,
            "target_tokenizer": target_tokenizer.token_to_id,
            "dataset_report": split_report,
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "best_monitor_metric": monitor_metric,
            "best_monitor_value": best_monitor_value,
            "stale_epochs": stale_epochs,
            "train_row": row,
            "args": config,
        }
        torch.save(checkpoint_state, last_checkpoint_path)
        if epoch % save_every == 0:
            torch.save(checkpoint_state, output_dir / f"checkpoint_epoch_{epoch}.pt")
        if improved:
            torch.save(checkpoint_state, best_checkpoint_path)
        if valid_metrics["top1"] > best_valid_top1:
            best_valid_top1 = valid_metrics["top1"]
            checkpoint_state["best_valid_top1"] = best_valid_top1
            torch.save(checkpoint_state, best_top1_checkpoint_path)

        if stale_epochs >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best {monitor_metric}={best_monitor_value:.6f}"
            )
            break


if __name__ == "__main__":
    main()
