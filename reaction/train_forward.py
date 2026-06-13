import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.bert_model import BertConfig
from reaction.dataset import (
    FingerprintReactionDataset,
    canonicalize_smiles,
    collate_reaction_batch,
    load_parallel_reaction_records,
    load_reaction_records,
)
from reaction.model import FingerprintReactionModel, build_sinusoidal_positional_encoding
from reaction.tokenizers import SmilesTokenizer


def _choose_attention_heads(hidden_size, preferred_heads):
    if preferred_heads and hidden_size % preferred_heads == 0:
        return preferred_heads
    for candidate in range(min(hidden_size, preferred_heads or hidden_size), 1, -1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def infer_encoder_config(checkpoint_path, vocab_size, default_hidden_size, default_layers, default_heads, default_ff):
    if not checkpoint_path:
        return BertConfig(
            vocab_size=vocab_size,
            hidden_size=default_hidden_size,
            num_hidden_layers=default_layers,
            num_attention_heads=default_heads,
            intermediate_size=default_ff,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    embedding_key = None
    for key in checkpoint:
        if key.endswith("embeddings.word_embeddings.weight"):
            embedding_key = key
            break
    hidden_size = default_hidden_size
    if embedding_key is not None:
        hidden_size = checkpoint[embedding_key].shape[1]
    layer_indices = set()
    for key in checkpoint:
        if "encoder.layer." in key:
            parts = key.split("encoder.layer.", 1)[1].split(".")
            if parts and parts[0].isdigit():
                layer_indices.add(int(parts[0]))
    num_hidden_layers = len(layer_indices) if layer_indices else default_layers
    attention_key = None
    for key in checkpoint:
        if key.endswith("attention.self.query.weight"):
            attention_key = key
            break
    num_attention_heads = _choose_attention_heads(hidden_size, default_heads)
    ff_key = None
    for key in checkpoint:
        if key.endswith("intermediate.dense.weight"):
            ff_key = key
            break
    intermediate_size = default_ff
    if ff_key is not None:
        intermediate_size = checkpoint[ff_key].shape[0]
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )


def build_source_positional_encoding(batch_size, seq_len, hidden_size, device):
    encoding = build_sinusoidal_positional_encoding(seq_len, hidden_size, device=device)
    return encoding.expand(batch_size, -1, -1)


def prepare_datasets(args):
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
        target_tokenizer = SmilesTokenizer.load(args.target_tokenizer_path) if args.target_tokenizer_path else SmilesTokenizer.build(
            [
                canonicalize_smiles(record["product"]) or record["product"]
                for record in train_records
            ]
        )
        if args.target_tokenizer_path is None:
            target_tokenizer.save(args.output_dir / "target_tokenizer.json")
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
        return train_dataset, valid_dataset, test_dataset, target_tokenizer
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
        test_len = total - train_len - valid_len
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(total, generator=generator).tolist()
        train_indices = permutation[:train_len]
        valid_indices = permutation[train_len:train_len + valid_len]
        test_indices = permutation[train_len + valid_len:]
        train_records = [records[index] for index in train_indices]
        valid_records = [records[index] for index in valid_indices]
        test_records = [records[index] for index in test_indices]
        target_tokenizer = SmilesTokenizer.load(args.target_tokenizer_path) if args.target_tokenizer_path else SmilesTokenizer.build(
            [
                canonicalize_smiles(record["product"]) or record["product"]
                for record in train_records
            ]
        )
        if args.target_tokenizer_path is None:
            target_tokenizer.save(args.output_dir / "target_tokenizer.json")
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
        return train_dataset, valid_dataset, test_dataset, target_tokenizer
    raise ValueError("Provide either train/valid/test files or train_size and valid_size split fractions.")


def make_loader(dataset, batch_size, shuffle, pad_source_id, pad_target_id, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_reaction_batch(batch, pad_source_id=pad_source_id, pad_target_id=pad_target_id),
    )


@torch.no_grad()
def evaluate(model, loader, device, max_top_k=10):
    model.eval()
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    exact_matches = 0
    oov_stats = None
    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        batch_size = source_ids.size(0)
        total += batch_size
        for index in range(batch_size):
            src = source_ids[index : index + 1]
            src_mask = source_mask[index : index + 1]
            pos = build_source_positional_encoding(1, src.size(1), model.hidden_size, device)
            beams = model.beam_search(src, pos, source_mask=src_mask, beam_size=max_top_k, max_length=loader.dataset.dataset.max_target_len if hasattr(loader.dataset, "dataset") else loader.dataset.max_target_len)
            target = batch["target_smiles"][index]
            candidates = []
            for tokens, _score in beams:
                decoded = loader.dataset.dataset.target_tokenizer.decode(tokens) if hasattr(loader.dataset, "dataset") else loader.dataset.target_tokenizer.decode(tokens)
                canonical = canonicalize_smiles(decoded) or decoded
                candidates.append(canonical)
            canonical_target = canonicalize_smiles(target) or target
            if candidates and candidates[0] == canonical_target:
                exact_matches += 1
            for k in top_hits:
                if canonical_target in candidates[:k]:
                    top_hits[k] += 1
        if oov_stats is None and hasattr(loader.dataset, "dataset"):
            oov_stats = loader.dataset.dataset.source_tokenizer.coverage_report()
        elif oov_stats is None:
            oov_stats = loader.dataset.source_tokenizer.coverage_report()
    metrics = {
        "exact_match": exact_matches / max(total, 1),
        "top1": top_hits[1] / max(total, 1),
        "top3": top_hits[3] / max(total, 1),
        "top5": top_hits[5] / max(total, 1),
        "top10": top_hits[10] / max(total, 1),
        "oov_total_tokens": oov_stats["total_tokens"] if oov_stats else 0,
        "oov_tokens": oov_stats["oov_tokens"] if oov_stats else 0,
        "source_coverage": oov_stats["coverage"] if oov_stats else 1.0,
    }
    return metrics


def train_epoch(model, loader, optimizer, device, gradient_clip=1.0):
    model.train()
    running_loss = 0.0
    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_output_ids = batch["target_output_ids"].to(device)
        pos = build_source_positional_encoding(source_ids.size(0), source_ids.size(1), model.hidden_size, device)
        outputs = model(
            source_ids=source_ids,
            source_positional_enc=pos,
            target_input_ids=target_input_ids,
            source_mask=source_mask,
            target_output_ids=target_output_ids,
        )
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fingerprint-to-SMILES forward reaction predictor with MolBERT encoder")
    parser.add_argument("--data_path", type=str, default=None, help="Single reaction file or table path")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing src-train.txt/tgt-train.txt style splits")
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
    parser.add_argument("--encoder_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("reaction_output"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1200)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=6)
    parser.add_argument("--decoder_ff_size", type=int, default=0, help="Set 0 to use a safe hidden_size*4 default")
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--disable_target_canonicalization", action="store_true")
    parser.add_argument("--tie_decoder_weights", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset, valid_dataset, test_dataset, target_tokenizer = prepare_datasets(args)
    encoder_config = infer_encoder_config(
        args.encoder_checkpoint,
        vocab_size=train_dataset.dataset.source_tokenizer.vocab_size if hasattr(train_dataset, "dataset") else train_dataset.source_tokenizer.vocab_size,
        default_hidden_size=args.hidden_size,
        default_layers=args.num_hidden_layers,
        default_heads=args.num_attention_heads,
        default_ff=args.intermediate_size,
    )
    if hasattr(train_dataset, "dataset"):
        target_pad_id = train_dataset.dataset.target_tokenizer.pad_id
        target_bos_id = train_dataset.dataset.target_tokenizer.bos_id
        target_eos_id = train_dataset.dataset.target_tokenizer.eos_id
    else:
        target_pad_id = train_dataset.target_tokenizer.pad_id
        target_bos_id = train_dataset.target_tokenizer.bos_id
        target_eos_id = train_dataset.target_tokenizer.eos_id

    model = FingerprintReactionModel(
        encoder_config=encoder_config,
        target_vocab_size=target_tokenizer.vocab_size,
        target_pad_id=target_pad_id,
        target_bos_id=target_bos_id,
        target_eos_id=target_eos_id,
        encoder_checkpoint=args.encoder_checkpoint,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ff_size=args.decoder_ff_size or None,
        decoder_dropout=args.decoder_dropout,
        max_target_len=args.max_target_len,
        tie_decoder_weights=args.tie_decoder_weights,
    ).to(args.device)

    encoder_params = list(model.encoder.parameters())
    decoder_params = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": decoder_params, "lr": args.lr_decoder},
        ]
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pad_source_id=train_dataset.dataset.source_tokenizer.pad_token_id if hasattr(train_dataset, "dataset") else train_dataset.source_tokenizer.pad_token_id,
        pad_target_id=target_pad_id,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pad_source_id=valid_dataset.dataset.source_tokenizer.pad_token_id if hasattr(valid_dataset, "dataset") else valid_dataset.source_tokenizer.pad_token_id,
        pad_target_id=target_pad_id,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pad_source_id=test_dataset.dataset.source_tokenizer.pad_token_id if hasattr(test_dataset, "dataset") else test_dataset.source_tokenizer.pad_token_id,
        pad_target_id=target_pad_id,
        num_workers=args.num_workers,
    )

    log_path = args.output_dir / "metrics.csv"
    rows = []
    best_valid = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, gradient_clip=args.gradient_clip)
        valid_metrics = evaluate(model, valid_loader, args.device)
        test_metrics = evaluate(model, test_loader, args.device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_top1": valid_metrics["top1"],
            "valid_top3": valid_metrics["top3"],
            "valid_top5": valid_metrics["top5"],
            "valid_top10": valid_metrics["top10"],
            "valid_exact_match": valid_metrics["exact_match"],
            "test_top1": test_metrics["top1"],
            "test_top3": test_metrics["top3"],
            "test_top5": test_metrics["top5"],
            "test_top10": test_metrics["top10"],
            "test_exact_match": test_metrics["exact_match"],
            "source_coverage": train_dataset.dataset.source_tokenizer.coverage() if hasattr(train_dataset, "dataset") else train_dataset.source_tokenizer.coverage(),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        print(json.dumps(row, indent=2))
        if valid_metrics["top1"] > best_valid:
            best_valid = valid_metrics["top1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_config": encoder_config.__dict__,
                    "target_tokenizer": target_tokenizer.token_to_id,
                    "source_vocab_coverage": train_dataset.dataset.source_tokenizer.coverage_report() if hasattr(train_dataset, "dataset") else train_dataset.source_tokenizer.coverage_report(),
                },
                args.output_dir / "best_model.pt",
            )
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_config": encoder_config.__dict__,
                    "target_tokenizer": target_tokenizer.token_to_id,
                },
                args.output_dir / f"checkpoint_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    main()
