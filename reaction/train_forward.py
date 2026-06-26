import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

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


def progress(iterable, total=None, desc=None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=True)


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


def unwrap_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def get_source_tokenizer(dataset):
    return unwrap_dataset(dataset).source_tokenizer


def get_target_tokenizer(dataset):
    return unwrap_dataset(dataset).target_tokenizer


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


@torch.no_grad()
def evaluate_loss(model, loader, device, desc="valid-loss"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    dataset = unwrap_dataset(loader.dataset)
    iterator = progress(loader, total=len(loader), desc=desc)
    for batch in iterator:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_output_ids = batch["target_output_ids"].to(device)
        batch_size = source_ids.size(0)
        pos = build_source_positional_encoding(batch_size, source_ids.size(1), model.hidden_size, device)
        outputs = model(
            source_ids=source_ids,
            source_positional_enc=pos,
            target_input_ids=target_input_ids,
            source_mask=source_mask,
            target_output_ids=target_output_ids,
        )
        logits = outputs["logits"]
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_output_ids.reshape(-1),
            ignore_index=model.target_pad_id,
            reduction="sum",
        )
        non_pad_tokens = target_output_ids.ne(model.target_pad_id).sum().item()
        total_loss += float(loss_sum.item())
        total_tokens += int(non_pad_tokens)
        if tqdm is not None and iterator is not loader:
            iterator.set_postfix(loss=total_loss / max(total_tokens, 1))
    return {
        "valid_loss": total_loss / max(total_tokens, 1),
        "source_coverage": dataset.source_tokenizer.coverage_report()["coverage"] if dataset else 1.0,
    }


@torch.no_grad()
def evaluate_beam(model, loader, device, beam_size=5, max_examples=128):
    model.eval()
    metric_ks = [k for k in (1, 3, 5, 10) if k <= beam_size]
    top_hits = {k: 0 for k in metric_ks}
    total = 0
    exact_matches = 0
    oov_stats = None
    dataset = unwrap_dataset(loader.dataset)
    iterator = progress(loader, total=len(loader), desc="valid-beam")
    for batch in iterator:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        batch_size = source_ids.size(0)
        total += batch_size
        for index in range(batch_size):
            if total - batch_size + index >= max_examples:
                break
            src = source_ids[index : index + 1]
            src_mask = source_mask[index : index + 1]
            src_pos = build_source_positional_encoding(1, src.size(1), model.hidden_size, device)
            beams = model.beam_search(
                src,
                src_pos,
                source_mask=src_mask,
                beam_size=beam_size,
                max_length=dataset.max_target_len,
            )
            target = batch["target_smiles"][index]
            candidates = []
            for tokens, _score in beams:
                decoded = dataset.target_tokenizer.decode(tokens)
                canonical = canonicalize_smiles(decoded) or decoded
                candidates.append(canonical)
            canonical_target = canonicalize_smiles(target) or target
            if candidates and candidates[0] == canonical_target:
                exact_matches += 1
            for k in top_hits:
                if canonical_target in candidates[:k]:
                    top_hits[k] += 1
        if oov_stats is None:
            oov_stats = dataset.source_tokenizer.coverage_report()
        if total >= max_examples:
            break
    metrics = {
        "exact_match": exact_matches / max(total, 1),
        "top1": top_hits.get(1, 0) / max(total, 1),
        "top3": top_hits.get(3, 0) / max(total, 1),
        "top5": top_hits.get(5, 0) / max(total, 1),
        "top10": top_hits.get(10, 0) / max(total, 1),
        "oov_total_tokens": oov_stats["total_tokens"] if oov_stats else 0,
        "oov_tokens": oov_stats["oov_tokens"] if oov_stats else 0,
        "source_coverage": oov_stats["coverage"] if oov_stats else 1.0,
    }
    return metrics


def make_record_subset_loader(dataset, batch_size, shuffle, pad_source_id, pad_target_id, num_workers=0, max_examples=None, random_subset=False, seed=42):
    records = _select_records(dataset.records, max_examples=max_examples, random_subset=random_subset, seed=seed)
    subset = FingerprintReactionDataset.from_records(
        records,
        fingerprint_vocab_path=dataset.source_tokenizer.vocab_path,
        smiles_tokenizer=dataset.target_tokenizer,
        max_source_len=dataset.max_source_len,
        max_target_len=dataset.max_target_len,
        canonicalize_targets=dataset.canonicalize_targets,
    )
    return make_loader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        pad_source_id=pad_source_id,
        pad_target_id=pad_target_id,
        num_workers=num_workers,
    )


def train_epoch(model, loader, optimizer, device, gradient_clip=1.0):
    model.train()
    running_loss = 0.0
    iterator = progress(enumerate(loader, start=1), total=len(loader), desc="train")
    for step, batch in iterator:
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
        if tqdm is not None:
            iterator.set_postfix(loss=running_loss / max(step, 1))
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
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--beam_eval_examples", type=int, default=128, help="Number of validation examples to run beam-search metrics on. Set 0 to skip beam metrics.")
    parser.add_argument("--valid_eval_examples", type=int, default=None, help="Limit validation evaluation to N examples.")
    parser.add_argument("--test_eval_examples", type=int, default=None, help="Limit test evaluation to N examples.")
    parser.add_argument("--random_eval_subset", action="store_true", help="Randomly sample validation/test evaluation examples instead of using the first N.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--disable_target_canonicalization", action="store_true")
    parser.add_argument("--tie_decoder_weights", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
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
        vocab_size=get_source_tokenizer(train_dataset).vocab_size,
        default_hidden_size=args.hidden_size,
        default_layers=args.num_hidden_layers,
        default_heads=args.num_attention_heads,
        default_ff=args.intermediate_size,
    )
    target_pad_id = get_target_tokenizer(train_dataset).pad_id
    target_bos_id = get_target_tokenizer(train_dataset).bos_id
    target_eos_id = get_target_tokenizer(train_dataset).eos_id

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

    if args.valid_eval_examples is not None:
        valid_loader = make_record_subset_loader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pad_source_id=get_source_tokenizer(valid_dataset).pad_token_id,
            pad_target_id=target_pad_id,
            num_workers=args.num_workers,
            max_examples=args.valid_eval_examples,
            random_subset=args.random_eval_subset,
            seed=args.seed,
        )
    if args.test_eval_examples is not None:
        test_loader = make_record_subset_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pad_source_id=get_source_tokenizer(test_dataset).pad_token_id,
            pad_target_id=target_pad_id,
            num_workers=args.num_workers,
            max_examples=args.test_eval_examples,
            random_subset=args.random_eval_subset,
            seed=args.seed,
        )

    log_path = args.output_dir / "training_log.csv"
    last_checkpoint_path = args.output_dir / "last_checkpoint.pt"
    best_checkpoint_path = args.output_dir / "best_checkpoint.pt"
    rows = []
    best_valid_loss = float("inf")
    stale_epochs = 0
    start_epoch = 1

    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, model, optimizer, map_location=args.device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", best_valid_loss))
        stale_epochs = int(checkpoint.get("stale_epochs", stale_epochs))
        print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}")
    elif args.auto_resume and last_checkpoint_path.exists():
        checkpoint = load_checkpoint(last_checkpoint_path, model, optimizer, map_location=args.device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", best_valid_loss))
        stale_epochs = int(checkpoint.get("stale_epochs", stale_epochs))
        print(f"Auto-resumed from {last_checkpoint_path} at epoch {start_epoch}")
    if (args.resume_checkpoint or (args.auto_resume and last_checkpoint_path.exists())) and log_path.exists():
        rows = pd.read_csv(log_path).to_dict(orient="records")

    target_tokenizer_obj = get_target_tokenizer(train_dataset)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, args.device, gradient_clip=args.gradient_clip)
        print("Running validation loss...", flush=True)
        valid_loss_metrics = evaluate_loss(model, valid_loader, args.device, desc="valid-loss")
        valid_metrics = {"valid_loss": valid_loss_metrics["valid_loss"], "exact_match": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0}
        test_metrics = {"valid_loss": 0.0, "exact_match": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0}
        if args.beam_eval_examples and args.beam_eval_examples > 0:
            print("Running validation beam metrics...", flush=True)
            beam_valid_metrics = evaluate_beam(model, valid_loader, args.device, beam_size=args.beam_size, max_examples=args.beam_eval_examples)
            valid_metrics.update(beam_valid_metrics)
        print("Running test loss...", flush=True)
        test_loss_metrics = evaluate_loss(model, test_loader, args.device, desc="test-loss")
        test_metrics["valid_loss"] = test_loss_metrics["valid_loss"]
        if args.beam_eval_examples and args.beam_eval_examples > 0:
            print("Running test beam metrics...", flush=True)
            beam_test_metrics = evaluate_beam(model, test_loader, args.device, beam_size=args.beam_size, max_examples=args.beam_eval_examples)
            test_metrics.update(beam_test_metrics)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["valid_loss"],
            "valid_top1": valid_metrics["top1"],
            "valid_top3": valid_metrics["top3"],
            "valid_top5": valid_metrics["top5"],
            "valid_top10": valid_metrics["top10"],
            "valid_exact_match": valid_metrics["exact_match"],
            "test_loss": test_metrics["valid_loss"],
            "test_top1": test_metrics["top1"],
            "test_top3": test_metrics["top3"],
            "test_top5": test_metrics["top5"],
            "test_top10": test_metrics["top10"],
            "test_exact_match": test_metrics["exact_match"],
            "source_coverage": get_source_tokenizer(train_dataset).coverage(),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        print(json.dumps(row, indent=2))
        improved = valid_metrics["valid_loss"] < (best_valid_loss - args.min_delta)
        if improved:
            best_valid_loss = valid_metrics["valid_loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "encoder_config": encoder_config.__dict__,
            "target_tokenizer": target_tokenizer_obj.token_to_id,
            "source_vocab_coverage": get_source_tokenizer(train_dataset).coverage_report(),
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "stale_epochs": stale_epochs,
            "train_row": row,
            "args": vars(args),
        }
        save_checkpoint(last_checkpoint_path, checkpoint_state)
        if epoch % args.save_every == 0:
            save_checkpoint(args.output_dir / f"checkpoint_epoch_{epoch}.pt", checkpoint_state)
        if improved:
            save_checkpoint(best_checkpoint_path, checkpoint_state)

        if stale_epochs >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best valid_loss={best_valid_loss:.6f}"
            )
            break


if __name__ == "__main__":
    main()
