import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature import mol2alt_sentence
from models.bert_model import BertConfig, BertForPreTraining


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def canonicalize_smiles(smiles):
    smiles = "".join(str(smiles).split())
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_vocab(vocab_path):
    import pickle

    with open(vocab_path, "rb") as handle:
        return pickle.load(handle)


def build_identifiers(mol):
    radius0 = mol2alt_sentence(mol, 0)
    radius1 = mol2alt_sentence(mol, 1)
    if len(radius0) == 0 or len(radius1) < 2 * len(radius0):
        return [str(token) for token in radius1 if token is not None]
    return [str(radius1[len(radius0) + i]) + str(radius0[i]) for i in range(len(radius0))]


class ZincMlmDataset(Dataset):
    def __init__(self, smiles_path, vocab_path, seq_len=100, on_memory=True, mask_prob=0.15, random_token_prob=0.1, keep_token_prob=0.1):
        self.smiles_path = Path(smiles_path)
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob
        self.keep_token_prob = keep_token_prob
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.ident_dict = load_vocab(vocab_path)
        if on_memory:
            with self.smiles_path.open("r", encoding="utf-8") as handle:
                self.lines = [line.strip() for line in handle if line.strip()]
            self.corpus_lines = len(self.lines)
        else:
            self.file = self.smiles_path.open("r", encoding="utf-8")
            self.corpus_lines = sum(1 for _ in self.file)
            self.file.close()
            self.file = self.smiles_path.open("r", encoding="utf-8")

    def __len__(self):
        return self.corpus_lines

    def _get_line(self, idx):
        if self.on_memory:
            return self.lines[idx]
        self.file.seek(0)
        for current, line in enumerate(self.file):
            if current == idx:
                return line.strip()
        raise IndexError(idx)

    def _tokenize(self, smiles):
        smiles = canonicalize_smiles(smiles)
        if smiles is None:
            return []
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return build_identifiers(mol)

    def _mask_tokens(self, tokens):
        input_ids = []
        labels = []
        for token in tokens:
            token_id = self.ident_dict.get(token, self.unk_index)
            labels.append(-100)
            if random.random() < self.mask_prob:
                prob = random.random()
                if prob < 1.0 - self.random_token_prob - self.keep_token_prob:
                    input_ids.append(self.mask_index)
                    labels[-1] = token_id
                elif prob < 1.0 - self.keep_token_prob:
                    input_ids.append(random.randrange(len(self.ident_dict)))
                    labels[-1] = token_id
                else:
                    input_ids.append(token_id)
                    labels[-1] = token_id
            else:
                input_ids.append(token_id)
        return input_ids, labels

    def __getitem__(self, idx):
        line = self._get_line(idx)
        smiles = line.split()[0]
        tokens = self._tokenize(smiles)
        input_ids, labels = self._mask_tokens(tokens)
        input_ids = [self.cls_index] + input_ids + [self.sep_index]
        labels = [self.cls_index] + labels + [self.sep_index]
        input_ids = input_ids[:self.seq_len]
        labels = labels[:self.seq_len]
        return {
            "bert_input": torch.tensor(input_ids, dtype=torch.long),
            "bert_label": torch.tensor(labels, dtype=torch.long),
        }


class Pretrainer:
    def __init__(self, config):
        self.config = config
        cuda_condition = torch.cuda.is_available() and config["device"].startswith("cuda")
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.vocab_size = len(load_vocab(config["vocab_path"]))
        bertconfig = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        )
        self.model = BertForPreTraining(bertconfig).to(self.device)
        self.positional_enc = self.init_positional_encoding(bertconfig.hidden_size, config["max_seq_len"]).unsqueeze(0).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    @staticmethod
    def init_positional_encoding(hidden_dim, max_seq_len):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
            if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)
        ])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        return torch.from_numpy(position_enc).float()

    def make_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["num_workers"],
            collate_fn=lambda batch: batch,
        )

    @staticmethod
    def pad_batch(batch):
        inputs = torch.nn.utils.rnn.pad_sequence([item["bert_input"] for item in batch], batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence([item["bert_label"] for item in batch], batch_first=True, padding_value=-100)
        return {"bert_input": inputs, "bert_label": labels}

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        for batch in tqdm(loader, desc="train" if train else "valid"):
            batch = self.pad_batch(batch)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            pos = self.positional_enc[:, :batch["bert_input"].size(1), :]
            mlm_preds, _ = self.model(input_ids=batch["bert_input"], positional_enc=pos)
            loss = self.loss_fn(mlm_preds.view(-1, mlm_preds.size(-1)), batch["bert_label"].view(-1))
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            preds = torch.argmax(mlm_preds, dim=-1)
            mask = batch["bert_label"] != -100
            acc = ((preds == batch["bert_label"]) & mask).float().sum() / (mask.float().sum() + 1e-8)
            total_loss += loss.item()
            total_acc += acc.item()
            total_batches += 1
        return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)

    def save_checkpoint(self, output_dir, epoch):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"bert.model.epoch.{epoch}"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "epoch": epoch,
            },
            save_path,
        )
        return save_path

    def _load_tensor_with_resize(self, current_tensor, checkpoint_tensor, key):
        if current_tensor.shape == checkpoint_tensor.shape:
            current_tensor.copy_(checkpoint_tensor)
            return True
        if current_tensor.dim() != checkpoint_tensor.dim():
            return False
        if current_tensor.dim() == 1:
            rows = min(current_tensor.size(0), checkpoint_tensor.size(0))
            current_tensor[:rows].copy_(checkpoint_tensor[:rows])
            return True
        if current_tensor.shape[1:] == checkpoint_tensor.shape[1:]:
            rows = min(current_tensor.size(0), checkpoint_tensor.size(0))
            current_tensor[:rows].copy_(checkpoint_tensor[:rows])
            return True
        return False

    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = self.model.state_dict()
        loaded = []
        skipped = []
        for key, value in checkpoint["model_state_dict"].items():
            if key not in model_state:
                skipped.append(key)
                continue
            current_value = model_state[key]
            if current_value.shape == value.shape:
                current_value.copy_(value)
                loaded.append(key)
                continue
            if key in {
                "bert.embeddings.word_embeddings.weight",
                "cls.predictions.decoder.weight",
                "cls.predictions.bias",
            } and self._load_tensor_with_resize(current_value, value, key):
                loaded.append(key)
                continue
            skipped.append(key)
        self.model.load_state_dict(model_state, strict=False)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:
                print(f"Skipping optimizer state load: {exc}")
        if skipped:
            print(
                f"Loaded checkpoint with partial vocab resize. "
                f"Loaded {len(loaded)} tensors, skipped {len(skipped)} tensors."
            )
        return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Mol-BERT on ZINC-20 with MLM")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config_zinc20.json")))
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--clean_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    config["clean_only"] = bool(config.get("clean_only", False) or args.clean_only)
    config_path = Path(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pretrain_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    if not Path(config["clean_data_path"]).exists():
        from pre_training.clean_zinc20 import clean_zinc20

        stats = clean_zinc20(config["raw_data_path"], config["clean_data_path"])
        print(stats)

    if config["clean_only"]:
        return

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    dataset = ZincMlmDataset(
        config["clean_data_path"],
        config["vocab_path"],
        seq_len=config["max_seq_len"],
        on_memory=True,
        mask_prob=config["mask_prob"],
        random_token_prob=config["random_token_prob"],
        keep_token_prob=config["keep_token_prob"],
    )
    train_len = int(len(dataset) * 0.95)
    valid_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(config["seed"])
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len], generator=generator)

    trainer = Pretrainer(config)
    start_epoch = 0
    best_valid_loss = float("inf")
    stale_epochs = 0
    last_checkpoint = output_dir / "last_checkpoint.pt"
    best_checkpoint = output_dir / "best_checkpoint.pt"

    if args.resume_checkpoint:
        checkpoint = trainer.load_checkpoint(args.resume_checkpoint, load_optimizer=False)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if config.get("reset_best_on_resume", False):
            best_valid_loss = float("inf")
            stale_epochs = 0
            print("Resetting best_valid_loss for a new adaptation stage.")
        else:
            best_valid_loss = float(checkpoint.get("best_valid_loss", best_valid_loss))
            stale_epochs = int(checkpoint.get("stale_epochs", 0))
    elif config.get("auto_resume", True) and last_checkpoint.exists():
        checkpoint = trainer.load_checkpoint(last_checkpoint, load_optimizer=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", best_valid_loss))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        print(f"Resumed from {last_checkpoint} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, config["epochs"]):
        train_loss, train_acc = trainer.run_epoch(trainer.make_loader(train_dataset, shuffle=True), train=True)
        valid_loss, valid_acc = trainer.run_epoch(trainer.make_loader(valid_dataset, shuffle=False), train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(row)
        checkpoint_state = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "stale_epochs": stale_epochs,
        }
        torch.save(checkpoint_state, last_checkpoint)
        trainer.save_checkpoint(output_dir, epoch)

        if valid_loss < (best_valid_loss - config["min_delta"]):
            best_valid_loss = valid_loss
            stale_epochs = 0
            checkpoint_state["best_valid_loss"] = best_valid_loss
            checkpoint_state["stale_epochs"] = stale_epochs
            torch.save(checkpoint_state, best_checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best valid_loss={best_valid_loss:.6f}"
                )
                break


if __name__ == "__main__":
    main()
