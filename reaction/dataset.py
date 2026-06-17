from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reaction.tokenizers import FingerprintTokenizer, SmilesTokenizer


def _read_table(path):
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, header=None)
    return pd.read_csv(path)


def _infer_columns(frame):
    columns = list(frame.columns)
    lowered = {str(col).lower(): col for col in columns}
    for candidate in ["reaction", "rxn", "reaction_smiles"]:
        if candidate in lowered:
            return lowered[candidate], None
    reactant_col = None
    product_col = None
    reagent_col = None
    for candidate in ["reactants", "reactant", "source"]:
        if candidate in lowered:
            reactant_col = lowered[candidate]
            break
    for candidate in ["reagents", "reagent"]:
        if candidate in lowered:
            reagent_col = lowered[candidate]
            break
    for candidate in ["products", "product", "target", "smiles"]:
        if candidate in lowered:
            product_col = lowered[candidate]
            break
    return (reactant_col, reagent_col, product_col)


def _split_reaction_smiles(reaction):
    reaction = compact_smiles(reaction)
    if ">>" in reaction:
        left, product = reaction.split(">>", 1)
        return left, "", product
    parts = reaction.split(">")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return reaction, "", ""


def canonicalize_smiles(smiles):
    smiles = compact_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compact_smiles(smiles):
    return "".join(str(smiles).split())


def load_reaction_records(path, reaction_column=None, reactants_column=None, reagents_column=None, product_column=None):
    frame = _read_table(path)
    if frame.shape[1] == 1 and reaction_column is None and reactants_column is None and product_column is None:
        records = []
        for value in frame.iloc[:, 0].astype(str).tolist():
            reactants, reagents, product = _split_reaction_smiles(value)
            records.append({"reactants": reactants, "reagents": reagents, "product": product})
        return records

    if reaction_column is None and reactants_column is None and product_column is None:
        inferred = _infer_columns(frame)
        if isinstance(inferred[0], str) and inferred[1] is None:
            reaction_column = inferred[0]
        else:
            reactants_column, reagents_column, product_column = inferred

    records = []
    if reaction_column is not None:
        for value in frame[reaction_column].astype(str).tolist():
            reactants, reagents, product = _split_reaction_smiles(value)
            reactants = compact_smiles(reactants)
            reagents = compact_smiles(reagents)
            product = compact_smiles(product)
            records.append({"reactants": reactants, "reagents": reagents, "product": product})
    else:
        reactants_values = frame[reactants_column].astype(str).tolist() if reactants_column is not None else [""] * len(frame)
        reagents_values = frame[reagents_column].astype(str).tolist() if reagents_column is not None else [""] * len(frame)
        products_values = frame[product_column].astype(str).tolist() if product_column is not None else [""] * len(frame)
        for reactants, reagents, product in zip(reactants_values, reagents_values, products_values):
            reactants = compact_smiles(reactants)
            reagents = compact_smiles(reagents)
            product = compact_smiles(product)
            records.append({"reactants": reactants, "reagents": reagents, "product": product})
    return records


def load_parallel_reaction_records(src_path, tgt_path):
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)
    with src_path.open("r", encoding="utf-8") as src_handle, tgt_path.open("r", encoding="utf-8") as tgt_handle:
        src_lines = src_handle.readlines()
        tgt_lines = tgt_handle.readlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Mismatched source/target lengths: {len(src_lines)} vs {len(tgt_lines)}")
    records = []
    for source_line, target_line in zip(src_lines, tgt_lines):
        source = compact_smiles(source_line)
        product = compact_smiles(target_line)
        reactants, reagents, _product = _split_reaction_smiles(source)
        if not _product:
            _product = product
        records.append({"reactants": reactants, "reagents": reagents, "product": product})
    return records


class FingerprintReactionDataset(Dataset):
    def __init__(
        self,
        data_path,
        fingerprint_vocab_path,
        smiles_tokenizer=None,
        max_source_len=256,
        max_target_len=256,
        reaction_column=None,
        reactants_column=None,
        reagents_column=None,
        product_column=None,
        canonicalize_targets=True,
        records=None,
    ):
        if records is not None:
            self.records = list(records)
        else:
            self.records = load_reaction_records(
                data_path,
                reaction_column=reaction_column,
                reactants_column=reactants_column,
                reagents_column=reagents_column,
                product_column=product_column,
            )
        self.source_tokenizer = FingerprintTokenizer(fingerprint_vocab_path)
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.canonicalize_targets = canonicalize_targets
        if smiles_tokenizer is None:
            target_texts = []
            for record in self.records:
                product = record["product"]
                if canonicalize_targets:
                    product = canonicalize_smiles(product) or product
                target_texts.append(product)
            self.target_tokenizer = SmilesTokenizer.build(target_texts)
        else:
            self.target_tokenizer = smiles_tokenizer

    def __len__(self):
        return len(self.records)

    def _encode_source(self, record):
        source_ids = [self.source_tokenizer.cls_token_id]
        source_ids.extend(self.source_tokenizer.encode_reaction_context(record["reactants"], record["reagents"]))
        source_ids.append(self.source_tokenizer.sep_token_id)
        source_ids = source_ids[:self.max_source_len]
        source_mask = [1] * len(source_ids)
        return source_ids, source_mask

    def _encode_target(self, record):
        product = record["product"]
        if self.canonicalize_targets:
            product = canonicalize_smiles(product) or product
        target_ids = self.target_tokenizer.encode(product, add_bos=True, add_eos=True)
        target_ids = target_ids[:self.max_target_len]
        if len(target_ids) < 2:
            target_ids = [self.target_tokenizer.bos_id, self.target_tokenizer.eos_id]
        target_input_ids = target_ids[:-1]
        target_output_ids = target_ids[1:]
        return target_input_ids, target_output_ids, product

    def __getitem__(self, index):
        record = self.records[index]
        source_ids, source_mask = self._encode_source(record)
        target_input_ids, target_output_ids, product = self._encode_target(record)
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.long),
            "target_input_ids": torch.tensor(target_input_ids, dtype=torch.long),
            "target_output_ids": torch.tensor(target_output_ids, dtype=torch.long),
            "source_smiles": record["reactants"],
            "target_smiles": product,
        }

    @classmethod
    def from_records(
        cls,
        records,
        fingerprint_vocab_path,
        smiles_tokenizer=None,
        max_source_len=256,
        max_target_len=256,
        canonicalize_targets=True,
    ):
        return cls(
            data_path=None,
            fingerprint_vocab_path=fingerprint_vocab_path,
            smiles_tokenizer=smiles_tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            canonicalize_targets=canonicalize_targets,
            records=records,
        )


def collate_reaction_batch(batch, pad_source_id=0, pad_target_id=0):
    source_ids = torch.nn.utils.rnn.pad_sequence(
        [item["source_ids"] for item in batch], batch_first=True, padding_value=pad_source_id
    )
    source_mask = torch.nn.utils.rnn.pad_sequence(
        [item["source_mask"] for item in batch], batch_first=True, padding_value=0
    )
    target_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["target_input_ids"] for item in batch], batch_first=True, padding_value=pad_target_id
    )
    target_output_ids = torch.nn.utils.rnn.pad_sequence(
        [item["target_output_ids"] for item in batch], batch_first=True, padding_value=pad_target_id
    )
    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_input_ids": target_input_ids,
        "target_output_ids": target_output_ids,
        "source_smiles": [item["source_smiles"] for item in batch],
        "target_smiles": [item["target_smiles"] for item in batch],
    }
