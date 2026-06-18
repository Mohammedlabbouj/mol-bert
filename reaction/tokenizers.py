import json
import pickle
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature import mol2alt_sentence


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\[\]]+\]|Br?|Cl?|Si?|Se?|Na?|Li?|Al?|Ca?|@@?|==|!=|<=|>=|=>|=<|"
    r"\%\d{2}|\.|\(|\)|\[|\]|\{|\}|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\d|"
    r"[A-Za-z])"
)


class FingerprintTokenizer:
    def __init__(self, vocab_path, unk_token_id=1, cls_token_id=2, sep_token_id=3, pad_token_id=0, mask_token_id=4):
        self.vocab_path = str(vocab_path)
        with open(self.vocab_path, "rb") as handle:
            self.vocab = pickle.load(handle)
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.total_tokens = 0
        self.oov_tokens = 0

    @property
    def vocab_size(self):
        return len(self.vocab)

    def canonicalize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _morgan_tokens_for_smiles(self, smiles):
        canonical = self.canonicalize_smiles(smiles)
        if canonical is None:
            return []
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            return []
        radius0 = mol2alt_sentence(mol, 0)
        radius1 = mol2alt_sentence(mol, 1)
        if len(radius0) > 0 and len(radius1) >= 2 * len(radius0):
            return [str(radius1[len(radius0) + i]) + str(radius0[i]) for i in range(len(radius0))]
        return [str(token) for token in radius1 if token is not None]

    def encode_smiles(self, smiles):
        tokens = self._morgan_tokens_for_smiles(smiles)
        encoded = []
        for token in tokens:
            self.total_tokens += 1
            token_id = self.vocab.get(token, self.unk_token_id)
            if token_id == self.unk_token_id and token not in self.vocab:
                self.oov_tokens += 1
            encoded.append(token_id)
        return encoded

    def encode_reaction_context(self, reactants, reagents=None):
        parts = []
        if reactants:
            parts.append(reactants)
        if reagents:
            parts.append(reagents)
        source = ".".join(parts)
        encoded = []
        for smiles in source.split("."):
            smiles = smiles.strip()
            if smiles:
                encoded.extend(self.encode_smiles(smiles))
        return encoded

    def coverage(self):
        if self.total_tokens == 0:
            return 1.0
        return 1.0 - (self.oov_tokens / float(self.total_tokens))

    def coverage_report(self):
        return {
            "total_tokens": self.total_tokens,
            "oov_tokens": self.oov_tokens,
            "coverage": self.coverage(),
        }


class SmilesTokenizer:
    def __init__(self, token_to_id=None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.token_to_id = token_to_id or {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self):
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self):
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self):
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @staticmethod
    def tokenize(smiles):
        return [token for token in SMILES_TOKEN_PATTERN.findall(smiles) if token]

    @classmethod
    def build(cls, smiles_list):
        token_set = set()
        for smiles in smiles_list:
            for token in cls.tokenize(smiles):
                token_set.add(token)
        token_to_id = {token: idx for idx, token in enumerate(["<pad>", "<unk>", "<bos>", "<eos>"])}
        for token in sorted(token_set):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(token_to_id=data["token_to_id"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"token_to_id": self.token_to_id}, handle, indent=2, sort_keys=True)

    def encode(self, smiles, add_bos=True, add_eos=True):
        tokens = self.tokenize(smiles)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for idx in ids:
            if isinstance(idx, list):
                idx = idx[0]
            token = self.id_to_token.get(int(idx), self.unk_token)
            if token == self.eos_token:
                break
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)
