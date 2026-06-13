import math

import torch
from torch import nn

from models.bert_model import BertConfig, BertModel


def build_sinusoidal_positional_encoding(length, hidden_size, device=None):
    position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / hidden_size)
    )
    encoding = torch.zeros(length, hidden_size, device=device)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding.unsqueeze(0)


def _shift_right(ids, bos_id):
    shifted = ids.new_full(ids.shape, bos_id)
    shifted[:, 1:] = ids[:, :-1]
    return shifted


class FingerprintReactionModel(nn.Module):
    def __init__(
        self,
        encoder_config,
        target_vocab_size,
        target_pad_id,
        target_bos_id,
        target_eos_id,
        encoder_checkpoint=None,
        decoder_layers=4,
        decoder_heads=8,
        decoder_ff_size=None,
        decoder_dropout=0.1,
        max_target_len=256,
        tie_decoder_weights=False,
    ):
        super(FingerprintReactionModel, self).__init__()
        if not isinstance(encoder_config, BertConfig):
            raise ValueError("encoder_config must be a BertConfig instance")
        self.encoder = BertModel(encoder_config)
        self.target_embedding = nn.Embedding(target_vocab_size, encoder_config.hidden_size, padding_idx=target_pad_id)
        self.target_positional_embedding = nn.Embedding(max_target_len, encoder_config.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=encoder_config.hidden_size,
                    nhead=decoder_heads,
                    dim_feedforward=decoder_ff_size or max(encoder_config.hidden_size * 4, 256),
                    dropout=decoder_dropout,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.dropout = nn.Dropout(decoder_dropout)
        self.output_projection = nn.Linear(encoder_config.hidden_size, target_vocab_size)
        self.target_pad_id = target_pad_id
        self.target_bos_id = target_bos_id
        self.target_eos_id = target_eos_id
        self.max_target_len = max_target_len
        self.hidden_size = encoder_config.hidden_size
        self.tie_decoder_weights = tie_decoder_weights
        if tie_decoder_weights:
            self.output_projection.weight = self.target_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=target_pad_id)
        if encoder_checkpoint:
            self.load_pretrained_encoder(encoder_checkpoint)

    def load_pretrained_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        encoder_state = {}
        for key, value in checkpoint.items():
            if key.startswith("bert."):
                encoder_state[key[len("bert."):]] = value
            elif key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = value
            elif key.startswith("embeddings.") or key.startswith("pooler."):
                encoder_state[key] = value
            elif key.startswith("word_embeddings.") or key.startswith("token_type_embeddings."):
                encoder_state[key] = value
        self.encoder.load_state_dict(encoder_state, strict=False)
        return self

    def freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def encode(self, source_ids, source_positional_enc, source_mask=None):
        if source_mask is None:
            source_mask = source_ids.ne(0).long()
        encoded_layers, _ = self.encoder(
            source_ids,
            source_positional_enc,
            attention_mask=source_mask,
            output_all_encoded_layers=False,
        )
        if isinstance(encoded_layers, list):
            memory = encoded_layers[-1]
        else:
            memory = encoded_layers
        return memory, source_mask

    def _decode_step(self, tgt_embeddings, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        output = tgt_embeddings.transpose(0, 1)
        memory = memory.transpose(0, 1)
        for layer in self.decoder_layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return output.transpose(0, 1)

    def decode(self, target_input_ids, memory, source_mask=None):
        batch_size, target_len = target_input_ids.size()
        device = target_input_ids.device
        positions = torch.arange(target_len, device=device).unsqueeze(0).expand(batch_size, target_len)
        tgt_embeddings = self.target_embedding(target_input_ids) + self.target_positional_embedding(positions)
        tgt_embeddings = self.dropout(tgt_embeddings)
        tgt_mask = torch.triu(torch.ones(target_len, target_len, device=device, dtype=torch.bool), diagonal=1)
        tgt_key_padding_mask = target_input_ids.eq(self.target_pad_id)
        memory_key_padding_mask = None if source_mask is None else source_mask.eq(0)
        decoded = self._decode_step(tgt_embeddings, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        logits = self.output_projection(decoded)
        return logits

    def forward(
        self,
        source_ids,
        source_positional_enc,
        target_input_ids,
        source_mask=None,
        target_output_ids=None,
    ):
        memory, source_mask = self.encode(source_ids, source_positional_enc, source_mask=source_mask)
        logits = self.decode(target_input_ids, memory, source_mask=source_mask)
        loss = None
        if target_output_ids is not None:
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target_output_ids.reshape(-1))
        return {"logits": logits, "loss": loss, "memory": memory}

    @torch.no_grad()
    def generate(self, source_ids, source_positional_enc, source_mask=None, max_length=128):
        memory, source_mask = self.encode(source_ids, source_positional_enc, source_mask=source_mask)
        batch_size = source_ids.size(0)
        device = source_ids.device
        generated = source_ids.new_full((batch_size, 1), self.target_bos_id)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_length - 1):
            logits = self.decode(generated, memory, source_mask=source_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(self.target_eos_id)
            if bool(finished.all()):
                break
        return generated

    @torch.no_grad()
    def beam_search(self, source_ids, source_positional_enc, source_mask=None, beam_size=10, max_length=128):
        if source_ids.size(0) != 1:
            raise ValueError("beam_search expects a single example at a time")
        memory, source_mask = self.encode(source_ids, source_positional_enc, source_mask=source_mask)
        device = source_ids.device
        beams = [([self.target_bos_id], 0.0)]
        for _ in range(max_length - 1):
            candidates = []
            for tokens, score in beams:
                if tokens[-1] == self.target_eos_id:
                    candidates.append((tokens, score))
                    continue
                current = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = self.decode(current, memory, source_mask=source_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                top_scores, top_tokens = torch.topk(log_probs, k=beam_size)
                for value, token in zip(top_scores.tolist(), top_tokens.tolist()):
                    candidates.append((tokens + [token], score + value))
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
            beams = candidates
            if all(tokens[-1] == self.target_eos_id for tokens, _ in beams):
                break
        return beams
