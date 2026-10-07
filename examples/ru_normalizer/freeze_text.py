"""Format/dimension adapters around the original Freeze-Omni model."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party/Freeze-Omni"
sys.path.insert(0, str(UPSTREAM))
from models.adapter import LinearAdapter
from models.decoder.decoder import LLM2TTSCodecAR


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.paths = sorted(Path(path).glob("*.pt"))
        if not self.paths:
            raise ValueError(f"Empty feature directory: {path}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location="cpu", weights_only=True)


def collate(rows):
    return {**{k: pad_sequence([r[k] for r in rows], batch_first=True,
                               padding_value=-1 if k == "y" else 0)
               for k in ("x", "x_prefix", "y")},
            **{k + "_lens": torch.tensor([len(r[k]) for r in rows])
               for k in ("x", "x_prefix", "y")}}


class FreezeText(nn.Module):
    def __init__(self, model_config, source_dim, vocab_size, pretrained=None):
        super().__init__()
        config = json.loads(Path(model_config).read_text())[2]
        config.update(odim=vocab_size, kv_cache_prefix_finetune=1)
        args = argparse.Namespace(**config)
        self.decoder = LLM2TTSCodecAR(args.idim, args.odim, args)
        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            state = state.get("model", state)
            # Only the approved speech-vocabulary -> text-vocabulary replacement.
            replaced = {"embedding.weight", "out_fnn.weight", "out_fnn.bias"}
            result = self.decoder.load_state_dict({k: v for k, v in state.items() if k not in replaced}, strict=False)
            if set(result.missing_keys) != replaced or result.unexpected_keys:
                raise ValueError(f"Unexpected pretrained checkpoint mismatch: {result}")
        self.text_adapter = LinearAdapter(source_dim, args.idim)
        self.hidden_adapter = LinearAdapter(source_dim, args.idim)
        # New text output cannot use frozen speech-token embeddings/head.
        self.decoder.requires_grad_(True)
        self.decoder.reporter = SimpleNamespace(log_loss=lambda *args: None)

    def forward(self, x, x_lens, x_prefix, x_prefix_lens, y, y_lens):
        x, _ = self.text_adapter(x, None)
        x_prefix, _ = self.hidden_adapter(x_prefix, None)
        loss = self.decoder(dict(x=x, x_lens=x_lens, x_prefix=x_prefix,
                                 x_prefix_lens=x_prefix_lens, y=y.clone(), y_lens=y_lens))
        # Upstream CE is a sum; normalize for the stock Trainer's accumulation.
        return {"loss": loss / (y_lens + 1).sum()}

    @torch.no_grad()
    def generate(self, row, max_tokens=192):
        device = next(self.parameters()).device
        x, _ = self.text_adapter(row["x"].unsqueeze(0).to(device), None)
        prefix, _ = self.hidden_adapter(row["x_prefix"].unsqueeze(0).to(device), None)
        # top_k=1 is upstream greedy decoding; no custom generation loop.
        pieces = list(self.decoder.infer(x, 1, prefix, -1, 1.0, max_tokens=max_tokens))
        return torch.cat(pieces, dim=1)[0].cpu().tolist() if pieces else []
