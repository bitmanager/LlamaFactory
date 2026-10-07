"""Small stdio adapter to the unchanged, separately versioned Freeze-Omni runtime."""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from freeze_text import FreezeText


def main():
    root = Path(os.environ["NORMALIZER_ROOT"])
    tokenizer = AutoTokenizer.from_pretrained(root / "models/agent-tokenizer", local_files_only=True)
    model = FreezeText(Path(__file__).with_name("freeze_decoder.json"), 2560, len(tokenizer))
    model.load_state_dict(load_file(os.environ["NORMALIZER_CHECKPOINT"]), strict=True)
    model.cuda().eval().requires_grad_(False)
    print(json.dumps({"ready": True}), flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for line in sys.stdin:
            row = torch.load(json.loads(line)["features"], map_location="cpu", weights_only=True)
            ids = model.generate(row, max_tokens=192)
            if any(i >= len(tokenizer) for i in ids):
                raise ValueError("Decoder generated reserved non-text token")
            print(json.dumps({"tts_text": tokenizer.decode(ids, skip_special_tokens=False),
                              "tts_limit": len(ids) == 192}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
