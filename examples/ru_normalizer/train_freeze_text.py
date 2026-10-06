"""Stock HF Trainer with original Freeze-Omni forward, CE, KV-prefix and infer."""
import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed
from freeze_text import FreezeText, FeatureDataset, collate


@dataclass
class Inputs:
    features: str
    model_config: str
    pretrained: str
    resume: str = None
    warm_start: str = None


class Examples(TrainerCallback):
    def __init__(self, dataset, tokenizer):
        self.rows = [dataset[i] for i in range(min(4, len(dataset)))]
        # Include held-out normalization cases, not only stress copying.
        selected = {r["source_id"] for r in self.rows}
        for pattern in (r"[А-ЯЁ]{2,}", r"[A-Za-z]", r"\d"):
            count = 0
            for i in range(len(dataset)):
                row = dataset[i]
                if row["source_id"] not in selected and re.search(pattern, row["written"]):
                    self.rows.append(row)
                    selected.add(row["source_id"])
                    count += 1
                    if count == 2:
                        break
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        was_training = model.training
        model.eval()
        results = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for row in self.rows:
                ids = model.generate(row)
                prediction = self.tokenizer.decode(ids, skip_special_tokens=False)
                results.append(dict(source_id=row["source_id"], written=row["written"],
                                    reference=row["reference"], prediction=prediction))
        model.train(was_training)
        path = Path(args.output_dir) / f"examples-{state.global_step}.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps({"step": state.global_step, "examples": results}, ensure_ascii=False), flush=True)


def main():
    inputs, args = HfArgumentParser((Inputs, TrainingArguments)).parse_args_into_dataclasses()
    if not args.bf16 or args.remove_unused_columns or not args.prediction_loss_only:
        raise ValueError("Require BF16, retain input columns, and prediction_loss_only")
    if inputs.resume and inputs.warm_start:
        raise ValueError("Choose exact resume or model-only warm start, not both")
    root = Path(inputs.features)
    metadata = json.loads((root / "metadata.json").read_text())
    if "assembled_from" in metadata:
        complete = json.loads((root / "COMPLETE.json").read_text())
        if complete["metadata_sha256"] != hashlib.sha256((root / "metadata.json").read_bytes()).hexdigest():
            raise ValueError("Assembled feature cache completion hash mismatch")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not inputs.resume:
        raise FileExistsError("Choose a fresh run directory or explicitly resume")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = FreezeText(inputs.model_config, metadata["source_dim"], metadata["vocab_size"], inputs.pretrained)
    if inputs.warm_start:
        load_model(model, str(Path(inputs.warm_start) / "model.safetensors"), strict=True)
    train, validation = (FeatureDataset(root / s) for s in ("train", "validation"))
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (output / "run_inputs.json").write_text(json.dumps(vars(inputs) | {"trainable_parameters": params}, indent=2))
    print(json.dumps(dict(trainable_parameters=params, train=len(train), validation=len(validation))), flush=True)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=validation,
                      data_collator=collate, callbacks=[Examples(validation, tokenizer)])
    result = trainer.train(resume_from_checkpoint=inputs.resume)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())


if __name__ == "__main__":
    main()
