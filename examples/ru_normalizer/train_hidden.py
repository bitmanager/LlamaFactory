"""Launch the approved hidden-to-text experiment with the stock HF Trainer."""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments, set_seed)

from hidden_plan import HiddenPlan, PrefixCollator, encode_example


@dataclass
class Inputs:
    model_path: str
    agent_tokenizer: str
    data_dir: str
    max_length: int = 768
    lora_rank: int = 16


def main():
    args, training = HfArgumentParser((Inputs, TrainingArguments)).parse_args_into_dataclasses()
    if not training.bf16:
        raise ValueError("Use BF16 for this GPU experiment")
    if training.remove_unused_columns:
        raise ValueError("Set --remove_unused_columns false to retain source fields")
    if not training.prediction_loss_only:
        raise ValueError("Use --prediction_loss_only true; evaluate generation separately")
    set_seed(training.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.agent_tokenizer, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files={s: str(Path(args.data_dir) / f"{s}.jsonl")
                                              for s in ("train", "validation")})
    dataset = dataset.map(lambda row: encode_example(row, tokenizer, args.max_length),
                          remove_columns=dataset["train"].column_names)
    source = AutoModel.from_pretrained(args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    decoder = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    # Match the ASR branch's exact tokenizer/resized vocabulary, preserving common rows.
    source.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    decoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    decoder = get_peft_model(decoder, LoraConfig(task_type=TaskType.CAUSAL_LM,
                             r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                             target_modules="all-linear", lora_dropout=0.0))
    model = HiddenPlan(source, decoder)
    counts = {"source": sum(p.numel() for p in source.parameters()),
              "projector_trainable": sum(p.numel() for p in model.projector.parameters()),
              "decoder_trainable": sum(p.numel() for p in decoder.parameters() if p.requires_grad)}
    print(json.dumps(counts), flush=True)
    output = Path(training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_inputs.json").exists() and not training.resume_from_checkpoint:
        raise FileExistsError("Existing run; choose a new output directory or explicitly resume")
    (output / "run_inputs.json").write_text(json.dumps(vars(args) | counts, indent=2))
    tokenizer.save_pretrained(output)
    trainer = Trainer(model=model, args=training, processing_class=tokenizer,
                      train_dataset=dataset["train"], eval_dataset=dataset["validation"],
                      data_collator=PrefixCollator(tokenizer))
    result = trainer.train(resume_from_checkpoint=training.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())


if __name__ == "__main__":
    main()
