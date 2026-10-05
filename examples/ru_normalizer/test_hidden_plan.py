import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM

from hidden_plan import HiddenPlan


def model():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        tie_word_embeddings=True)
    decoder = get_peft_model(Qwen3ForCausalLM(config),
                            LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"]))
    return HiddenPlan(Qwen3Model(config), decoder)


def batch():
    return dict(source_ids=torch.tensor([[1, 5, 6], [1, 7, 0]]),
                source_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
                answer_mask=torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.bool),
                input_ids=torch.tensor([[1, 9, 10, 2], [1, 11, 2, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                labels=torch.tensor([[-100, -100, 10, 2], [-100, 11, 2, -100]]))


def test_only_projector_and_decoder_lora_receive_gradients():
    m = model().train()
    loss = m(**batch()).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert not m.source.training
    assert all(p.grad is None for p in m.source.parameters())
    assert sum(p.grad.abs().sum() for p in m.projector.parameters()) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in m.decoder.named_parameters() if "lora_" in n)
    assert all(p.grad is None for n, p in m.decoder.named_parameters() if "lora_" not in n)


def test_padding_does_not_change_valid_logits():
    m = model().eval()
    b = batch()
    together = m(**b).logits[1, -4:-1]
    alone = {k: v[1:2, :2] if k in ("source_ids", "source_mask", "answer_mask")
             else v[1:2, :3] for k, v in b.items()}
    separate = m(**alone).logits[0, -3:]
    torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-4)


def test_stock_generation_and_state_reload():
    m = model().eval()
    b = batch()
    b = {k: v[:1] for k, v in b.items() if k != "labels"}
    b["input_ids"] = b["input_ids"][:, :2]
    b["attention_mask"] = b["attention_mask"][:, :2]
    tokens = m.generate(**b, do_sample=False, max_new_tokens=3)
    restored = model().eval()
    restored.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(tokens, restored.generate(**b, do_sample=False, max_new_tokens=3))


def test_stock_trainer_checkpoint_resume(tmp_path):
    from transformers import Trainer, TrainingArguments
    b = batch()
    rows = [{k: v[i] for k, v in b.items()} for i in range(2)]
    def args(steps):
        return TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=steps,
                                 per_device_train_batch_size=2, save_steps=1,
                                 remove_unused_columns=False, report_to="none")
    first = Trainer(model=model(), args=args(1), train_dataset=rows)
    first.train()
    restored = model()
    second = Trainer(model=restored, args=args(2), train_dataset=rows)
    second.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))
    assert second.state.global_step == 2
    assert restored.decoder.get_input_embeddings().weight is restored.decoder.get_output_embeddings().weight
