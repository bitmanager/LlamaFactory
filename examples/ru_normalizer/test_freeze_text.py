import json
import torch
from freeze_text import FreezeText, collate


def model(tmp_path):
    config = dict(idim=32, odim=64, encoder_pre_norm_type="ln", encoder_drop_rate=0.,
                  encoder_criterion="ce", encoder_upsample_rate=1, encoder_output_dim=32,
                  transformer_attention_dim=32, transformer_linear_units=64,
                  transformer_num_blocks=2, transformer_attention_heads=4,
                  transformer_dropout_rate=0., kv_cache_prefix_finetune=1)
    path = tmp_path / "model.json"
    path.write_text(json.dumps([32, 64, config]))
    torch.manual_seed(4)
    return FreezeText(path, 16, 64)


def rows():
    return [dict(x=torch.randn(n, 16), x_prefix=torch.randn(n, 16),
                 y=torch.tensor([5, 7, 9][:n])) for n in (3, 2)]


def test_gradients_reach_original_prefix_and_adapters(tmp_path):
    m = model(tmp_path).train()
    batch = collate(rows())
    original_y = batch["y"].clone()
    loss = m(**batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(batch["y"], original_y)
    for module in (m.text_adapter, m.hidden_adapter, m.decoder.layers_prefix,
                   m.decoder.layers, m.decoder.embedding, m.decoder.out_fnn):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_upstream_teacher_forcing_matches_first_inference_logit(tmp_path):
    m = model(tmp_path).eval()
    row = rows()[0]
    logits = []
    hook = m.decoder.out_fnn.register_forward_hook(lambda module, inputs, output: logits.append(output.detach()))
    m(**collate([row]))
    expected = logits.pop()[0, 0]
    m.generate(row, max_tokens=1)
    actual = logits.pop()[0, 0]
    hook.remove()
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)


def test_padding_preserves_all_valid_target_logits_and_loss(tmp_path):
    m = model(tmp_path).eval()
    examples = rows()
    # Prefix and text padding vary independently.
    examples[0]["x_prefix"] = torch.randn(2, 16)
    examples[1]["x_prefix"] = torch.randn(5, 16)
    logits = []
    hook = m.decoder.out_fnn.register_forward_hook(lambda module, inputs, output: logits.append(output.detach()))
    batch_loss = m(**collate(examples))["loss"]
    batch_logits = logits.pop()
    losses, lengths = [], []
    for index, row in enumerate(examples):
        losses.append(m(**collate([row]))["loss"])
        lengths.append(len(row["y"]) + 1)
        torch.testing.assert_close(batch_logits[index, :lengths[-1]], logits.pop()[0], atol=1e-5, rtol=1e-4)
    expected_loss = sum(loss * length for loss, length in zip(losses, lengths)) / sum(lengths)
    torch.testing.assert_close(batch_loss, expected_loss, atol=1e-5, rtol=1e-4)
    hook.remove()
