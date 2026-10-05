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
