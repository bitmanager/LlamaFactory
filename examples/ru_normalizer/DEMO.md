# Research text demo

`demo_streamlit.py` sends the typed question and conversation history to frozen
Qwen3-4B-Instruct-2507 with the existing Russian system prompt. It displays the
answer and the trained Freeze-Omni text plan side by side. No audio is generated.
It uses stock HF greedy generation, recomputes exactly the exporter's pre-token
final-normalized hidden states, and calls existing `FreezeText.generate()`.

Both models use one GPU. Qwen runs in the modern environment; `demo_decoder.py`
keeps the unchanged Transformers 4.45.2 decoder in a resident subprocess. The
adapter sends temporary feature-file paths over local stdin/stdout, serializes
requests, and fails visibly on worker failure or timeout. It does not implement
a decoder, modify weights, or silently substitute another normalizer.

Set `NORMALIZER_ROOT`, `NORMALIZER_CHECKPOINT` (an immutable full safetensors
snapshot), and optionally `NORMALIZER_LABEL`. The root contains the existing
models and `freeze-omni-deps`; expose the modern environment through `PYTHONPATH`
and install Streamlit 1.55.0 in a separate dependency directory.

```bash
python -m streamlit run examples/ru_normalizer/demo_streamlit.py \
  --global.developmentMode=false --server.address=0.0.0.0 \
  --server.port=8510 --server.headless=true --browser.gatherUsageStats=false
```

The demo limits the prompt to 640 tokens and the answer to 128, matching the
exporter's 768-token source bound. Native text-plan generation is limited to
192 tokens as in evaluation. Reaching an output limit produces a visible warning;
long histories produce a visible error rather than silent truncation.

Deployment on 2026-10-06: exp GPU 3, container `normalizer-text-demo`, host loopback
8616; minipc SSH tunnel `normalizer-demo-tunnel.service` forwards local 8510.
Tailscale Serve HTTPS 8444 forwards to 8510, restricted to the tailnet. Existing
443/8443 routes are preserved. This deployment serves V22 final; training is
independent and checkpoints do not change underneath an active session.

Validation: Streamlit AppTest completed two actual GPU requests, including a
follow-up referring to the first question. HTTPS health returned `ok`; the
external-origin Streamlit WebSocket upgraded with HTTP 101. Warm requests took
1.70 s and 0.19 s in this small smoke test (not a latency benchmark). The test
also exposed model errors: Qwen preserved `Битменеджер`, but the text head produced
`Г+итменеджер` and later `М+е`. The interface deliberately displays these raw
outputs. Passing the interface test is not a model-quality claim; most training
examples lacked dialogue history, so contextual live inputs need further evaluation.
