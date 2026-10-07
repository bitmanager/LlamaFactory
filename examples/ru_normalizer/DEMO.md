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

Redeployed on 2026-10-07 at the user's request: dev GPU 2 (UUID
`GPU-dd032e60-3b80-204f-01a2-531b2bb6349f`), container
`normalizer-text-demo-v45-dev`, reviewed **V45 final**. Its immutable snapshot
SHA256 is `dae82842b57abdc6c87c9bc9aa43eb3d844eee032d84417a8285846e9245af27`.
The old exp demo is stopped. The existing UI/decoder code is reused unchanged;
both models share this one GPU. The container publishes only loopback 8616.
The persistent minipc `normalizer-demo-tunnel.service` now forwards 8510 to dev,
and the existing Tailscale Serve 8444 route is retained:
`https://minipc.tail683b27.ts.net:8444/`. Other serving routes are untouched.

Deployment validation: HTTPS health and HTML returned HTTP 200, and the
external-origin Streamlit WebSocket upgraded with HTTP 101. Two actual AppTest
requests through the existing interface returned nonempty Qwen and decoder
outputs without UI errors, including a follow-up using conversation history.
The first smoke attempt failed because the source directory was absent from
the test process's import path; the container working directory and PYTHONPATH
were corrected before the successful test, without changing Python code.

Smoke input: `Повтори точно: встреча в МГУ завтра в 15:30.` Qwen returned
`Встреча в МГУ завтра в 15:30.`, but the head returned
`Встр+еча в М+орту з+автра в пятн+адцать час+ов тридцат+и.` The follow-up
`Где и во сколько встреча?` produced Qwen's `В МГУ, завтра в 15:30.` and
the head's `В М+э у, з+автра в пятн+адцать час+ов тр+идцать мин+ут.`
Request times were 1.01 s and 0.48 s in the test process; this is not a latency
benchmark. These examples expose ongoing acronym/pronunciation errors and do
not establish model quality. V46 is not promoted to this demo. Deployment and
test receipts are in `research/demo-v45-20261007` on dev; the tailnet check is
also saved under local `validation/demo-v45-20261007`.
