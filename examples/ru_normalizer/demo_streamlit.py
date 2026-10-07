"""Text question -> stock Qwen generation -> stock Freeze-Omni text plan."""
import atexit
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from export_qwen_features import AGENT_SYSTEM


class Demo:
    def __init__(self):
        root = Path(os.environ["NORMALIZER_ROOT"])
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(root / "models/agent-tokenizer", local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            root / "models/qwen3-4b-instruct-2507", dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).cuda().eval()
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self.model.requires_grad_(False)
        env = dict(os.environ, PYTHONPATH=str(root / "freeze-omni-deps") + ":" + os.environ["PYTHONPATH"])
        self.worker = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("demo_decoder.py"))],
                                       env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        atexit.register(self.worker.terminate)
        if not self.read_worker().get("ready"):
            raise RuntimeError("Freeze-Omni did not initialize")

    def read_worker(self):
        if not select.select([self.worker.stdout], [], [], 180)[0]:
            self.worker.terminate()
            raise TimeoutError("Freeze-Omni timed out; inspect server logs")
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("Freeze-Omni worker exited; inspect server logs")
        return json.loads(line)

    @torch.inference_mode()
    def answer(self, messages):
        with self.lock:
            start_time = time.monotonic()
            prefix = self.tokenizer.apply_chat_template(messages, tokenize=True,
                         add_generation_prompt=True, return_dict=False)
            if len(prefix) > 640:
                raise ValueError("Контекст слишком длинный для этого теста. Начните новый разговор или сократите вопрос.")
            inputs = torch.tensor([prefix], device="cuda")
            config = GenerationConfig(do_sample=False, max_new_tokens=128,
                                      eos_token_id=self.model.generation_config.eos_token_id,
                                      pad_token_id=self.tokenizer.eos_token_id)
            generated = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
            answer_ids = generated[0, len(prefix):].tolist()
            eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
            ended = bool(answer_ids and answer_ids[-1] in eos)
            if ended:
                answer_ids.pop()
            if not answer_ids:
                raise ValueError("Qwen returned an empty answer")
            # Exactly the training export's pre-token final-normalized alignment.
            full = torch.tensor([prefix + answer_ids], device="cuda")
            hidden = self.model.model(full, attention_mask=torch.ones_like(full), use_cache=False).last_hidden_state
            embeddings = self.model.get_input_embeddings()(full)
            start, end = len(prefix), full.shape[1]
            row = {"x": embeddings[0, start:end].cpu(), "x_prefix": hidden[0, start-1:end-1].cpu()}
            with tempfile.TemporaryDirectory(prefix="normalizer-demo-") as directory:
                path = Path(directory) / "features.pt"
                torch.save(row, path)
                self.worker.stdin.write(json.dumps({"features": str(path)}) + "\n")
                self.worker.stdin.flush()
                result = self.read_worker()
            return dict(result, answer=self.tokenizer.decode(answer_ids, skip_special_tokens=True),
                        answer_limit=not ended, seconds=round(time.monotonic() - start_time, 2))


@st.cache_resource
def engine():
    return Demo()


def main():
    st.set_page_config(page_title="Qwen · текст для TTS", page_icon="💬")
    st.title("Qwen → текст для TTS")
    st.caption("Задайте вопрос: слева — ответ Qwen, справа — его обработка нашей обученной головой.")
    st.session_state.setdefault("turns", [])
    with st.sidebar:
        st.write("Qwen3-4B-Instruct-2507")
        st.caption(os.environ.get("NORMALIZER_LABEL", Path(os.environ["NORMALIZER_CHECKPOINT"]).stem))
        st.caption("Одна GPU · экспериментальный чекпойнт")
        system = st.text_area("Системный промпт", AGENT_SYSTEM)
        if st.button("Новый разговор"):
            st.session_state.turns = []
            st.rerun()
        st.caption("Знак + стоит перед ударной гласной. Числа и ударения ещё могут содержать ошибки. Здесь выводится текст, без озвучки.")
    question = st.chat_input("Например: повтори — встреча в МГУ в 15:30")
    if question:
        messages = [{"role": "system", "content": system}]
        for turn in st.session_state.turns:
            messages.extend([{"role": "user", "content": turn["question"]},
                             {"role": "assistant", "content": turn["answer"]}])
        messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Qwen отвечает, затем голова готовит текст для TTS…"):
                result = engine().answer(messages)
            st.session_state.turns.append(dict(result, question=question))
        except Exception as error:
            st.error(f"Не удалось выполнить запрос: {error}")
            st.stop()
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        left, right = st.columns(2)
        with left:
            st.markdown("**Ответ Qwen**")
            st.write(turn["answer"])
        with right:
            st.markdown("**Текст для TTS**")
            st.write(turn["tts_text"])
        st.caption(f"{turn['seconds']} с")
        if turn["answer_limit"] or turn["tts_limit"]:
            st.warning("Достигнут лимит длины; один из текстов мог оборваться.")


if __name__ == "__main__":
    main()
