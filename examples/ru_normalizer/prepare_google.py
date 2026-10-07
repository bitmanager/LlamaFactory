# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for permissions and limitations under the License.
"""Export Google TRAIN numeric labels unchanged; no normalization or accent model.

read_google_data below is verbatim from NVIDIA/NeMo v1.23.0, commit
 d2283e3620cd7f99dbe29fdf079757ab9f6cdf01,
 examples/nlp/duplex_text_normalization/data/data_split.py (Apache-2.0).
Only its tiny constants dependency is supplied locally; no NeMo model import.
"""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from prepare_local import TEXT_FIELDS, key, read

constants = SimpleNamespace(ENGLISH="en", RUSSIAN="ru")
TEST_SIZE_EN, TEST_SIZE_RUS = 100002, 100007
UPSTREAM = "NVIDIA/NeMo@d2283e3620cd7f99dbe29fdf079757ab9f6cdf01"
NUMERIC = {"CARDINAL", "DATE", "TIME", "MONEY", "DECIMAL"}
HOLDOUT_FIELDS = TEXT_FIELDS + ("written", "spoken", "spoken_stressed", "source_text",
                               "source_text_stressed", "normalized_runorm", "target", "norm")


def read_google_data(data_file: str, lang: str, split: str, add_test_full=False):
    """
    The function can be used to read the raw data files of the Google Text Normalization
    dataset (which can be downloaded from https://www.kaggle.com/google-nlu/text-normalization)

    Args:
        data_file: Path to the data file. Should be of the form output-xxxxx-of-00100
        lang: Selected language.
        split: data split
        add_test_full: do not truncate test data i.e. take the whole test file not #num of lines
    Return:
        data: list of examples
    """
    data = []
    cur_classes, cur_tokens, cur_outputs = [], [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        for linectx, line in tqdm(enumerate(f)):
            es = line.strip().split('\t')
            if split == "test" and not add_test_full:
                # For the results reported in the paper "RNN Approaches to Text Normalization: A Challenge":
                # + For English, the first 100,002 lines of output-00099-of-00100 are used for the test set
                # + For Russian, the first 100,007 lines of output-00099-of-00100 are used for the test set
                if lang == constants.ENGLISH and linectx == TEST_SIZE_EN:
                    break
                if lang == constants.RUSSIAN and linectx == TEST_SIZE_RUS:
                    break
            if len(es) == 2 and es[0] == '<eos>':
                data.append((cur_classes, cur_tokens, cur_outputs))
                # Reset
                cur_classes, cur_tokens, cur_outputs = [], [], []
                continue

            # Remove _trans (for Russian)
            if lang == constants.RUSSIAN:
                es[2] = es[2].replace('_trans', '')
            # Update the current example
            assert len(es) == 3
            cur_classes.append(es[0])
            cur_tokens.append(es[1])
            cur_outputs.append(es[2])
    return data



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--holdouts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()
    if args.limit < 1 or args.output.exists():
        raise ValueError("Need positive limit and a new output directory")
    heldout, holdout_hashes = set(), {}
    for path in sorted(set(args.holdouts)):
        rows = read(path)
        values = [key(row[field]) for row in rows for field in HOLDOUT_FIELDS
                  if isinstance(row.get(field), str) and row[field].strip()]
        if not values:
            raise ValueError(f"No recognized holdout text: {path}")
        heldout.update(values)
        holdout_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    output, seen, stats, class_counts, opened = [], set(), Counter(), Counter(), []
    with tarfile.open(args.archive, "r|gz") as archive, tempfile.TemporaryDirectory() as tmp:
        for member in archive:
            match = re.fullmatch(r"ru_with_types/output-(\d{5})-of-00100", member.name)
            if not member.isfile() or not match or int(match[1]) >= 90:
                continue
            path = Path(tmp) / "train.tsv"
            with archive.extractfile(member) as source, path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            opened.append(member.name)
            # Upstream parser fails loudly on malformed TSV rather than relabeling it.
            for index, (classes, written_tokens, spoken_tokens) in enumerate(
                    read_google_data(str(path), "ru", "train")):
                stats["scanned_sentences"] += 1
                if not classes or not (len(classes) == len(written_tokens) == len(spoken_tokens)):
                    stats["empty_or_malformed"] += 1
                    continue
                selected = NUMERIC.intersection(classes)
                if not selected or not any(re.search(r"\d", w) and c in NUMERIC
                                           for c, w in zip(classes, written_tokens)):
                    stats["no_numeric_conversion"] += 1
                    continue
                if "LETTERS" in classes:
                    stats["letters_class"] += 1
                    continue
                # Same self/sil resolution as upstream NeMo TN sentence dataset.
                written = " ".join(written_tokens)
                spoken = " ".join(w if s in ("<self>", "sil") else s
                                  for w, s in zip(written_tokens, spoken_tokens))
                if any(not value.strip() for value in written_tokens + spoken_tokens):
                    stats["empty_token"] += 1
                    continue
                if re.search(r"[A-Za-z0-9_<>{}+]", spoken) or re.search(r"[A-Za-z_<>{}+]", written):
                    stats["unspoken_or_marker"] += 1
                    continue
                # Kestrel can embed spelling labels inside MONEY, e.g. "долларов сэ ш а".
                if re.search(r"\b[бгджзйлмнпртфхцчшщъыь]\b", spoken.lower()):
                    stats["embedded_letter_spelling"] += 1
                    continue
                if not re.search(r"[а-яё]", spoken.lower()) or len(written) > 900 or len(spoken) > 1400:
                    stats["invalid_or_long"] += 1
                    continue
                identities = {key(written), key(spoken)}
                if identities & heldout:
                    stats["existing_holdout"] += 1
                    continue
                if identities & seen:
                    stats["duplicate_written_or_spoken"] += 1
                    continue
                seen.update(identities)
                class_counts.update(selected)
                output.append(dict(written=written, spoken=spoken, source_id=f"google:{member.name}:{index}",
                    source_group=member.name, classes=sorted(selected),
                    provenance="Wikipedia 2016 / Google Kestrel pseudo labels; CC BY-SA 4.0; no stress"))
                if len(output) >= args.limit:
                    break
            if len(output) >= args.limit:
                break
    if not output:
        raise ValueError("No eligible numeric training sentences")
    args.output.mkdir(parents=True)
    with (args.output / "train.jsonl").open("x") as dest:
        for row in output:
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = dict(rows=len(output), filters=dict(stats), classes=dict(class_counts),
        opened_train_shards=opened, holdout_keys=len(heldout), holdout_sha256=holdout_hashes,
        upstream=UPSTREAM, archive=str(args.archive),
        limitations=["Kestrel labels are automatic, not gold", "No stress or relabeling",
                     "Token-spaced sentence text retained", "Only numeric classes selected; not an abbreviation dataset",
                     "Exact normalized written/spoken dedup, not semantic/paraphrase dedup"])
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "holdout_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
