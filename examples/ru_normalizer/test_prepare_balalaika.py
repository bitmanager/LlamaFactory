"""Check label integrity, conservative groups, and the real Parquet/exclusion path."""
import hashlib
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prepare_balalaika import REPO, REVISION, convert, family, main


def test_source_labels_preserved_and_chunk_ids_do_not_define_split():
    raw = dict(filepath="asr_public_phone_calls_1/a.opus", punct="Привет!", accent="Прив+ет!",
               podcast_id="a")
    row = convert(raw, "asr_public_phone_calls_1")
    assert row["written"] == raw["punct"] and row["spoken_stressed"] == raw["accent"]
    assert row["source_group"] == "balalaika:family:asr_public_phone_calls"
    assert family("asr_public_phone_calls_2") == family("asr_public_phone_calls_1")
    assert family("public_youtube1120_hq") == family("public_youtube700")


@pytest.mark.parametrize("punct,accent", [("Привет!", "Пок+а!"), ("Привет!", "Привет!"),
                                        ("Привет!", "Пр+ив+ет!"), ("Привет!", "Прив+ет 12!"),
                                        ("Звоните А. Пушкину.", "Звон+ите А. П+ушкину."),
                                        ("Дом с литерой г.", "Дом с лит+ерой г."),
                                        ("Позвони в МГУ.", "Позвон+и в МГУ.")])
def test_corrupt_labels_rejected(punct, accent):
    with pytest.raises(ValueError):
        convert(dict(filepath="part/a.opus", punct=punct, accent=accent), "part")


def test_parquet_pipeline_filters_aliases_duplicates_and_keeps_whole_families(tmp_path, monkeypatch):
    files = []
    for part, texts in [("asr_public_phone_calls_1", [("Привет!", "Прив+ет!"), ("Пока!", "Пок+а!")]),
                        ("asr_public_phone_calls_2", [("Пока!", "Пок+а!"), ("Спасибо!", "Спас+ибо!"),
                                                       ("Доверие!", "Дов+ерие!")]),
                        ("public_youtube1120_hq", [("Улыбка!", "Ул+ыбка!")])]:
        path = tmp_path / (part + "_balalaika.parquet")
        pq.write_table(pa.Table.from_pylist([dict(filepath=f"{part}/{i}.opus", punct=w, accent=t)
                                            for i, (w, t) in enumerate(texts)]), path)
        files.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(dict(repo=REPO, revision=REVISION, files=files)))
    train = tmp_path / "existing-train.jsonl"; train.write_text("")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps(dict(source_id="external:1",
                                     written="привет", spoken_stressed="прив+ет")) + "\n")
    excluded = tmp_path / "reviewed.jsonl"
    excluded.write_text(json.dumps(dict(source_id="balalaika:asr_public_phone_calls_2/2.opus",
                                       reason="reviewed_wrong_stress")) + "\n")
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare_balalaika", "--source-manifest", str(manifest),
        "--existing-train", str(train), "--heldouts", str(heldout), "--validation-family",
        "public_youtube", "--review-exclusions", str(excluded), "--output", str(output)])
    main()
    a = [json.loads(x) for x in (output / "train.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (output / "validation.jsonl").read_text().splitlines()]
    assert [x["written"] for x in a] == ["Спасибо!"]
    assert [x["written"] for x in b] == ["Улыбка!"]
    assert {x["source_group"] for x in a}.isdisjoint(x["source_group"] for x in b)
    quarantine = [json.loads(x) for x in (output / "quarantine.jsonl").read_text().splitlines()]
    assert any(x["reason"] == "reviewed_source:reviewed_wrong_stress" for x in quarantine)
    assert any(x["reason"] == "existing_train_or_heldout" for x in quarantine)
    assert json.loads((output / "manifest.json").read_text())["training_eligible"] is False
    with pytest.raises(ValueError, match="new dataset"):
        main()
