"""Focused Unicode and label-equivalence checks; no dataset/model inference."""
import json

import pytest

from prepare_local import digest
from prepare_ruslan import convert, signature


def test_utf8_conversion_preserves_source_and_original_target(tmp_path):
    original = "С трево́жным чу́вством беру́сь я́ за́ перо́."
    row = convert("000042_RUSLAN", original)
    assert row["reference_stressed"] == "С трев+ожным ч+увством бер+усь +я з+а пер+о."
    assert row["spoken_stressed"] == row["reference_stressed"]
    assert row["written"] == row["spoken"] == "С тревожным чувством берусь я за перо."
    assert row["original_accented"] == original
    path = tmp_path / "row.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == row
    assert b"\\u" not in path.read_bytes()


@pytest.mark.parametrize("left,right", [
    ("+Я з+а пер+о.", "Я за пер+о."),
    ("Вс+ё решен+о.", "Всё решен+о."),
    ("+Ёлка.", "Ёлка."),
])
def test_only_optional_monosyllables_and_explicit_yo_are_equivalent(left, right):
    assert signature(left) == signature(right)


@pytest.mark.parametrize("left,right", [
    ("Всё решен+о.", "Все решен+о."),
    ("з+амок", "зам+ок"),
    ("Прив+ет!", "Прив+ет."),
    ("он+а", "Он+а"),
])
def test_content_and_real_stress_changes_are_not_consensus(left, right):
    assert signature(left) != signature(right)


@pytest.mark.parametrize("text", ["+привет", "пр+ив+ет", "ё+лка", "тр+ёхэтажн+ый", "привет", "приве\u0301т"])
def test_malformed_or_missing_primary_stress_rejected(text):
    with pytest.raises(ValueError):
        signature(text)


@pytest.mark.parametrize("text", ["Прив́ет.", "При́ве́т.", "Ёлка́.", "Звони́ в МГУ.", "Звони́ А. Пушки́ну.", "Верси́я 25.", "При́вет Python."])
def test_bad_source_markers_raw_abbreviations_and_nonspoken_input_rejected(text):
    with pytest.raises(ValueError):
        convert("000042_RUSLAN", text)


def test_contiguous_hundred_id_group_and_hash_split():
    rows = [convert(f"{i:06d}_RUSLAN", "Приве́т.") for i in [0, 99, 100]]
    assert rows[0]["source_group"] == rows[1]["source_group"] == "ruslan:block:0"
    assert rows[2]["source_group"] == "ruslan:block:1"
    for row in rows:
        expected = "validation" if int(digest(row["source_group"])[:8], 16) % 20 == 0 else "train"
        assert row["split"] == expected
