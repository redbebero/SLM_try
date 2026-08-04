from training.token_data import lexicon_to_records, split_token_records


def test_lexicon_emits_synonyms_as_same_fire_attribute():
    records = lexicon_to_records({
        "entries": [
            {
                "id": "fire",
                "surfaces": ["붉은", "빨간", "화염"],
                "attributes": [{"kind": "ELEMENT", "value": "FIRE", "delta": 1}],
                "provenance": {"type": "HUMAN_AUTHORED", "source_id": "fire", "verified": True}
            }
        ]
    })

    assert {record["input"]["token"] for record in records} == {"붉은", "빨간", "화염"}
    assert all(record["target"][0]["value"] == "FIRE" for record in records)


def test_lexicon_keeps_hard_negative_without_magic_attribute():
    records = lexicon_to_records({
        "entries": [
            {
                "id": "negative",
                "surfaces": ["평범한"],
                "attributes": [],
                "provenance": {"type": "HARD_NEGATIVE", "source_id": "negative", "verified": True}
            }
        ]
    })

    assert records[0]["target"] == []
    assert records[0]["input"]["token"] == "평범한"


def test_token_split_holds_out_surface_forms_without_hiding_label_from_train():
    records = lexicon_to_records({
        "entries": [{
            "id": "fire",
            "surfaces": ["불", "불꽃", "화염", "붉은", "빨간", "불길", "타오르는", "열기", "홍염", "연소"],
            "attributes": [{"kind": "ELEMENT", "value": "FIRE", "delta": 1}],
            "provenance": {"type": "HUMAN_AUTHORED", "source_id": "fire", "verified": True}
        }]
    })

    splits = split_token_records(records)

    assert len(splits["train"]) == 8
    assert len(splits["dev"]) == 1
    assert len(splits["test"]) == 1
    assert all(record["target"][0]["value"] == "FIRE" for record in splits["train"])
