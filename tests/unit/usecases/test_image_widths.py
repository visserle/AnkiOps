from ankiops.image_widths import fix_image_widths_in_data


def _data(answer: str, extra: str = ""):
    return {
        "decks": [
            {
                "name": "Deck",
                "notes": [
                    {
                        "note_key": "nk-1",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Answer": answer, "Extra": extra},
                    }
                ],
            }
        ]
    }


def test_auto_normalizes_near_equal_widths_to_first_width():
    data = _data(
        "![a](<media/a.png>){width=700}\n"
        "![b](<media/b.png>){width=703}\n"
        "![c](<media/c.png>){width=705}"
    )

    result = fix_image_widths_in_data(data, tolerance=5)

    assert result.images_checked == 3
    assert result.images_changed == 2
    fields = data["decks"][0]["notes"][0]["fields"]
    assert fields["Answer"].count("{width=700}") == 3


def test_auto_keeps_separate_clusters_within_one_note():
    data = _data(
        "![a](a.png){width=40}\n"
        "![b](b.png){width=39}\n"
        "![c](c.png){width=41}\n"
        "![d](d.png){width=100}\n"
        "![e](e.png){width=101}"
    )

    result = fix_image_widths_in_data(data, tolerance=5)

    assert result.images_changed == 3
    fields = data["decks"][0]["notes"][0]["fields"]
    assert fields["Answer"].count("{width=40}") == 3
    assert fields["Answer"].count("{width=100}") == 2


def test_auto_ignores_images_without_explicit_width():
    data = _data("![a](a.png)\n![b](b.png){width=402}\n![c](c.png){width=400}")

    result = fix_image_widths_in_data(data, tolerance=5)

    assert result.images_checked == 3
    assert result.images_changed == 1
    fields = data["decks"][0]["notes"][0]["fields"]
    assert "![a](a.png)\n" in fields["Answer"]
    assert "{width=402}" in fields["Answer"]
    assert "{width=400}" not in fields["Answer"]


def test_auto_uses_one_cluster_set_across_fields_in_note():
    data = _data(
        "![a](a.png){width=500}",
        extra="![b](b.png){width=504}",
    )

    result = fix_image_widths_in_data(data, tolerance=5)

    assert result.notes_changed == 1
    fields = data["decks"][0]["notes"][0]["fields"]
    assert fields["Extra"] == "![b](b.png){width=500}"


def test_force_replaces_existing_widths_and_adds_missing_widths():
    data = _data(
        "![a](<media/a.png>){width=700}\n![b](<media/b.png>)\n![c](c.png){width=250}"
    )

    result = fix_image_widths_in_data(data, tolerance=5, width=320)

    assert result.images_checked == 3
    assert result.images_changed == 3
    fields = data["decks"][0]["notes"][0]["fields"]
    assert fields["Answer"].count("{width=320}") == 3
    assert "{width=700}" not in fields["Answer"]
    assert "{width=250}" not in fields["Answer"]
