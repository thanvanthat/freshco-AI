"""Regional foods in the knowledge base, and the lookups that reach them.

The lookup falls back to substring matching, so every new name is a chance to
quietly claim a query meant for another food. The table below is the names the
vision model and users actually produce; each must land on the right record.
"""

from __future__ import annotations

import pytest

from app.freshness import HIGH_RISK_SCORE_CAP, assess
from app.knowledge.food_data import ALL_FOODS
from app.knowledge.service import knowledge
from tests.stubs import pristine


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # What the vision model said in live tests.
        ("Bitter melon", "Bitter gourd"),
        ("Idli", "Idli"),
        ("moldy bread", "Bread"),
        # Regional names people type.
        ("Bhindi", "Okra"),
        ("Lady finger", "Okra"),
        ("Karela", "Bitter gourd"),
        ("Curd", "Yogurt"),
        ("Dahi", "Yogurt"),
        ("Roti", "Chapati"),
        ("Parotta", "Paratha"),
        ("Chikoo", "Sapota"),
        ("Methi", "Fenugreek leaves"),
        ("Moringa", "Drumstick (moringa)"),
        ("Drumstick", "Drumstick (moringa)"),
        ("Cottage cheese", "Paneer"),
        ("Plantain", "Raw banana"),
        # Collisions that would be wrong, and in one case unsafe.
        ("Chicken drumstick", "Chicken"),  # raw poultry must stay capped
        ("Melon", "Muskmelon"),  # not Bitter gourd via "Bitter melon"
        ("Custard apple", "Custard apple"),  # not Apple
        ("Sweet potato", "Sweet potato"),  # not Potato
        ("Banana", "Banana"),  # not Raw banana
        ("Cheese", "Cheese"),  # not Paneer
        ("Pineapple", "Pineapple"),
    ],
)
def test_names_resolve_to_the_right_food(query: str, expected: str) -> None:
    record = knowledge.resolve(query)
    assert record is not None, f"{query!r} resolved to nothing"
    assert record.name == expected


def test_no_name_or_alias_is_claimed_twice() -> None:
    """The index keeps the first record for a key and drops the rest silently."""
    seen: dict[str, str] = {}
    for record in ALL_FOODS:
        for key in [record.name, *record.aliases]:
            owner = seen.setdefault(key.lower(), record.name)
            assert owner == record.name, f"{key!r} belongs to {owner} and {record.name}"


def test_no_short_names_for_the_substring_stage_to_misuse() -> None:
    for record in ALL_FOODS:
        for key in [record.name, *record.aliases]:
            assert len(key) >= 4, f"{record.name}: {key!r} is short enough to collide"


def test_every_record_is_usable() -> None:
    for record in ALL_FOODS:
        assert record.shelf_life.for_storage(record.preferred_storage), (
            f"{record.name} has no shelf life for its own preferred storage"
        )
        if record.healthy_hue is not None:
            low, high = record.healthy_hue
            assert 0 <= low <= high <= 179, f"{record.name}: bad hue {record.healthy_hue}"


@pytest.mark.parametrize("name", ["Cooked rice", "Idli", "Paneer"])
def test_cooked_staples_are_capped_however_good_they_look(name: str) -> None:
    """Bacillus cereus in cooked rice does not show in a photo."""
    record = knowledge.resolve(name)
    result = assess(pristine(), record.category, record)
    assert result.capped is True
    assert result.score <= HIGH_RISK_SCORE_CAP
    assert result.status != "fresh"


def test_flatbreads_are_not_capped() -> None:
    record = knowledge.resolve("Chapati")
    assert assess(pristine(), record.category, record).capped is False
