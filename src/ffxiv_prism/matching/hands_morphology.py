from __future__ import annotations

import re


HANDS_MORPHOLOGY_CLASSES = ("minimal", "long_armor", "glove")
_MINIMAL = re.compile(
    r"\b(nails?|claws?|wrist|ringbands?|bracelets?|chaplets?|runes?|dressing|bands?)\b",
    re.IGNORECASE,
)
_LONG_ARMOR = re.compile(
    r"\b(gauntlets?|vambraces?|armguards?|armlets?|kote|tekko|sleeves?)\b",
    re.IGNORECASE,
)


def hands_morphology_class(item_name: str | None) -> str:
    name = str(item_name or "")
    if _MINIMAL.search(name):
        return "minimal"
    if _LONG_ARMOR.search(name):
        return "long_armor"
    return "glove"
