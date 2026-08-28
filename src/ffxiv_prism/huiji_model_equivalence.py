from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable


def build_piece_equivalence_from_models(
    pieces: Iterable[dict[str, Any]],
    items: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Reproduce HuijiWiki's visible-armour same-model relation.

    Huiji groups armour by ItemUICategory and ModelMain[0]. It labels rows
    with equal ModelMain[1] as completely identical models. Weapons and tools
    use another branch and are outside the current five-slot sample library.
    """
    piece_rows = [normalize_piece(row) for row in pieces]
    item_rows = [normalize_item_model(row) for row in items]
    items_by_id = {
        row["game_item_id"]: row
        for row in item_rows
        if row["game_item_id"] is not None
    }
    items_by_en: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        if row["name_en"]:
            items_by_en[normalize_name(row["name_en"])].append(row)

    piece_models: dict[int, list[dict[str, Any]]] = {}
    mapping_method_counts: dict[str, int] = defaultdict(int)
    unmapped_pieces: list[dict[str, Any]] = []
    for piece in piece_rows:
        matched: dict[int, dict[str, Any]] = {}
        for item_id in piece["game_item_ids"]:
            item = items_by_id.get(item_id)
            if item is not None:
                matched[item_id] = item
        method = "game_item_id" if matched else None
        if not matched and piece["name_en"]:
            candidates = items_by_en.get(normalize_name(piece["name_en"]), [])
            for item in candidates:
                item_id = item["game_item_id"]
                if item_id is not None:
                    matched[item_id] = item
            method = (
                "unique_name_en"
                if len(matched) == 1
                else ("name_en_multiple" if matched else None)
            )
        usable = [
            item
            for item in matched.values()
            if valid_armour_model(item)
        ]
        if not usable:
            unmapped_pieces.append(
                {
                    "piece_id": piece["piece_id"],
                    "name_en": piece["name_en"],
                    "slot": piece["slot"],
                    "candidate_item_count": len(matched),
                }
            )
            continue
        piece_models[piece["piece_id"]] = usable
        mapping_method_counts[str(method)] += 1

    exact_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    broad_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for piece in piece_rows:
        for item in piece_models.get(piece["piece_id"], []):
            model = item["model_main"]
            broad_key = (
                piece["slot"],
                item["item_ui_category_id"],
                model[0],
            )
            exact_key = broad_key + (model[1],)
            broad_groups[broad_key].add(piece["piece_id"])
            exact_groups[exact_key].add(piece["piece_id"])

    broad_equivalents = groups_to_equivalents(broad_groups.values())
    exact_equivalents = groups_to_equivalents(exact_groups.values())
    broad_edge_count = count_equivalence_edges(broad_groups.values())
    exact_edge_count = count_equivalence_edges(exact_groups.values())
    return {
        "schema_version": 1,
        "source": {
            "relation_rule": (
                "https://ff14.huijiwiki.com/wiki/"
                "%E6%A8%A1%E5%9D%97:Item/Page"
            ),
            "model_fields": "https://cafemaker.wakingsands.com/Item",
        },
        "method": "huijiwiki_rule_over_game_item_model_fields",
        "scope": "visible armour slots only",
        "inference_used": False,
        "default_relation": "same_model_including_main_model",
        "direct_equivalents": stringify_map(broad_equivalents),
        "same_model_equivalents": stringify_map(broad_equivalents),
        "exact_model_equivalents": stringify_map(exact_equivalents),
        "piece_count": len(piece_rows),
        "mapped_piece_count": len(piece_models),
        "unmapped_piece_count": len(unmapped_pieces),
        "mapping_method_counts": dict(sorted(mapping_method_counts.items())),
        "broad_group_count": sum(
            len(values) > 1 for values in broad_groups.values()
        ),
        "exact_group_count": sum(
            len(values) > 1 for values in exact_groups.values()
        ),
        "edge_count": broad_edge_count,
        "exact_model_edge_count": exact_edge_count,
        "main_model_edge_count": broad_edge_count - exact_edge_count,
        "unmapped_pieces": unmapped_pieces,
    }


def normalize_piece(row: dict[str, Any]) -> dict[str, Any]:
    item_ids: set[int] = set()
    direct_id = normalize_int(row.get("game_item_id"))
    if direct_id is not None:
        item_ids.add(direct_id)
    values = row.get("game_item_ids")
    if values is None:
        values = row.get("game_item_ids_json")
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = []
    if isinstance(values, list):
        for value in values:
            item_id = normalize_int(value)
            if item_id is not None:
                item_ids.add(item_id)
    return {
        "piece_id": int(row["piece_id"]),
        "slot": str(row.get("slot") or ""),
        "name_en": str(row.get("name_en") or row.get("name") or "").strip(),
        "game_item_ids": sorted(item_ids),
    }


def normalize_item_model(row: dict[str, Any]) -> dict[str, Any]:
    model = row.get("model_main", row.get("ModelMain"))
    if isinstance(model, str):
        values: list[Any] = model.split(",")
    elif isinstance(model, (list, tuple)):
        values = list(model)
    else:
        values = []
    parsed_model = [
        parsed
        for value in values
        if (parsed := normalize_int(value)) is not None
    ]
    return {
        "game_item_id": normalize_int(
            row.get("game_item_id", row.get("ID"))
        ),
        "name_en": str(
            row.get("name_en", row.get("Name_en")) or ""
        ).strip(),
        "item_ui_category_id": normalize_int(
            row.get(
                "item_ui_category_id",
                row.get("ItemUICategoryTargetID"),
            )
        ),
        "model_main": parsed_model,
    }


def valid_armour_model(item: dict[str, Any]) -> bool:
    model = item["model_main"]
    return (
        item["item_ui_category_id"] is not None
        and len(model) >= 2
        and model[0] > 0
    )


def groups_to_equivalents(
    groups: Iterable[set[int]],
) -> dict[int, list[int]]:
    equivalents: dict[int, set[int]] = defaultdict(set)
    for group in groups:
        if len(group) < 2:
            continue
        for piece_id in group:
            equivalents[piece_id].update(group - {piece_id})
    return {
        piece_id: sorted(values)
        for piece_id, values in sorted(equivalents.items())
    }


def count_equivalence_edges(groups: Iterable[set[int]]) -> int:
    edges: set[tuple[int, int]] = set()
    for group in groups:
        values = sorted(group)
        for offset, left in enumerate(values):
            for right in values[offset + 1 :]:
                edges.add((left, right))
    return len(edges)


def stringify_map(values: dict[int, list[int]]) -> dict[str, list[int]]:
    return {
        str(piece_id): equivalents
        for piece_id, equivalents in sorted(values.items())
    }


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
