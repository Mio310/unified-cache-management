"""Helpers for preserving request-prefix relations across task slicing.

These helpers do not register relations with a Store.  Callers must use a
complete, ordered request hash chain, not the output of rank sampling, to
construct them.
"""

from collections.abc import Callable, Iterable, Sequence


# None denotes an explicit root, not an unknown parent.
BlockRelation = tuple[bytes, bytes | None]


def _deduplicate(relations: Iterable[BlockRelation]) -> dict[bytes, bytes | None]:
    result: dict[bytes, bytes | None] = {}
    for child, parent in relations:
        if child in result and result[child] != parent:
            raise ValueError("conflicting parents for the same block")
        result[child] = parent
    return result


def relations_for_slice(
    chain: Sequence[bytes], start: int, end: int
) -> list[BlockRelation]:
    """Construct relations for [start, end) using the unsliced hash chain."""
    if not 0 <= start <= end <= len(chain):
        raise ValueError("invalid block slice")
    relations = [
        (chain[index], chain[index - 1] if index else None)
        for index in range(start, end)
    ]
    return list(_deduplicate(relations).items())


def select_relations(
    relations: Iterable[BlockRelation], block_ids: Iterable[bytes]
) -> list[BlockRelation]:
    """Select by child after rank sampling or null-block filtering.

    A missing child is an error: it must never be interpreted as a root.
    Duplicate IDs are emitted only once, in their first requested order.
    """
    indexed = _deduplicate(relations)
    selected: dict[bytes, bytes | None] = {}
    for child in block_ids:
        if child not in indexed:
            raise ValueError("missing parent relation for block")
        selected[child] = indexed[child]
    return list(selected.items())


def map_relation_keys(
    relations: Iterable[BlockRelation], transform: Callable[[bytes], bytes]
) -> list[BlockRelation]:
    """Transform both keys into the same rank-specific Store namespace."""
    original = _deduplicate(relations)
    forward: dict[bytes, bytes] = {}
    reverse: dict[bytes, bytes] = {}

    def map_key(key: bytes) -> bytes:
        if key not in forward:
            mapped = transform(key)
            if mapped in reverse and reverse[mapped] != key:
                raise ValueError("rank key transformation collision")
            forward[key] = mapped
            reverse[mapped] = key
        return forward[key]

    mapped = (
        (map_key(child), None if parent is None else map_key(parent))
        for child, parent in original.items()
    )
    return list(_deduplicate(mapped).items())
