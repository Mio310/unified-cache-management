"""Fixed-size QSA records and explicit auxiliary block-table mappings."""

import re
from dataclasses import dataclass, field

def qsa_role(name: str) -> str | None:
    if name.endswith(("compressed_key_cache", "compress_key_cache")):
        return "compressed"
    if name.endswith("raw_key_cache"):
        return "raw"
    return None


def layer_index(name: str) -> int:
    match = re.search(r"(?:layers|blocks|h)\.(\d+)(?:\.|$)", name)
    if match is None:
        raise ValueError(f"Cannot associate QSA cache with a model layer: {name}")
    return int(match.group(1))


@dataclass
class QSARecord:
    main_block: int
    attention: bool = False
    compressed: dict[int, int] = field(default_factory=dict)
    raw: dict[int, int] = field(default_factory=dict)


class QSATopology:
    def __init__(self, config, attention_names: set[str]):
        self.roles: dict[int, dict[str, tuple[str, int]]] = {}
        self.auxiliary_groups: set[int] = set()
        main_groups = set()
        for gid, group in enumerate(config.kv_cache_groups):
            if all(qsa_role(name) for name in group.layer_names):
                self.auxiliary_groups.add(gid)
            for name in group.layer_names:
                if name in attention_names:
                    main_groups.add(gid)
                role = qsa_role(name)
                if role:
                    roles = self.roles.setdefault(layer_index(name), {})
                    if role in roles:
                        raise ValueError(f"Duplicate QSA {role} cache for {name}")
                    roles[role] = (name, gid)
        if len(main_groups) != 1:
            raise ValueError("CUDA QSA HLA requires one main full-attention group")
        self.main_group = next(iter(main_groups))
        expected_layers = {layer_index(name) for name in attention_names}
        if set(self.roles) != expected_layers or any(
            set(roles) != {"compressed", "raw"} for roles in self.roles.values()
        ):
            raise ValueError(
                "Every main attention layer needs compressed and raw QSA caches"
            )
        self.compressed_groups = {r["compressed"][1] for r in self.roles.values()}
        self.raw_groups = {r["raw"][1] for r in self.roles.values()}
        if (
            self.compressed_groups & self.raw_groups
            or self.main_group in self.raw_groups
        ):
            raise ValueError(
                "QSA historical and circular caches need distinct block tables"
            )

    def record(self, main_block, index, group_blocks, *, raw_valid=False):
        def block(gid, pos):
            ids = group_blocks[gid]
            if pos >= len(ids) or ids[pos] <= 0:
                raise ValueError(
                    f"Missing QSA physical block: group={gid}, index={pos}"
                )
            return ids[pos]

        return QSARecord(
            main_block=main_block,
            attention=True,
            compressed={gid: block(gid, index) for gid in self.compressed_groups},
            raw={gid: block(gid, 0) for gid in self.raw_groups} if raw_valid else {},
        )


def make_qsa_records(
    topology,
    keys,
    blocks,
    primary_hashes,
    group_blocks,
    raw_index=None,
    *,
    is_load=False,
):
    """Keep original keys; put the final raw state directly in its record."""
    if len(keys) != len(blocks):
        raise ValueError("QSA transfer keys and blocks must have equal lengths")
    indices = {key: i for i, key in enumerate(primary_hashes)}
    records = []
    keys, blocks = list(keys), list(blocks)
    terminal = False
    for key, main_block in zip(keys, blocks):
        index = indices.get(key)
        if index is None:
            records.append(QSARecord(main_block))
            continue
        raw_valid = index == raw_index
        records.append(
            topology.record(main_block, index, group_blocks, raw_valid=raw_valid)
        )
        terminal = terminal or raw_valid
    if is_load and raw_index is not None and not terminal:
        raise ValueError("QSA resume point is missing its final attention block")
    return keys, blocks, records
