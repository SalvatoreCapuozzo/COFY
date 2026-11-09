"""True full-annotation oracle baseline -- for comparison only, not a mode
this system would ever run in practice (the whole point of federating is
that no single node has every class labeled on every image; if it did,
there'd be nothing to federate).

Distinct from `centralized.py`'s baseline: that one still respects each
node's own partial labels (masked loss, same information federation gets,
isolating the cost of aggregation specifically). The oracle here gets a
single "node" that owns every global class with real ground truth on every
image (built by `tools/split_dataset_into_nodes.py --n-nodes 1`, which
keeps the full multi-class annotation instead of splitting/masking it) --
the actual full-annotation ceiling `centralized.py`'s own docstring names
but never builds.

Implementation is almost nothing: `_class_mask(nc, owned_ids, device)`
produces an all-ones mask when `owned_ids` covers every global class, which
is mathematically identical to no masking at all. So this just reuses
`centralized.py`'s training loop unchanged, pointed at a single node that
owns everything.
"""

from __future__ import annotations

import dataclasses

from .centralized import train_centralized_baseline
from .config import FedYoloConfig, NodeConfig


def train_oracle_baseline(cfg: FedYoloConfig, oracle_data_yaml: str) -> dict:
    oracle_node = NodeConfig(
        name="oracle",
        data_yaml=oracle_data_yaml,
        owned_classes=list(cfg.global_classes),
    )
    oracle_cfg = dataclasses.replace(cfg, nodes=[oracle_node])
    return train_centralized_baseline(oracle_cfg)
