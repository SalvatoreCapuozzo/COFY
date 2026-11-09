"""Per-class image/instance counts across every node's train+val splits.

Surfaces classes that have little or no real data before a long run is
launched chasing a target that some classes may not be able to reach no
matter how long training runs.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from .config import FedYoloConfig


def compute_class_counts(cfg: FedYoloConfig) -> dict:
    img_count = {c: {"train": 0, "val": 0} for c in cfg.global_classes}
    inst_count = {c: {"train": 0, "val": 0} for c in cfg.global_classes}
    node_totals: dict[str, dict] = {}

    for node in cfg.nodes:
        data_yaml = yaml.safe_load(Path(node.data_yaml).read_text())
        base = Path(data_yaml.get("path", Path(node.data_yaml).parent))
        node_totals[node.name] = {"train_imgs": 0, "val_imgs": 0, "train_inst": 0, "val_inst": 0}
        for split in ("train", "val"):
            img_dir = base / data_yaml[split]
            lbl_dir = Path(str(img_dir).replace("images", "labels"))
            n_imgs = n_inst = 0
            for lbl_file in lbl_dir.glob("*.txt"):
                n_imgs += 1
                seen = set()
                for line in lbl_file.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    local_id = int(line.split()[0])
                    global_name = node.owned_classes[local_id]
                    inst_count[global_name][split] += 1
                    seen.add(global_name)
                    n_inst += 1
                for c in seen:
                    img_count[c][split] += 1
            node_totals[node.name][f"{split}_imgs"] = n_imgs
            node_totals[node.name][f"{split}_inst"] = n_inst

    owners = {c: [n.name for n in cfg.nodes if c in n.owned_classes] for c in cfg.global_classes}
    return {"img_count": img_count, "inst_count": inst_count, "node_totals": node_totals, "owners": owners}


def print_class_counts(cfg: FedYoloConfig, console: Console) -> dict:
    stats = compute_class_counts(cfg)

    table = Table(title="Per-class images / instances (train+val, all nodes)", show_lines=False)
    table.add_column("class", style="cyan")
    table.add_column("owner(s)")
    table.add_column("train imgs", justify="right")
    table.add_column("val imgs", justify="right")
    table.add_column("train inst", justify="right")
    table.add_column("val inst", justify="right")
    table.add_column("flag")

    for c in cfg.global_classes:
        ic, nc_ = stats["img_count"][c], stats["inst_count"][c]
        if nc_["val"] == 0 and nc_["train"] == 0:
            flag = "[bold red]zero real data anywhere -- unlearnable[/bold red]"
        elif nc_["val"] == 0:
            flag = "[red]no val instances -- unmeasurable[/red]"
        elif nc_["train"] < 250:
            flag = "[yellow]data-scarce[/yellow]"
        else:
            flag = ""
        table.add_row(
            c, ",".join(stats["owners"][c]),
            str(ic["train"]), str(ic["val"]), str(nc_["train"]), str(nc_["val"]), flag,
        )
    console.print(table)

    node_table = Table(title="Per-node totals")
    node_table.add_column("node")
    node_table.add_column("train imgs", justify="right")
    node_table.add_column("val imgs", justify="right")
    node_table.add_column("train instances", justify="right")
    node_table.add_column("val instances", justify="right")
    for name, t in stats["node_totals"].items():
        node_table.add_row(name, str(t["train_imgs"]), str(t["val_imgs"]), str(t["train_inst"]), str(t["val_inst"]))
    console.print(node_table)

    return stats
