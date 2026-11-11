"""Peer behavioural validation with per-class rollback (candidate defence P3; opt-in:
federation.robust.peer_check). Exploratory, after the registered study (analysis/DEFENCE_SOTA.md).

Why: a node that reverses the training of its own class (sign flip) makes that class fire on every
image. No party can check the class's rows during training, since only the owner has data for the
class, but every node can see the effect on its own images, without labels.

How (BaFFLe, ICDCS 2021, and CrowdGuard, NDSS 2024, adapted to class ownership):
  1. after every `peer_every`-th new global version, each node runs the model on a fixed sample of
     `peer_images` of its own training images (images only, no labels) and reports, for every class,
     its coverage: the share of the head's locations (anchor points, all scales) where the class scores
     >= `peer_conf`, averaged over the images (one number per class is what a node discloses);
  2. a non-owner votes against a class whose coverage on its images is at least `peer_cover` and at
     least `peer_ratio` times the median coverage of the other classes there (one class firing much
     more widely than the rest, not the whole model in a transient); a class is flagged when a
     majority of the nodes that do not own it vote against it; its owner does not vote;
  3. a class flagged at `peer_patience` consecutive checks is rolled back to its rows at the last check
     it passed and quarantined: from then on its owner's row updates are discarded (the rows stay at
     the accepted values), and the stored versions the final model averages get the same rows;
  4. a class that passes a check has its rows recorded as the last accepted ones.
The rows only ever come from their owner (R1), the server holds no data (R2: the rates are computed at
the nodes; this simulation runs that step on the nodes' own images), and no node changes another node's
class (R3). An attacker can lie in its reports (`peer_lie`: it reports that every class it does not own
covers every location); the majority among the non-owners stops it from rolling back an honest class.
The node named as attacker lies also in runs with attack "none", so that the clean twins test the vote.
Every check is appended to <output_dir>/peer_log.jsonl.

P3b (`peer_exclude`): the rollback alone leaves the owner's updates to the shared layers and BatchNorm
statistics in the merge, and in CWT and FedAsync those reversed updates make the frozen rows fire again.
With exclusion, the owner of a rolled-back class is excluded from then on: the server discards its uploads
on arrival (no new version in the asynchronous schemes; left out of the round in parallel sync, its rows kept
at the previous version), and its reports no longer count in the vote. Its rows still only ever hold values
it produced (R3); nothing it sends after the exclusion reaches the model.

Why coverage and not the share of images with a detection: in a pilot on the kept final models of the
robustness study (WBC, before any P3 run), four of the seven classes were detected on 80-100 % of the
images of nodes that do not own them in clean models too, because those cells do appear there, without
labels. Coverage separates: at most 0.1 % in the 18 clean and backdoored models, 16-90 % for the
sign-flipped class (seed 0 of the three schemes). `peer_cover` = 0.02 lies between the two. In a short
smoke run (15 % of the data, 3 cycles) every class rose together to 0.2-3.6 % on every honest node in one
early version, and fell back at the next: hence the ratio to the other classes and the patience of 2.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .robust import attacker_name


def _letterbox_u8(path, size):
    """Image -> (3, size, size) uint8 tensor, letterboxed with grey 114 (as Ultralytics validation)."""
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = im[:, :, ::-1]
    h, w = im.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = torch.full((3, size, size), 114, dtype=torch.uint8)
    canvas[:, top:top + nh, left:left + nw] = torch.from_numpy(im.copy()).permute(2, 0, 1)
    return canvas


def node_images(node, cfg, n: int, seed: int) -> torch.Tensor:
    """A fixed sample of `n` of the node's own training images (no labels read)."""
    import yaml
    root = Path.cwd()
    meta = yaml.safe_load((root / node.data_yaml).read_text())
    base = Path(meta["path"]) if Path(meta["path"]).is_absolute() else (root / node.data_yaml).parent / meta["path"]
    files = sorted(p for p in (base / meta["train"]).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    random.Random(f"peer-{seed}-{node.name}").shuffle(files)
    out = []
    for p in files:
        t = _letterbox_u8(p, cfg.model.imgsz)
        if t is not None:
            out.append(t)
        if len(out) >= n:
            break
    return torch.stack(out)


@torch.no_grad()
def class_coverage(model, images: torch.Tensor, nc: int, conf: float, device, batch: int = 16) -> list[float]:
    """Per class, the share of the head's locations where it scores >= conf, averaged over the images
    (raw head output, before NMS, so that one class filling the detection budget hides no other)."""
    tot = torch.zeros(nc)
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device).float() / 255.0
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = model(x)
        p = (p[0] if isinstance(p, (list, tuple)) else p).float()       # (B, 4 + nc, anchors)
        tot += (p[:, 4:4 + nc] >= conf).float().mean(dim=2).sum(0).cpu()
    return (tot / len(images)).tolist()


class PeerCheck:
    """One instance per server; `after_merge` is called with every new global version."""

    def __init__(self, cfg, model, init_sd: dict, out_dir):
        from .model import per_class_param_names
        self.cfg, self.rc = cfg, cfg.federation.robust
        self.nc = cfg.nc
        self.attacker = attacker_name(cfg)   # with attack none (clean twins) it trains honestly but may still lie
        self.pairs = per_class_param_names(model)
        self.registry = {n.name: set(n.owned_global_ids(cfg.global_classes)) for n in cfg.nodes}
        self.owner = {c: nm for nm, cs in self.registry.items() for c in cs}
        self.images = {n.name: node_images(n, cfg, self.rc.peer_images, cfg.seed) for n in cfg.nodes}
        self.accepted = {c: self._rows(init_sd, c) for c in range(self.nc)}
        self.strikes = {c: 0 for c in range(self.nc)}
        self.quarantined: dict[int, int] = {}      # class -> version at which it was rolled back
        self.excluded: dict[str, int] = {}         # P3b: node -> version from which its uploads are discarded
        self.n_versions = 0
        self.model = None
        self.log_path = Path(out_dir) / "peer_log.jsonl"

    def _rows(self, sd, c):
        return {k: sd[k][c].detach().cpu().clone() for p in self.pairs for k in p}

    def _set_rows(self, sd, c, rows):
        for k, v in rows.items():
            t = sd[k].clone()
            t[c] = v.to(device=t.device, dtype=t.dtype)
            sd[k] = t

    def keep_excluded_rows(self, sd: dict, prev: dict) -> None:
        """Parallel sync, P3b: the classes of excluded nodes keep their rows of the previous version
        (the merge left those nodes out, so no other node may supply the rows)."""
        for nm in self.excluded:
            for c in self.registry[nm]:
                self._set_rows(sd, c, self._rows(prev, c))

    def _rates(self, sd):
        from .model import build_model
        dev = torch.device(self.cfg.federation.device)   # resolved at load: cuda, mps or cpu
        if self.model is None:
            self.model = build_model(self.cfg.model.arch, self.nc, self.cfg.model.imgsz, pretrained=False,
                                     class_names=self.cfg.global_classes)
        self.model.load_state_dict(sd)
        self.model.eval().to(dev)
        try:
            return {nm: class_coverage(self.model, imgs, self.nc, self.rc.peer_conf, dev)
                    for nm, imgs in self.images.items() if nm not in self.excluded}
        finally:
            self.model.to("cpu")          # the GPU is the training's between checks
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    def _votes_against(self, cov: list[float], c: int) -> bool:
        others = sorted(x for i, x in enumerate(cov) if i != c)
        med = others[len(others) // 2] if len(others) % 2 else (others[len(others) // 2 - 1] + others[len(others) // 2]) / 2
        return cov[c] >= self.rc.peer_cover and cov[c] >= self.rc.peer_ratio * med

    def after_merge(self, sd: dict, stored: list[dict], version: int) -> dict:
        """Enforce the quarantine on the new global `sd` (in place), run the check if due, and roll back
        newly flagged classes in `sd` and in the stored versions the final model averages."""
        for c in self.quarantined:
            self._set_rows(sd, c, self.accepted[c])
        self.n_versions += 1
        if self.n_versions % max(1, self.rc.peer_every):
            return sd
        reported = {nm: r for nm, r in self._rates(sd).items() if nm not in self.excluded}   # P3b: no vote
        if self.attacker and self.rc.peer_lie and self.attacker in reported:   # the attacker reports every class it does not own as covering everything
            reported[self.attacker] = [1.0 if c not in self.registry[self.attacker] else r
                                       for c, r in enumerate(reported[self.attacker])]
        flagged, rolled = [], []
        for c in range(self.nc):
            if c in self.quarantined:
                continue
            voters = [nm for nm in reported if nm != self.owner.get(c)]
            votes = sum(self._votes_against(reported[nm], c) for nm in voters)
            if votes >= len(voters) // 2 + 1:
                flagged.append(c)
                self.strikes[c] += 1
                if self.strikes[c] >= self.rc.peer_patience:
                    self.quarantined[c] = version
                    rolled.append(c)
                    if self.rc.peer_exclude and c in self.owner:
                        self.excluded.setdefault(self.owner[c], version)
                    self._set_rows(sd, c, self.accepted[c])
                    for st in stored:
                        self._set_rows(st, c, self.accepted[c])
            else:
                self.strikes[c] = 0
                self.accepted[c] = self._rows(sd, c)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"version": version, "coverage": {nm: [round(r, 5) for r in v] for nm, v in reported.items()},
                                "flagged": flagged, "rolled_back": rolled,
                                "quarantined": sorted(self.quarantined), "excluded": sorted(self.excluded)}) + "\n")
        return sd
