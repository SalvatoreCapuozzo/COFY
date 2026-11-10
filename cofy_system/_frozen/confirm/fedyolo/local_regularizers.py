"""Optional local-training additions for one node round (all opt-in, config-driven).

- ETF head (federation.etf_head): per-class rows frozen (class_geometry.py).
- Class spreading (federation.class_spread): client-side FedAwS-style penalty.
- FedProx (federation.prox_mu): mu/2 ||w - w_global||^2  (Li et al., MLSys 2020).
- SCAFFOLD (federation.scaffold, sync): gradient correction c - c_i and control
  variate update option II, c_i+ = c_i - c + (x - y_i)/(K*lr)  (Karimireddy et
  al., ICML 2020). Step size uses lr0 (the post-warmup rate).
- FL-JSDDC (federation.jsddc, sync; Frontiers in Neurorobotics 2026): local
  self-distillation from this node's model of the previous round, with
  lambda1=0.3 box L2, lambda2=0.3 class-probability term, lambda3=0.4 channel
  term on P3-P5 (weights 0.5/0.3/0.2); temperatures are not given in the paper
  (jsddc_tau, default 2). Per-class sigmoid outputs use a binary soft
  cross-entropy for the class term. The paper's drift-compensation update is
  under-specified, so it follows FedDC (Gao et al., CVPR 2022), which has the
  same "upload local + drift" form: penalty alpha/2 ||theta + h - w||^2 during
  training, h <- h + (theta - w) afterwards, upload theta + h.

Node-persistent state (SCAFFOLD's c_i, JSDDC's previous model and drift h)
travels in `node_state`; server-side state (SCAFFOLD's c) in `server_state`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .class_geometry import freeze_etf_head, spread_penalty
from .model import build_model


class LocalExtras:
    def __init__(self, model, cfg, global_state_dict: dict, owned_ids: list[int], device,
                 node_state: dict | None, server_state: dict | None):
        fed = cfg.federation
        self.cfg, self.fed, self.device, self.owned_ids = cfg, fed, device, owned_ids
        self.node_state = dict(node_state or {})
        self.server_state = server_state or {}
        self.global_sd = global_state_dict
        if fed.etf_head:
            freeze_etf_head(model)
        self.trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        need_global = fed.prox_mu > 0 or fed.jsddc or fed.scaffold
        self.w_global = ({n: global_state_dict[n].to(device, torch.float32) for n in self.trainable}
                         if need_global else None)
        self.scaffold = None
        if fed.scaffold:
            c = self.server_state.get("scaffold_c")
            ci = self.node_state.get("scaffold_ci")
            self.scaffold = {n: ((c[n] if c else torch.zeros_like(p)).to(device, torch.float32)
                                 - (ci[n] if ci else torch.zeros_like(p)).to(device, torch.float32))
                             for n, p in self.trainable.items()}
            self._c = c
            self._ci = ci
        self.teacher = None
        self.drift = None
        if fed.jsddc:
            hist = self.node_state.get("jsddc_hist")
            if hist is not None:
                self.teacher = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz,
                                           pretrained=cfg.model.pretrained,
                                           class_names=cfg.global_classes).to(device)
                self.teacher.load_state_dict(hist)
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
            h = self.node_state.get("jsddc_drift")
            self.drift = {n: (h[n].to(device, torch.float32) if h else torch.zeros_like(p, dtype=torch.float32))
                          for n, p in self.trainable.items()}

    @property
    def active(self) -> bool:
        f = self.fed
        return f.etf_head or f.class_spread > 0 or f.prox_mu > 0 or f.scaffold or f.jsddc

    # ── extra loss terms ────────────────────────────────────────────────────
    def extra_loss(self, model, parsed: dict | None, img: torch.Tensor) -> torch.Tensor | None:
        f, terms = self.fed, []
        if f.class_spread > 0:
            terms.append(f.class_spread * spread_penalty(model, self.owned_ids, self.global_sd, self.cfg.nc))
        if f.prox_mu > 0:
            terms.append(0.5 * f.prox_mu * sum(((p - self.w_global[n]) ** 2).sum()
                                               for n, p in self.trainable.items()))
        if f.jsddc:
            terms.append(0.5 * f.jsddc_alpha * sum(((p + self.drift[n] - self.w_global[n]) ** 2).sum()
                                                   for n, p in self.trainable.items()))
            if self.teacher is not None and parsed is not None:
                terms.append(self._self_distillation(parsed, img))
        return sum(terms) if terms else None

    def _self_distillation(self, parsed: dict, img: torch.Tensor) -> torch.Tensor:
        tau = self.fed.jsddc_tau
        with torch.no_grad():
            t = self.teacher.forward(img)
        t = t[1] if isinstance(t, tuple) else t
        box = F.mse_loss(parsed["boxes"], t["boxes"].detach())
        s_cls, t_cls = parsed["scores"], t["scores"].detach()
        cls = F.binary_cross_entropy_with_logits(s_cls / tau, (t_cls / tau).sigmoid()) * tau ** 2
        chan = 0.0
        for eps, fs, ft in zip((0.5, 0.3, 0.2), parsed["feats"], t["feats"]):
            ps = F.log_softmax(fs.mean(dim=(2, 3)) / tau, dim=1)
            pt = F.softmax(ft.detach().mean(dim=(2, 3)) / tau, dim=1)
            chan = chan + eps * F.kl_div(ps, pt, reduction="batchmean") * tau ** 2
        return 0.3 * box + 0.3 * cls + 0.4 * chan

    # ── gradient correction ─────────────────────────────────────────────────
    def after_backward(self) -> None:
        if self.scaffold is not None:
            for n, p in self.trainable.items():
                if p.grad is not None:
                    p.grad.add_(self.scaffold[n])

    # ── end of round ────────────────────────────────────────────────────────
    def finalize(self, state_dict: dict, steps: int) -> tuple[dict, dict | None]:
        """Adjust the uploaded state_dict (JSDDC), return (node_state, extras)."""
        extras = None
        trained = {n: state_dict[n].to(torch.float32) for n in self.trainable}
        if self.fed.scaffold:
            lr = self.fed.lr0
            k = max(steps, 1)
            new_ci, delta = {}, {}
            for n, p in self.trainable.items():
                c = (self._c[n] if self._c else torch.zeros_like(trained[n])).to(torch.float32)
                ci = (self._ci[n] if self._ci else torch.zeros_like(trained[n])).to(torch.float32)
                new = ci - c + (self.w_global[n].cpu() - trained[n]) / (k * lr)
                new_ci[n], delta[n] = new, new - ci
            self.node_state["scaffold_ci"] = new_ci
            extras = {"scaffold_delta_c": delta}
        if self.fed.jsddc:
            self.node_state["jsddc_hist"] = {k: v.clone() for k, v in state_dict.items()}
            new_h = {}
            for n in self.trainable:
                h = self.drift[n].cpu() + (trained[n] - self.w_global[n].cpu())
                new_h[n] = h
                state_dict[n] = (trained[n] + h).to(state_dict[n].dtype)
            self.node_state["jsddc_drift"] = new_h
        return (self.node_state or None), extras
