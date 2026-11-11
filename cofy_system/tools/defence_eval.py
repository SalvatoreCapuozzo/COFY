#!/usr/bin/env python3
"""Candidate defences for the attacks the study's defence does not stop (analysis/DEFENCE_SOTA.md).
Exploratory, after the registered study. Run by the queue as follow-up protocols of
tools/followup_runs.py, one job per seed; each writes runs/variants/<entry>/<ds>/s<seed>/summary.json.

  bnpool   (P1, entry p1_bnpool) robust BatchNorm re-estimation in the calibration.
           For each kept final model of the seed -- the defended clean runs and the BatchNorm-poisoning
           runs re-run with their final model kept (rb_*_bnpoison*_budget) -- every node computes the
           BatchNorm statistics of the model on its own training images (cofy/calibrate.py), and the
           server pools them by the registered rule (count-weighted mean) or by the coordinate-wise
           median, with every node honest or with the attacker also poisoning the statistics it sends
           (mean + 3 sd, variance x9: the upload attack bnpoison, cofy/robust.py). Only the BatchNorm
           step of the calibration is applied, so that its effect is isolated. mAP50 on the evaluation
           set, overall, on the attacker's classes and on the others'.

  cleanse  (P2, entry p2_cleanse) run-time screening of inputs at a node (Detector Cleanse, from
           BadDet, Chan et al. 2022). A node checks a detection by blending the detected region with
           N crops of objects from its own training images (alpha = 0.5) and averaging the entropy of
           the class scores the model gives the blended region (YOLO class scores are sigmoids: they are
           normalized to sum to one, and the entropy is divided by log(nc)). A trigger keeps its class
           whatever is blended in, so its entropy is abnormally low. The node accepts a detection when
           the average entropy lies within mean +/- 2 sd of the entropies of the detections on its own
           validation images, and flags the image otherwise. The screening node is an honest one (the
           node after the attacker) and checks the detections of classes it does not own, since it
           can vouch for its own. Rule (one-sided, as the object-generation backdoor gives low entropy):
           a detection is flagged when its average entropy is below the 5th percentile of the entropies
           of the detections on the node's own validation images, i.e. a 5 % false-rejection target on
           the node's own data (Detector Cleanse sets its band from clean data the same way). Its
           two-sided band of mean +/- 2 sd flagged 40-50 % of clean images in a pilot (seed 0, 10
           images), because clean detections of other nodes' classes vary upwards with the class; a
           fixed fraction of the median (0.5) was tried next and flagged 30 % on a clean concurrent
           model (seed 1), so the percentile rule, set from the node's own data only, is used. Every
           entropy is stored, so any other threshold can be evaluated afterwards.
           Threshold-free: AUROC of the entropy between the backdoor's
           detections on the trigger and the tested detections on clean images. Measured on 100
           triggered evaluation images (the attack success rate with and without screening) and on the
           same 100 images without the trigger (false alarms), for the backdoor runs and their clean
           twins.

    python tools/followup_runs.py --protocol bnpool --seeds 0 1 2 --datasets wbc
    python tools/followup_runs.py --protocol cleanse --seeds 0 1 2 --datasets wbc
    python tools/defence_eval.py --report         # analysis/DEFENCE_RESULTS.md from the summaries

  P4 and P3 are training-time defences (cofy/robust.py history_bound, cofy/peercheck.py), run as
  robustness variants rb_*_p4_budget / rb_*_p3_budget by the queue; --report adds their tables.
"""
import dataclasses, json, math, random, shutil, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VAR = ROOT / "runs/variants"

P1_MODELS = ["rb_seq_clean_ora", "rb_conc_clean_ora", "rb_par_clean_ora",
             "rb_seq_bnpoison_budget", "rb_seq_bnpoison_ora_budget",
             "rb_conc_bnpoison_budget", "rb_conc_bnpoison_ora_budget"]
P2_MODELS = [f"rb_{s}_{a}{d}" for a in ("backdoor", "clean") for s in ("seq", "conc", "par") for d in ("", "_ora")]
N_BLEND, ALPHA, N_FEAT, N_TEST, N_REF, MAX_DET_TESTED, CONF = 32, 0.5, 64, 100, 100, 5, 0.25


def log(msg):
    print(time.strftime("%m-%d %H:%M"), msg, flush=True)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def final_model(name, ds, s):
    run = VAR / f"{name}/{ds}/s{s}"
    ckpt = next((run / m / "global_final.pt" for m in ("async", "sync") if (run / m / "global_final.pt").exists()), None)
    if ckpt is None:
        return None, None
    import torch
    from cofy.config import load_config
    cfg = load_config(ROOT / f"configs/variants/{name}/{ds}_s{s}.yaml")
    return cfg, torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]


def attacker_of(cfg, s):
    from cofy.robust import attacker_name
    a = attacker_name(cfg)
    if not a:   # data attacks and clean runs name no attacker in the config: it rotates with the seed
        a = next((n.name for n in cfg.nodes if "data/attack/" in n.data_yaml), "") or cfg.nodes[s % len(cfg.nodes)].name
    return a


def _tick(entry, ds, s, t0, k, n):
    """A progress line for tools/queue_status.py after model k of n (else the job looks stalled)."""
    try:
        from followup_runs import tick
        el = (time.time() - t0) / 60
        tick(entry, ds, s, el, el / k * (n - k), f"model {k}/{n} done")
    except Exception:
        pass


def centralized(ds, s):
    return ((load(VAR / f"v2_central/{ds}/s{s}/summary.json") or {}).get("centralized") or {}).get("map50")


# ── P1: robust BatchNorm re-estimation ───────────────────────────────────────────────────────────
def run_bnpool(ds, s, out):
    import torch
    from cofy.calibrate import node_bn_stats, pool_bn_stats
    from cofy.evaluate import evaluate_state_dict, per_class_map50
    from cofy.robust import poison_bn_stats
    res, cent, t_start = {}, centralized(ds, s), time.time()
    for i, name in enumerate(P1_MODELS, 1):
        cfg, sd = final_model(name, ds, s)
        if cfg is None:
            log(f"  {name} {ds} s{s}: no final model -- skipped")
            continue
        cfg = dataclasses.replace(cfg, output_dir=str(out / "_work"))
        att = attacker_of(cfg, s)
        own = set(next(n.owned_classes for n in cfg.nodes if n.name == att))
        dev = torch.device(cfg.federation.device)

        def score(st, tag):
            m = evaluate_state_dict(cfg, st, name=tag)
            pc = per_class_map50(m)
            r = {"map50": float(m.box.map50),
                 "attacker_classes": statistics.mean(pc[c] for c in own),
                 "other_classes": statistics.mean(v for c, v in pc.items() if c not in own)}
            if cent:
                r["fed_cent"] = r["map50"] / cent * 100
            log(f"  {name} s{s} {tag:16s} mAP50 {r['map50']:.4f}")
            return r

        r = {"attacker": att, "uncalibrated": score(sd, "uncalibrated")}
        stats = [node_bn_stats(n, cfg, sd, dev) for n in cfg.nodes]
        poisoned = [(poison_bn_stats(st, 3.0) if n.name == att else st, k) for n, (st, k) in zip(cfg.nodes, stats)]
        for rule in ("mean", "median"):
            r[f"{rule}_honest"] = score(pool_bn_stats(sd, stats, rule), f"{rule}_honest")
            r[f"{rule}_poisoned"] = score(pool_bn_stats(sd, poisoned, rule), f"{rule}_poisoned")
        res[name] = r
        shutil.rmtree(out / "_work", ignore_errors=True)
        _tick("p1_bnpool", ds, s, t_start, i, len(P1_MODELS))
    return {"bnpool": res, "centralized_map50": cent}


# ── P2: run-time input screening (Detector Cleanse) ──────────────────────────────────────────────
def _letterbox(path, size):
    import cv2, torch
    im = cv2.imread(str(path))[:, :, ::-1]
    h, w = im.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = torch.full((3, size, size), 114 / 255.0)
    canvas[:, top:top + nh, left:left + nw] = torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 255.0
    return canvas, r, left, top


def _crops(node, n, rng):
    """Crops of the node's own labelled training objects (its 'clean features')."""
    import cv2, torch, yaml
    meta = yaml.safe_load((ROOT / node.data_yaml).read_text())
    base = Path(meta["path"]) if Path(meta["path"]).is_absolute() else (ROOT / node.data_yaml).parent / meta["path"]
    imgs = sorted((base / meta["train"]).glob("*"))
    rng.shuffle(imgs)
    out = []
    for p in imgs:
        lab = base / "labels" / Path(meta["train"]).name / (p.stem + ".txt")
        if not lab.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]
        for line in lab.read_text().split("\n"):
            v = line.split()
            if len(v) != 5:
                continue
            x, y, w, h = map(float, v[1:])
            x1, y1, x2, y2 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                out.append(torch.from_numpy(im[max(0, y1):y2, max(0, x1):x2, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
            if len(out) >= n:
                return out
    return out


class Screen:
    def __init__(self, model, nc, feats, device):
        self.model, self.nc, self.feats, self.dev = model, nc, feats, device

    def detect(self, img):
        import torch
        from ultralytics.utils.nms import non_max_suppression
        with torch.no_grad(), torch.autocast("cuda", enabled=self.dev.type == "cuda"):
            p = self.model(img[None].to(self.dev))[0].float()
        return non_max_suppression(p, conf_thres=CONF, iou_thres=0.7, max_det=300)[0].cpu()

    def entropy(self, img, box):
        """Average normalized class-score entropy of the region `box` blended with N_BLEND crops."""
        import torch
        import torch.nn.functional as F
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        S = img.shape[-1]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(S, x2), min(S, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        batch = img[None].repeat(N_BLEND, 1, 1, 1)
        for i in range(N_BLEND):
            f = F.interpolate(self.feats[i % len(self.feats)][None], size=(y2 - y1, x2 - x1), mode="bilinear",
                              align_corners=False)[0]
            batch[i, :, y1:y2, x1:x2] = ALPHA * batch[i, :, y1:y2, x1:x2] + (1 - ALPHA) * f
        with torch.no_grad(), torch.autocast("cuda", enabled=self.dev.type == "cuda"):
            p = self.model(batch.to(self.dev))[0].float()            # (N, 4 + nc, anchors), xywh in pixels
        xy, wh = p[:, :2], p[:, 2:4]
        a1, a2 = xy - wh / 2, xy + wh / 2
        b = torch.tensor([x1, y1, x2, y2], device=p.device, dtype=p.dtype)
        iw = (torch.minimum(a2[:, 0], b[2]) - torch.maximum(a1[:, 0], b[0])).clamp_min(0)
        ih = (torch.minimum(a2[:, 1], b[3]) - torch.maximum(a1[:, 1], b[1])).clamp_min(0)
        inter = iw * ih
        iou = inter / ((wh[:, 0] * wh[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
        cls = p[:, 4:]
        pick = torch.where(iou > 0.5, cls.max(1).values, torch.full_like(iou, -1.0))
        idx = torch.where(pick.max(1).values >= 0, pick.argmax(1), iou.argmax(1))
        sc = cls[torch.arange(len(idx)), :, idx].clamp_min(1e-12)
        q = sc / sc.sum(1, keepdim=True)
        h = -(q * q.log()).sum(1) / math.log(self.nc)
        return float(h.mean())


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def run_cleanse(ds, s, out):
    import torch, yaml
    from cofy.model import build_model
    res, t_start = {}, time.time()
    for i, name in enumerate(P2_MODELS, 1):
        cfg, sd = final_model(name, ds, s)
        if cfg is None:
            log(f"  {name} {ds} s{s}: no final model -- skipped")
            continue
        t0 = time.time()
        rng = random.Random(1000 + s)
        att = attacker_of(cfg, s)
        names = [n.name for n in cfg.nodes]
        scr = cfg.nodes[(names.index(att) + 1) % len(names)]
        scr_own = {cfg.global_classes.index(c) for c in scr.owned_classes}
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=False, class_names=cfg.global_classes)
        model.load_state_dict(sd)
        model.eval().to(dev)
        S = cfg.model.imgsz
        sc = Screen(model, cfg.nc, _crops(scr, N_FEAT, rng), dev)
        # the screening node's reference: entropies of the detections on its own validation images
        meta = yaml.safe_load((ROOT / scr.data_yaml).read_text())
        vbase = Path(meta["path"]) if Path(meta["path"]).is_absolute() else (ROOT / scr.data_yaml).parent / meta["path"]
        vimgs = sorted((vbase / meta["val"]).glob("*"))
        rng.shuffle(vimgs)
        ref = []
        for p in vimgs[:N_REF]:
            img = _letterbox(p, S)[0]
            for d in sc.detect(img)[:MAX_DET_TESTED]:
                h = sc.entropy(img, d[:4])
                if h is not None:
                    ref.append(h)
        m, sd_h, med = statistics.mean(ref), statistics.stdev(ref), statistics.median(ref)
        lo = sorted(ref)[int(0.05 * len(ref))]

        def screen(img):
            """(flagged, detections with their entropy) for one image, as the screening node sees it."""
            dets, flagged = [], False
            for d in sc.detect(img)[:MAX_DET_TESTED]:
                c = int(d[5])
                h = None if c in scr_own else sc.entropy(img, d[:4])
                bad = h is not None and h < lo
                flagged |= bad
                dets.append((d[:4].tolist(), c, float(d[4]), h, bad))
            return flagged, dets

        trig_dir = ROOT / f"data/attack/{ds}/_eval_trigger_{att}"
        t = json.loads((trig_dir / "triggers.json").read_text())
        clean_dir = ROOT / f"data/study/{ds}/_eval/images/val"
        files = sorted(t["boxes"])
        rng.shuffle(files)
        files = files[:N_TEST]
        fired = caught = flag_trig = flag_clean = 0
        h_trigger, h_clean = [], []
        for f in files:
            img, r, left, top = _letterbox(trig_dir / "images" / f, S)
            H, W = _letterbox_shape(trig_dir / "images" / f)
            x, y, w, h = t["boxes"][f]
            tb = [(x - w / 2) * W * r + left, (y - h / 2) * H * r + top, (x + w / 2) * W * r + left, (y + h / 2) * H * r + top]
            flagged, dets = screen(img)
            flag_trig += flagged
            hits = [d for d in dets if d[1] == t["target_id"] and _iou(d[0], tb) >= 0.5]
            if hits:
                fired += 1
                caught += any(d[4] for d in hits)
                h_trigger += [d[3] for d in hits if d[3] is not None]
            fc, dets_c = screen(_letterbox(clean_dir / f, S)[0])
            flag_clean += fc
            h_clean += [d[3] for d in dets_c if d[3] is not None]
        n = len(files)
        res[name] = {"attacker": att, "screening_node": scr.name, "n_images": n, "ref_detections": len(ref),
                     "ref_entropy_mean": m, "ref_entropy_sd": sd_h, "ref_entropy_median": med, "threshold": lo,
                     "auroc": _auroc(h_clean, h_trigger), "n_trigger_dets": len(h_trigger), "n_clean_dets": len(h_clean),
                     "entropies": {"reference": [round(x, 4) for x in ref], "clean": [round(x, 4) for x in h_clean],
                                   "trigger": [round(x, 4) for x in h_trigger]},
                     "asr": fired / n, "asr_after_screening": (fired - caught) / n,
                     "caught_share": caught / fired if fired else None,
                     "triggered_images_flagged": flag_trig / n, "false_alarms": flag_clean / n,
                     "trigger_entropy_mean": statistics.mean(h_trigger) if h_trigger else None}
        log(f"  {name} s{s}: ASR {fired / n:.0%} -> {(fired - caught) / n:.0%} after screening, "
            f"false alarms {flag_clean / n:.0%}, AUROC {res[name]['auroc'] or float('nan'):.3f} (threshold {lo:.3f}, trigger "
            f"{res[name]['trigger_entropy_mean'] if h_trigger else float('nan'):.3f}) [{(time.time() - t0) / 60:.0f} min]")
        del model
        torch.cuda.empty_cache()
        _tick("p2_cleanse", ds, s, t_start, i, len(P2_MODELS))
    return {"cleanse": res, "params": {"n_blend": N_BLEND, "alpha": ALPHA, "n_feat": N_FEAT, "n_test": N_TEST,
                                       "n_ref": N_REF, "conf": CONF, "rule": "entropy < 5th percentile of the reference"}}


def _auroc(neg, pos):
    """P(a clean detection's entropy > a backdoor detection's), ties counted half (None if either is empty)."""
    if not neg or not pos:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in neg for b in pos) / (len(neg) * len(pos))


def _letterbox_shape(path):
    import cv2
    return cv2.imread(str(path)).shape[:2]


# ── report ───────────────────────────────────────────────────────────────────────────────────────
def report(ds="wbc"):
    mean = lambda xs: statistics.mean(xs) if xs else None
    fmt = lambda v, f="{:.3f}": "-" if v is None else f.format(v)
    lines = ["# Candidate defences: results", "",
             "Generated by `tools/defence_eval.py --report` (exploratory, after the registered study; "
             "procedures in `analysis/DEFENCE_SOTA.md`). WBC, seeds 0-2, means over seeds.", "",
             "## P1: robust BatchNorm re-estimation (BatchNorm step of the calibration only)", "",
             "mAP50 (Fed:Cent against the uncalibrated centralized reference of the same seed). "
             "'poisoned': the attacker also poisons the statistics it sends in the calibration.", "",
             "| model | n | uncalibrated | mean, honest | mean, poisoned | median, honest | median, poisoned |",
             "|---|---|---|---|---|---|---|"]
    p1 = {}
    for s in (0, 1, 2):
        for k, v in ((load(VAR / f"p1_bnpool/{ds}/s{s}/summary.json") or {}).get("bnpool") or {}).items():
            p1.setdefault(k, []).append(v)
    for k in P1_MODELS:
        rs = p1.get(k, [])
        if not rs:
            continue
        cells = []
        for col in ("uncalibrated", "mean_honest", "mean_poisoned", "median_honest", "median_poisoned"):
            mp, fc = mean([r[col]["map50"] for r in rs]), mean([r[col].get("fed_cent") for r in rs if r[col].get("fed_cent")])
            cells.append(f"{fmt(mp)} ({fmt(fc, '{:.0f} %')})")
        lines.append(f"| {k} | {len(rs)} | " + " | ".join(cells) + " |")
    lines += ["", "Per seed:", ""]
    for k in P1_MODELS:
        for i, r in enumerate(p1.get(k, [])):
            lines.append(f"- {k} #{i}: " + ", ".join(f"{c} {r[c]['map50']:.3f}" for c in
                         ("uncalibrated", "mean_honest", "mean_poisoned", "median_honest", "median_poisoned")))
    lines += ["", "## P2: run-time input screening (Detector Cleanse)", "",
              "ASR: share of 100 triggered evaluation images on which the attacker's class is detected on the "
              "trigger. After screening: those not flagged by the screening node. False alarms: share of the "
              "same images without the trigger that the screening node flags.", "",
              "| model | n | ASR | ASR after screening | backdoor detections caught | false alarms | AUROC | entropy: threshold / trigger |",
              "|---|---|---|---|---|---|---|---|"]
    p2 = {}
    for s in (0, 1, 2):
        for k, v in ((load(VAR / f"p2_cleanse/{ds}/s{s}/summary.json") or {}).get("cleanse") or {}).items():
            p2.setdefault(k, []).append(v)
    for k in P2_MODELS:
        rs = p2.get(k, [])
        if not rs:
            continue
        lines.append(f"| {k} | {len(rs)} | {fmt(mean([r['asr'] for r in rs]), '{:.0%}')} | "
                     f"{fmt(mean([r['asr_after_screening'] for r in rs]), '{:.0%}')} | "
                     f"{fmt(mean([r['caught_share'] for r in rs if r['caught_share'] is not None]), '{:.0%}')} | "
                     f"{fmt(mean([r['false_alarms'] for r in rs]), '{:.0%}')} | "
                     f"{fmt(mean([r['auroc'] for r in rs if r.get('auroc') is not None]))} | "
                     f"{fmt(mean([r['threshold'] for r in rs]))} / "
                     f"{fmt(mean([r['trigger_entropy_mean'] for r in rs if r['trigger_entropy_mean'] is not None]))} |")
    lines += ["", "Post hoc, per detection (from the stored entropies; the thresholds below were not set in advance): "
              "share of the backdoor's detections on the trigger with entropy below a fixed threshold, and share of "
              "the checked detections on clean images below it.", "",
              "| model | caught at 0.01 | clean flagged at 0.01 | caught at 0.05 | clean flagged at 0.05 |",
              "|---|---|---|---|---|"]
    for k in P2_MODELS:
        rs = p2.get(k, [])
        if not rs or "entropies" not in rs[0]:
            continue
        trig = [h for r in rs for h in r["entropies"]["trigger"]]
        clean = [h for r in rs for h in r["entropies"]["clean"]]
        share = lambda xs, t: (sum(h < t for h in xs) / len(xs)) if xs else None
        lines.append(f"| {k} | {fmt(share(trig, 0.01), '{:.0%}')} | {fmt(share(clean, 0.01), '{:.0%}')} | "
                     f"{fmt(share(trig, 0.05), '{:.0%}')} | {fmt(share(clean, 0.05), '{:.0%}')} |")
    lines += _report_p3_p4(ds, fmt, mean)
    (ROOT / "analysis/DEFENCE_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# P3 / P4 runs and the runs they are compared with (same attack and defence, without the candidate)
P4_RUNS = [("rb_conc_bnpoison_ora_p4_budget", "rb_conc_bnpoison_ora_budget"),
           ("rb_conc_clean_ora_p4_budget", "rb_conc_clean_ora")]
P3_RUNS = [(f"rb_{sc}_{a}_ora_p3_budget", f"rb_{sc}_{a}_ora_budget" if a == "signflip" else f"rb_{sc}_clean_ora")
           for a in ("signflip", "clean") for sc in ("seq", "conc", "par")]
# P3b (peer_exclude) is compared with P3 itself: the same check and rollback, without the exclusion
P3_RUNS += [(f"rb_{sc}_signflip_ora_p3x_budget", f"rb_{sc}_signflip_ora_p3_budget") for sc in ("seq", "conc", "par")]


def _scores(name, ds, s):
    """(last global version from the run, saved final model standard, saved final model per-class budget):
    mAP50 each, None when missing."""
    summ = load(VAR / f"{name}/{ds}/s{s}/summary.json") or {}
    run = next((v.get("map50") for k, v in summ.items() if k.startswith("federated_")), None)
    cb = load(VAR / f"{name}/{ds}/s{s}/class_budget.json") or {}
    return run, (cb.get("standard") or {}).get("map50"), (cb.get("class_budget") or {}).get("map50")


def _report_p3_p4(ds, fmt, mean):
    from cofy.config import load_config
    out = ["", "## P4: consecutive merges bounded jointly (concurrent scheme)", "",
           "mAP50 as Fed:Cent (%), mean over seeds: last global version (as the robustness table), saved final "
           "model with standard evaluation and with a per-class budget; without P4 = the same attack and defence "
           "without the candidate. Joint-bound cuts: attacker uploads whose BatchNorm or shared part the joint "
           "bound scaled further than the single-update bound.", "",
           "| run | n | last version | saved, standard | saved, per-class budget | without P4: last version | "
           "without P4: saved, standard | joint-bound cuts (attacker / honest) |", "|---|---|---|---|---|---|---|---|"]
    fc = lambda v, s: None if v is None or not centralized(ds, s) else 100 * v / centralized(ds, s)
    for name, base in P4_RUNS:
        seeds = [s for s in (0, 1, 2) if (VAR / f"{name}/{ds}/s{s}/summary.json").exists()]
        if not seeds:
            continue
        sc = [_scores(name, ds, s) for s in seeds]
        bs = [_scores(base, ds, s) for s in seeds]
        cut = {True: 0, False: 0}
        for s in seeds:
            for f in (VAR / f"{name}/{ds}/s{s}").glob("*/robust_log.jsonl"):
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    cut[r["attacker"]] += any((r.get(p) or {}).get("scale", 1) < (r.get(p) or {}).get("scale_single", 1) - 1e-9
                                              for p in ("bn", "shared"))
        col = lambda rows, i: fmt(mean([x for x in (fc(r[i], s) for r, s in zip(rows, seeds)) if x is not None]), "{:.0f}")
        out.append(f"| {name} | {len(seeds)} | {col(sc, 0)} | {col(sc, 1)} | {col(sc, 2)} | {col(bs, 0)} | {col(bs, 1)} | "
                   f"{cut[True]} / {cut[False]} |")
    out += ["", "## P3: peer validation with per-class rollback", "",
            "mAP50 as Fed:Cent (%), mean over seeds (columns as for P4). Rolled back: classes quarantined, with the "
            "version, per seed; attacker's = the attacker's own classes (wanted), other = honest classes (false alarms; "
            "the node named as attacker also lies in its reports in the clean twins). Rows *_p3x_*: P3b, the owner "
            "of a rolled-back class is also excluded; their 'without' columns are P3 without the exclusion.", "",
            "| run | n | last version | saved, standard | saved, per-class budget | without P3: last version | "
            "without P3: saved, per-class budget | rolled back: attacker's / other (versions) |",
            "|---|---|---|---|---|---|---|---|"]
    for name, base in P3_RUNS:
        seeds = [s for s in (0, 1, 2) if (VAR / f"{name}/{ds}/s{s}/summary.json").exists()]
        if not seeds:
            continue
        sc = [_scores(name, ds, s) for s in seeds]
        bs = [_scores(base, ds, s) for s in seeds]
        rb = []
        for s in seeds:
            cfg = load_config(ROOT / f"configs/variants/{name}/{ds}_s{s}.yaml")
            own = set(next(n.owned_global_ids(cfg.global_classes) for n in cfg.nodes if n.name == attacker_of(cfg, s)))
            q = {}
            for f in (VAR / f"{name}/{ds}/s{s}").glob("*/peer_log.jsonl"):
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    for c in r["rolled_back"]:
                        q[c] = r["version"]
            att = [f"{cfg.global_classes[c]}@{v}" for c, v in q.items() if c in own]
            oth = [f"{cfg.global_classes[c]}@{v}" for c, v in q.items() if c not in own]
            rb.append(f"s{s}: {','.join(att) or '-'} / {','.join(oth) or '-'}")
        col = lambda rows, i: fmt(mean([x for x in (fc(r[i], s) for r, s in zip(rows, seeds)) if x is not None]), "{:.0f}")
        out.append(f"| {name} | {len(seeds)} | {col(sc, 0)} | {col(sc, 1)} | {col(sc, 2)} | {col(bs, 0)} | {col(bs, 2)} | "
                   f"{'; '.join(rb)} |")
    return out


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        print(__doc__)
