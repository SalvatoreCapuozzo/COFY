#!/bin/sh
# Waits for the ChestX-Det10 download, then prepares it for the study.
# Configs are written ONLY if every split invariant passes, so the study
# runner (which starts ChestX as soon as its configs exist) can never train
# on a broken split.
set -e
cd "$(dirname "$0")/.."
PY=/Users/salvatorecapuozzo/anaconda3/envs/dl_env/bin/python3
RAW=data/_raw/chestx_det10
until grep -q DOWNLOAD_OK $RAW/download.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M) download complete"
mkdir -p $RAW/extracted
unzip -q -o $RAW/train_data.zip -d $RAW/extracted
unzip -q -o $RAW/test_data.zip  -d $RAW/extracted
echo "$(date +%H:%M) extracted $(find $RAW/extracted -name '*.png' | wc -l | tr -d ' ') png files"
$PY tools/convert_chestx_det10.py --raw $RAW --out data/_pool/chestx
$PY tools/split_dataset_into_nodes.py --dataset data/_pool/chestx --out data/study/chestx \
    --oracle-out data/study/chestx_oracle --n-nodes 4 --drop-classes "" --seed 0 --val-frac 0.2 \
    --config-out /tmp/study_chestx.yaml
PYTHONPATH=. $PY - <<'PYEOF'
import sys, yaml, tempfile
from pathlib import Path
from cofy.sampling import cached_balanced_image_subset
names=lambda d:{p.name for p in d.iterdir()} if d.exists() else set()
root=Path("data/study/chestx"); orc=Path("data/study/chestx_oracle/node_A"); pool=Path("data/_pool/chestx")
split={}
bad=0
for n in "ABCD":
    for sp in ("train","val"):
        for f in names(root/f"node_{n}"/"images"/sp):
            if split.setdefault(f,sp)!=sp: bad+=1
ntrain=set().union(*(names(root/f"node_{n}"/"images"/"train") for n in "ABCD"))
ev=list((root/"_eval"/"images"/"val").iterdir()); evn={p.name for p in ev}
full=sum(len((pool/"labels"/(p.stem+".txt")).read_text().split("\n")) for p in ev)
got =sum(len((root/"_eval"/"labels"/"val"/(p.stem+".txt")).read_text().split("\n")) for p in ev)
checks={"train/val straddling nodes":bad,"eval seen by nodes":len(evn&ntrain),
        "eval seen by oracle":len(evn&names(orc/"images"/"train")),"eval duplicates":len(ev)-len(evn),
        "eval boxes missing":full-got,"node pool != oracle pool":len(ntrain^names(orc/"images"/"train"))}
for k,v in checks.items(): print(f"  {'PASS' if v==0 else 'FAIL'}  {k}: {v}")
tot=0
for n in "ABCD":
    d=root/f"node_{n}"
    with tempfile.TemporaryDirectory() as t:
        sub=cached_balanced_image_subset(d/"images"/"train",0.5,0,cache_file=Path(t)/"c.txt")
    own=list(yaml.safe_load((d/"data.yaml").read_text())["names"].values())
    print(f"  node_{n} owns {own} | train {len(names(d/'images'/'train'))} -> {len(sub)} at 0.5"); tot+=len(sub)
print(f"  training node-images at 0.5: {tot} | eval {len(evn)}")
sys.exit(1 if any(checks.values()) else 0)
PYEOF
$PY tools/make_study_configs.py chestx /tmp/study_chestx.yaml 0.5
rm -f $RAW/train_data.zip $RAW/test_data.zip
echo "$(date +%H:%M) CHESTX READY -- configs written, zips removed"
