# COFY — Class-Owned Federated YOLO

Federated object detection for sites that annotate different classes. Every class has
exactly one owner node, only the owner changes that class's output parameters, and the server holds no
data. Three aggregation schemes: **FedAvg** (parallel synchronous rounds), **CWT** (cyclic weight transfer:
sequential turns scheduled by the server) and **FedAsync** (concurrent, merged on arrival), a
barrier-free calibration, a secured client-server implementation and ownership-aware robust aggregation.

Article: *Federated Object Detection Under Extreme Non-IID Labels in Biomedical Imaging: System Proposal
and Analysis of Reliability and Robustness* (source in [`cofy_system/paper/`](cofy_system/paper/)).

**Try it on one machine:** `cd cofy_system && python demo/run_demo.py --use-case wbc --scheme sequential`
runs a real four-node federation over HTTP with small samples of the three use cases.
Online: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SalvatoreCapuozzo/DAFYOLO/blob/master/cofy_system/demo/COFY_demo.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/SalvatoreCapuozzo/DAFYOLO)

- [Repository layout](#repository-layout)
- [1. Install (every machine)](#1-install-every-machine)
- [2. Data](#2-data)
- [3. Federation over a local network: server and clients](#3-federation-over-a-local-network-server-and-clients)
- [4. Reproducing the study on one machine](#4-reproducing-the-study-on-one-machine)
- [Licence](#licence)

## Repository layout

```
cofy_system/                 the system and the study (run every command from here)
  cofy/                      the package: model, masked local training, schemes, calibration, defence
  cofy/net/                  networked server (server_app) and node client (client_app), web dashboards
  configs/                   study/ (registered), variants/ (all runs of the study), net/ (network rooms), arch/
  tools/                     data preparation, the study's runners and analyses, figures, network helpers
  demo/                      one-machine demonstration with small samples of the three use cases
  docs/                      network guide (NETWORK_FEDERATION.md) and design notes
  analysis/                  pre-registration, data and code manifests, result reports
  paper/                     the article
  _frozen/confirm/           frozen code of the confirmatory study (never edit; hashed)
  data/, runs/               local data and outputs (not in git)
MULTI_MACHINE_EXPERIMENTS.md the experiments to run on several machines, step by step
legacy/                      the earlier DAFYOLO prototype, reports and its data (not used by COFY)
```

## 1. Install (every machine)

Linux with an NVIDIA GPU (CUDA) and macOS on Apple silicon (MPS) are supported; the device is chosen
automatically (`federation.device: auto`: CUDA, then MPS, then CPU). The study itself ran on CUDA.

```bash
git clone <repository URL> cofy && cd cofy/cofy_system
conda create -n dafyolo python=3.10 -y && conda activate dafyolo
conda env config vars set PYTHONNOUSERSITE=1 -n dafyolo && conda activate dafyolo   # ignore ~/.local packages
# PyTorch: on Linux the CUDA build for your driver (https://pytorch.org/get-started), e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# on macOS the default build, which includes MPS:  pip install torch torchvision
pip install ultralytics==8.4.84          # exactly this version: COFY hooks into its internals
pip install -r requirements.txt          # re-pin ultralytics==8.4.84 if this upgraded it
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available(), ultralytics.__version__)"
```

The study model (YOLO26-L, 640 px, batch 4) needs about **6.6 GB of GPU memory per training node**. The
COCO weights `yolo26l.pt` are downloaded on first use, or copied into `cofy_system/`.

Every machine of a federation must run the same commit (`git rev-parse HEAD`) and the same versions.

## 2. Data

Data is not in git. Each use case is split once into four nodes and a fully labelled evaluation set, under
`cofy_system/data/study/<use case>/{node_A,node_B,node_C,node_D,_eval}/`, each a YOLO dataset with its
`data.yaml`. Classes, owners and counts: Table 2 of the article.

| Use case | Images and labels | Licence | Node folders (real files) |
|---|---|---|---|
| KFM12 | the authors' Kubic FLOTAC Microscope dataset of parasite eggs (12 classes; release in preparation; a two-class subset is public, doi:10.17632/6bjcxw8dx2.1) | the authors' | 3.0–10.7 GB each, eval 4.5 GB |
| WBC | PBC peripheral blood cell images (doi:10.1016/j.dib.2020.105474) with the bounding boxes of github.com/medmabcf/White-Blood-Cell-Detection-Dataset | CC BY 4.0 | 0.1 GB each |
| DENTEX | tooth-enumeration subset of DENTEX (huggingface.co/datasets/ibrahimhamamci/DENTEX), converted by `tools/convert_dentex.py` | CC BY-NC-SA 4.0 | 0.5 GB each (every node holds all 508 images) |

**On the machine that prepared the study** (`data/study/` exists), check it is unchanged:

```bash
PYTHONPATH=. python tools/study_manifest.py verify analysis/study_manifest.json   # ALL DATA VERIFIED
```

**On another machine**, either rebuild KFM12 and WBC from their label pools (`data/_pool/`, a few MB,
copied from the preparing machine) and the raw images, with the same verification at the end:

```bash
python tools/rebuild_study_data.py --kfm12-images /path/to/kfm12/images --wbc-images /path/to/wbc/images
```

or, simpler for a federation, receive only the node folders it hosts, exported with real files by the
preparing machine (`tools/export_node_data.py`; details in
[MULTI_MACHINE_EXPERIMENTS.md](MULTI_MACHINE_EXPERIMENTS.md#3-data-on-each-machine)). A node needs its
**whole** folder: the class-stratified training fraction is drawn on the node at training time, from the
seed, exactly as in the study.

## 3. Federation over a local network: server and clients

One machine runs the **server** (it needs no GPU and no training data); every node runs a **client** on
the machine that holds its data. A machine can host the server and one or more clients at once.
Full guide, with screenshots, security and troubleshooting:
[`cofy_system/docs/NETWORK_FEDERATION.md`](cofy_system/docs/NETWORK_FEDERATION.md).

**Server** (e.g. machine M1; its address from `hostname -I` on Linux or `ipconfig getifaddr en0` on macOS, say `192.168.1.10`):

```bash
cd cofy_system && conda activate dafyolo
sudo ufw allow 8900/tcp      # Linux, only if ufw is active; macOS: allow incoming connections for Python when asked
python tools/launch_detached.py runs/net_server.log -- \
    python -m cofy.net.server_app --port 8900 --config-dir configs/net
tail -f runs/net_server.log                               # prints the admin token
```

Open `http://127.0.0.1:8900` on M1, choose a config of `configs/net/` (made by `tools/make_net_configs.py`),
give the room an id and **Create room**; send the server URL, the room id and the room password to the
node operators. Add `--developer-mode` to let the server evaluate the model (it then reads the
evaluation set on M1; for experiments only).

**Client**, one per node, on the machine that holds the node's folder:

```bash
cd cofy_system && conda activate dafyolo
python tools/launch_detached.py runs/client_node_B.log -- python -m cofy.net.client_app --port 8901
```

Open `http://127.0.0.1:8901` **on that machine** (from elsewhere: `ssh -L 8901:127.0.0.1:8901 user@machine`),
fill in server URL `http://192.168.1.10:8900`, room id and password, **Reclaim node** = the node it plays
(e.g. `node_B`; on a first join this simply picks that node), and the node's `data.yaml`; then
**Join & prepare**. A second node on the same machine uses another port (`--port 8902`). When every node
has joined, the server operator presses **Start federation** and each node **Start training**; nodes then
train, upload and wait on their own, and run the calibration at the end.

All machines must be on the same network (Wi-Fi works: one model download and one upload of about
100 MB per node and update) and must not go to sleep during a run. HTTPS: section 9 of the guide.

## 4. Reproducing the study on one machine

The study runs from a queue of plan entries (`cofy_system/runs/variants/_plan.txt`, local), one GPU job at
a time; the commands to stop, restart and watch it are in
[`cofy_system/stop_restart_queue.txt`](cofy_system/stop_restart_queue.txt) and
[`cofy_system/QUEUE_COMMANDS.md`](cofy_system/QUEUE_COMMANDS.md). A single simulated run:

```bash
python -m cofy.simulate --config configs/variants/confirm_v2sync/wbc_s100.yaml --session s100 --mode async
```

Pre-registration, amendments and manifests: `cofy_system/analysis/CONFIRMATORY_PREREG.md`; check that the
frozen confirmatory code is unchanged with `python tools/confirm_manifest.py --check`. More on the
package, configuration keys and architectures: [`cofy_system/README.md`](cofy_system/README.md).

## Licence

Code under the GNU AGPL-3.0 ([`LICENSE`](LICENSE)), the licence of the Ultralytics library it builds on.
Datasets keep their own licences (table above); the demo images are listed in
[`cofy_system/demo/data/README.md`](cofy_system/demo/data/README.md).
