# Multi-machine experiments

The study simulated concurrency with a timing model on one GPU, and the networked implementation was
exercised with four clients on one machine (article, *Limitations*: "a multi-site deployment remains to be
measured"). These experiments run the same federations on **three machines on one Wi-Fi network**, with
real network transfers and real arrival times, and compare them with the simulated runs of the same seeds.

Setting up the client machines, step by step: [CLIENT_SETUP.md](CLIENT_SETUP.md). Installation, data and the server/client basics: [README.md](README.md). Full network guide:
[`cofy_system/docs/NETWORK_FEDERATION.md`](cofy_system/docs/NETWORK_FEDERATION.md).

- [1. What is measured](#1-what-is-measured)
- [2. Machines and roles](#2-machines-and-roles)
- [3. Data on each machine](#3-data-on-each-machine)
- [4. The experiments](#4-the-experiments)
- [5. Running one experiment, machine by machine](#5-running-one-experiment-machine-by-machine)
- [6. After each experiment](#6-after-each-experiment)
- [7. Wi-Fi and troubleshooting](#7-wi-fi-and-troubleshooting)

## 1. What is measured

For each use case and scheme, on seed 100 first (101 and 102 if time allows):

| Quantity | How | Compared with |
|---|---|---|
| Accuracy: mAP50 and Fed:Cent of the final and of the calibrated model | `tools/compare_net_run.py` on M1, after the run | the simulated run of the same seed and configuration, at its registered calibration stage, and the calibrated centralized reference of that seed |
| Wall-clock time: start of the federation to the calibrated model | the server log on M1 (timestamps), or the times noted at **Start federation** and at the end | the timing model of the article (CWT about 2.4–3.3 times FedAsync; FedAvg about as FedAsync) |
| Arrival order and staleness (FedAsync) | the server log: one line per merge with its staleness | the simulated timeline (`virtual_schedule.json` of the simulated run) |
| Communication | about 100 MB per download and per upload, 0.28 MB per node for the calibration | the article's communication figures |
| Calibration over the network | the room shows `final` and `calibrated` checkpoints | the post-hoc calibration of the study |

A networked run is not bit-identical to the simulated one even for CWT, because GPU kernels are not
deterministic across machines; the question is whether it falls within the seed-to-seed spread of the
confirmatory study (Table 5 of the article). For FedAsync, arrival order and staleness also come from the
real machines instead of the timing model.

## 2. Machines and roles

Four nodes (A, B, C, D) per use case, as in the study; three machines:

| Machine | Runs | Ports |
|---|---|---|
| **M1** — the machine that prepared the study (it has `data/study/`, the simulated results and the evaluation sets) | the **server** and the client of **node A** | 8900 (server, open to the network), 8901 (client page, local) |
| **M2** — the machine with the most GPU memory | the clients of **node B** and **node C** | 8901, 8902 (local) |
| **M3** | the client of **node D** | 8901 (local) |

GPU memory: the study model needs about 6.6 GB per node that is training.
- **CWT** trains one node at a time, so any GPU with at least 7 GB works, also with two nodes on M2.
- **FedAsync and FedAvg** train all nodes at the same time, so M2 trains two nodes at once: it needs about
  14 GB (or two GPUs: start one client with `CUDA_VISIBLE_DEVICES=0`, the other with `CUDA_VISIBLE_DEVICES=1`).
  If M2 has less, run these schemes at a smaller scale (`python tools/make_net_configs.py --scale s`) and
  compare them with a simulated run of the same scale-s config on M1 (the tool prints the command), not
  with the study. Check the memory of one training with `nvidia-smi` during the smoke test (E0).

The server needs no GPU and runs **without** `--developer-mode`, as in deployment: it holds no data, and
the models are scored afterwards on M1 (section 6). This also keeps M1's GPU for node A.

Every machine: same commit (`git rev-parse HEAD`), same environment (README section 1), power saving and
sleep disabled for the duration of a run (macOS: `caffeinate -dims` in a terminal), `nvidia-smi` working on
NVIDIA machines. A Mac (Apple silicon) can host nodes: training runs on MPS, whose memory is shared with the system,
and is slower than a recent NVIDIA GPU; prefer it for CWT, where one node trains at a time.

## 3. Data on each machine

Each node needs its **whole** folder of `data/study/<use case>/<node>/` (the training fraction is drawn
from it at training time). M1 already has everything; M2 and M3 receive their nodes' folders, exported
by M1 with real files instead of links:

```bash
# on M1, in cofy_system/ (replace <user>, <M2-ip>, <M3-ip>; --dest is the folder on the target machine)
for uc in wbc dentex kfm12; do
  python tools/export_node_data.py --use-case $uc --nodes node_B node_C --out ~/cofy_export/m2 --dest /home/<user>/cofy_data
  python tools/export_node_data.py --use-case $uc --nodes node_D        --out ~/cofy_export/m3 --dest /home/<user>/cofy_data
done
rsync -a --progress ~/cofy_export/m2/ <user>@<M2-ip>:/home/<user>/cofy_data/
rsync -a --progress ~/cofy_export/m3/ <user>@<M3-ip>:/home/<user>/cofy_data/
```

| Use case | M1 (node A, in place) | M2 (nodes B, C) | M3 (node D) |
|---|---|---|---|
| WBC | `data/study/wbc/node_A/data.yaml` | `~/cofy_data/wbc/node_B`, `node_C` (0.1 GB each) | `~/cofy_data/wbc/node_D` (0.1 GB) |
| DENTEX | `data/study/dentex/node_A/data.yaml` | `~/cofy_data/dentex/node_B`, `node_C` (0.5 GB each) | `~/cofy_data/dentex/node_D` (0.5 GB) |
| KFM12 | `data/study/kfm12/node_A/data.yaml` | `~/cofy_data/kfm12/node_B` (10.7 GB), `node_C` (5.8 GB) | `~/cofy_data/kfm12/node_D` (3.0 GB) |

KFM12 is large: copy it once, ahead of time, by cable or USB drive if the Wi-Fi is slow (the export folder
can be copied as it is). The client checks at join that each `data.yaml` lists exactly the node's classes.

Room configs: generated once on M1 with `python tools/make_net_configs.py` (already in `configs/net/`):
each is the frozen configuration of a simulated run of the study with `mode` set, the calibration run over
the network (`final_calibration: true`) and its own output folder `runs/net/<experiment>/`.

## 4. The experiments

Run them one at a time, in this order (durations are rough: one simulated run of the study took about
26 min on WBC, 40 min on DENTEX and 45 min on KFM12 on one GPU; transfers over Wi-Fi come on top).

| Id | Use case | Scheme | Room config (`configs/net/`) | Room id | Approx. time |
|---|---|---|---|---|---|
| E0 | WBC | CWT, smoke test (1 cycle, 2% of the images) | `smoke_wbc_cwt.yaml` | `smoke` | 5–10 min |
| E1 | WBC | CWT | `wbc_cwt_s100.yaml` | `wbc-cwt-s100` | 35–45 min |
| E2 | WBC | FedAsync | `wbc_fedasync_s100.yaml` | `wbc-fedasync-s100` | 15–25 min |
| E3 | WBC | FedAvg (wall-clock reference; development seed 0) | `wbc_fedavg_s0.yaml` | `wbc-fedavg-s0` | 15–25 min |
| E4 | DENTEX | CWT | `dentex_cwt_s100.yaml` | `dentex-cwt-s100` | 45–60 min |
| E5 | DENTEX | FedAsync | `dentex_fedasync_s100.yaml` | `dentex-fedasync-s100` | 20–30 min |
| E6 | KFM12 | CWT | `kfm12_cwt_s100.yaml` | `kfm12-cwt-s100` | 55–70 min |
| E7 | KFM12 | FedAsync | `kfm12_fedasync_s100.yaml` | `kfm12-fedasync-s100` | 25–35 min |
| E8 | all | E1, E2, E4–E7 on seeds 101 and 102 | `<use case>_<scheme>_s101.yaml`, `..._s102.yaml` | `<use case>-<scheme>-s101`, ... | as above |

Each experiment uses the same four nodes on the same machines (node A on M1, B and C on M2, D on M3).
Results per experiment: section 6.

## 5. Running one experiment, machine by machine

Example: **E1**, WBC with CWT. For another experiment change the config, the room id and the use case
of the data folders. Everything is run from `cofy_system/` with `conda activate cofy`.

**M1 — server** (once; it keeps running across experiments):
```bash
hostname -I                                      # M1's address, e.g. 192.168.1.10 (macOS: ipconfig getifaddr en0)
sudo ufw allow 8900/tcp                          # Linux with ufw active; macOS: allow Python's incoming connections
python tools/launch_detached.py runs/net_server.log -- python -m cofy.net.server_app --port 8900 --config-dir configs/net
tail -f runs/net_server.log                      # admin token near the top
```
From M2 and M3, check the connection: `curl http://192.168.1.10:8900/api/info` must answer
`{"developer_mode":false,...}`.

**M1 — create the room:** open `http://127.0.0.1:8900`, choose `wbc_cwt_s100.yaml`, room id
`wbc-cwt-s100`, **Create room**. Note the room password shown on the card.

**M1 — node A:**
```bash
python tools/launch_detached.py runs/net/client_node_A.log -- python -m cofy.net.client_app --port 8901
```
Open `http://127.0.0.1:8901` and fill in: server URL `http://127.0.0.1:8900`, room id `wbc-cwt-s100`, the
password, **Reclaim node** `node_A`, data `.../cofy_system/data/study/wbc/node_A/data.yaml`; **Join & prepare**.

**M2 — nodes B and C** (two clients, two pages):
```bash
python tools/launch_detached.py runs/net/client_node_B.log -- python -m cofy.net.client_app --port 8901
python tools/launch_detached.py runs/net/client_node_C.log -- python -m cofy.net.client_app --port 8902
```
`http://127.0.0.1:8901`: server URL `http://192.168.1.10:8900`, room `wbc-cwt-s100`, password,
**Reclaim node** `node_B`, data `~/cofy_data/wbc/node_B/data.yaml`; **Join & prepare**.
`http://127.0.0.1:8902`: the same with `node_C` and `~/cofy_data/wbc/node_C/data.yaml`.

**M3 — node D:**
```bash
python tools/launch_detached.py runs/net/client_node_D.log -- python -m cofy.net.client_app --port 8901
```
`http://127.0.0.1:8901`: server URL `http://192.168.1.10:8900`, room `wbc-cwt-s100`, password,
**Reclaim node** `node_D`, data `~/cofy_data/wbc/node_D/data.yaml`; **Join & prepare**.

**Start:** when the room card on M1 shows the four nodes `IDLE`, note the time and press
**Start federation** on M1, then **Start training** on each of the four client pages. Leave everything
running: the nodes train (CWT: in turns, the others wait for theirs), upload, and run the calibration at
the end. The room is finished when its card shows the `calibrated` checkpoint; note the time.

The clients can stay open for the next experiment: on each page, join the new room (new room id and
password, same **Reclaim node** and the data of the next use case).

## 6. After each experiment

On M1, in `cofy_system/`:
```bash
python tools/compare_net_run.py --experiment wbc_cwt_s100 --room wbc-cwt-s100
curl -s http://127.0.0.1:8900/api/rooms/wbc-cwt-s100 > runs/net/wbc_cwt_s100/room.json
cp runs/net_server.log runs/net/wbc_cwt_s100/server.log
```
`compare_net_run.py` scores the room's `final` and `calibrated` models on the evaluation set and writes
`runs/net/<experiment>/comparison.json` next to the room's checkpoints (`runs/net/<experiment>/net_room_<room>/`),
with the simulated run and the centralized reference of the same seed. Copy the client logs of M2 and M3
(`runs/net/client_node_*.log`) into the same folder. For FedAsync the staleness of every merge is in the
server log (`staleness=`).

## 7. Wi-Fi and troubleshooting

- Give the three machines fixed addresses (DHCP reservation on the router) so the server URL does not change.
- Use the 5 GHz band if available; check the throughput once (`iperf3 -s` on M1, `iperf3 -c <M1-ip>` on M2):
  each update moves about 100 MB each way.
- A node silent for 180 s is marked `LOST` (the room continues without it; CWT skips its turns). A client
  that crashed rejoins with **Reclaim node** after about 3 minutes.
- More symptoms and fixes: section 11 of the network guide.
