# Setting up the client machines

How to turn other machines into COFY clients that connect to one machine acting as both server and
client. Roles, as in [MULTI_MACHINE_EXPERIMENTS.md](MULTI_MACHINE_EXPERIMENTS.md):

| Machine | Runs |
|---|---|
| **M1** (this machine, the one that has `data/study/`) | the server and the client of node A |
| **M2** | the clients of nodes B and C |
| **M3** | the client of node D |

All three machines must be on the same local network. Run every command from `cofy_system/` unless stated
otherwise.

- [0. Before you start](#0-before-you-start)
- [1. M1: prepare the server side](#1-m1-prepare-the-server-side)
- [2. M2 and M3: install](#2-m2-and-m3-install)
- [3. M1: send each client its data](#3-m1-send-each-client-its-data)
- [4. Run a federation](#4-run-a-federation)
- [5. Checklist when something does not connect](#5-checklist-when-something-does-not-connect)

## 0. Before you start

- **Push the code from M1.** The clients clone the repository, so it must contain the current code:
  ```bash
  cd ~/COFY
  git remote set-url origin git@github.com:SalvatoreCapuozzo/COFY.git   # once: SSH works, the HTTPS token does not
  git commit ... && git push
  git rev-parse HEAD                                                     # note it: every machine must run this commit
  ```
- **M1's GPU must be free.** Node A trains on M1's GPU, so no queue or simulation may be running there
  (check with `python tools/queue_status.py` and `nvidia-smi`). The server itself needs no GPU.
- **GPU memory on the clients.** One node in training needs about 6.6 GB. CWT trains one node at a time;
  FedAsync and FedAvg train all nodes at once, so M2 (two nodes) then needs about 14 GB, or two GPUs
  (start one client with `CUDA_VISIBLE_DEVICES=0`, the other with `CUDA_VISIBLE_DEVICES=1`).

## 1. M1: prepare the server side

```bash
hostname -I                     # M1's address on the network, e.g. 192.168.1.10 (macOS: ipconfig getifaddr en0)
sudo ufw allow 8900/tcp         # open the server port (Linux with ufw active)
```
Reserve this address for M1 in the router (DHCP reservation), so that the server URL does not change
between sessions. In the rest of this file, `<M1-ip>` is that address.

M1 needs no new installation: it already has the `cofy` environment, the data and the pretrained weights.

## 2. M2 and M3: install

On each client machine:

1. **Access to the repository.** If the repository is private, add this machine's SSH key to the GitHub
   account (`ssh-keygen -t ed25519`, then paste `~/.ssh/id_ed25519.pub` in GitHub, *Settings, SSH keys*).
2. **Clone, at the same commit as M1:**
   ```bash
   git clone git@github.com:SalvatoreCapuozzo/COFY.git ~/COFY
   cd ~/COFY && git checkout <commit noted on M1>
   cd cofy_system
   ```
3. **Environment** (same versions as M1):
   ```bash
   conda create -n cofy python=3.10 -y && conda activate cofy
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # macOS: pip install torch torchvision
   pip install ultralytics==8.4.84            # exactly this version
   pip install -r requirements.txt            # re-pin ultralytics==8.4.84 if this upgraded it
   conda env config vars set PYTHONNOUSERSITE=1 -n cofy && conda activate cofy   # ignore packages in ~/.local
   python -c "import torch, ultralytics; print(torch.cuda.is_available(), ultralytics.__version__)"
   ```
   The last line must print `True 8.4.84` on an NVIDIA machine (on a Mac, `False`: training runs on MPS).
   No weight files are needed on the clients: the server sends the model.
4. **Let M1 copy files in** (for step 3): on Linux `sudo apt install openssh-server`; on macOS enable
   *System Settings, General, Sharing, Remote Login*. Or skip this and copy the data by USB drive.
5. **Check that the server is reachable** (once the server runs, section 4):
   `curl http://<M1-ip>:8900/api/info` must answer `{"developer_mode":false,...}`.

## 3. M1: send each client its data

Each node needs its whole folder `data/study/<use case>/<node>/`, exported with real files instead of
links and with paths rewritten for the target machine (`--dest`):

```bash
# on M1, in cofy_system/ (replace <user>, <M2-ip>, <M3-ip>)
for uc in wbc dentex kfm12; do
  python tools/export_node_data.py --use-case $uc --nodes node_B node_C --out ~/cofy_export/m2 --dest /home/<user>/cofy_data
  python tools/export_node_data.py --use-case $uc --nodes node_D        --out ~/cofy_export/m3 --dest /home/<user>/cofy_data
done
rsync -a --progress ~/cofy_export/m2/ <user>@<M2-ip>:/home/<user>/cofy_data/
rsync -a --progress ~/cofy_export/m3/ <user>@<M3-ip>:/home/<user>/cofy_data/
```

`--dest` must be the folder the data will have on the client (on macOS: `/Users/<user>/cofy_data`).
KFM12 is large (about 16 GB for M2, 3 GB for M3): copy it once, ahead of time, by cable or USB drive if
the Wi-Fi is slow (the export folder can be copied as it is).

## 4. Run a federation

Example: WBC with CWT, room `wbc-cwt-s100`. For another experiment change the config, the room id and the
use case in the data paths.

**M1, server** (once; it keeps running across experiments):
```bash
cd ~/COFY/cofy_system && conda activate cofy
python tools/launch_detached.py runs/net_server.log -- python -m cofy.net.server_app --port 8900 --config-dir configs/net
tail -f runs/net_server.log                      # admin token near the top
```
Open `http://127.0.0.1:8900`, choose `wbc_cwt_s100.yaml`, room id `wbc-cwt-s100`, **Create room**.
Note the room password shown on the card.

**M1, node A:**
```bash
python tools/launch_detached.py runs/net/client_node_A.log -- python -m cofy.net.client_app --port 8901
```
Open `http://127.0.0.1:8901` and fill in: server URL `http://127.0.0.1:8900`, room id `wbc-cwt-s100`,
the password, **Reclaim node** `node_A`, data `~/COFY/cofy_system/data/study/wbc/node_A/data.yaml`;
**Join & prepare**.

**M2, nodes B and C** (two clients, two pages):
```bash
cd ~/COFY/cofy_system && conda activate cofy
python tools/launch_detached.py runs/net/client_node_B.log -- python -m cofy.net.client_app --port 8901
python tools/launch_detached.py runs/net/client_node_C.log -- python -m cofy.net.client_app --port 8902
```
- `http://127.0.0.1:8901`: server URL `http://<M1-ip>:8900`, room `wbc-cwt-s100`, password,
  **Reclaim node** `node_B`, data `~/cofy_data/wbc/node_B/data.yaml`; **Join & prepare**.
- `http://127.0.0.1:8902`: the same with `node_C` and `~/cofy_data/wbc/node_C/data.yaml`.

**M3, node D:**
```bash
cd ~/COFY/cofy_system && conda activate cofy
python tools/launch_detached.py runs/net/client_node_D.log -- python -m cofy.net.client_app --port 8901
```
`http://127.0.0.1:8901`: server URL `http://<M1-ip>:8900`, room `wbc-cwt-s100`, password,
**Reclaim node** `node_D`, data `~/cofy_data/wbc/node_D/data.yaml`; **Join & prepare**.

The client checks at join that the `data.yaml` lists exactly the node's classes.

**Start:** when the room card on M1 shows the four nodes `IDLE`, press **Start federation** on M1, then
**Start training** on each of the four client pages. The nodes train (with CWT in turns), upload and run
the calibration at the end. The room is finished when its card shows the `calibrated` checkpoint.

During a run, keep every machine awake: disable suspend in the power settings (macOS: run
`caffeinate -dims` in a terminal). To follow a client from its terminal:
`tail -f runs/net/client_node_B.log`; GPU use: `watch -n 5 nvidia-smi`.

The clients can stay open for the next experiment: on each page, join the new room (new room id and
password, same **Reclaim node**, data of the next use case). Scoring the finished room on M1 and the list
of experiments: [MULTI_MACHINE_EXPERIMENTS.md](MULTI_MACHINE_EXPERIMENTS.md), sections 4 and 6.

## 5. Checklist when something does not connect

- `curl http://<M1-ip>:8900/api/info` fails from a client: the server is not running
  (`tail runs/net_server.log` on M1), the port is closed (`sudo ufw status` on M1), or the machines are on
  different networks (guest Wi-Fi often isolates devices).
- The join is refused: wrong room id or password, or the node is already taken (use **Reclaim node**
  with the node's name).
- The join fails on the data: the `data.yaml` path is wrong, or the export was made with a `--dest` that
  differs from the folder the data is in on the client.
- A node shows `LOST`: it was silent for 180 s (crash, sleep, network). Restart its client and join again
  with **Reclaim node** after about 3 minutes; the room continues meanwhile (CWT skips its turns).
- Different results on different machines: check that `git rev-parse HEAD` and
  `python -c "import ultralytics; print(ultralytics.__version__)"` match M1.

More symptoms and fixes: section 11 of [`cofy_system/docs/NETWORK_FEDERATION.md`](cofy_system/docs/NETWORK_FEDERATION.md).
