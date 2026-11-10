# COFY demo: a class-owned federation on one machine

`run_demo.py` runs a real COFY federation (Class-Owned Federated YOLO) over HTTP on your machine:
the server process, and one client process per node, each with its own small dataset. Nothing is
simulated: the clients download the global model, train on their own images, and upload their
updates; the server merges them with the scheme you choose, then runs the calibration phase.

```bash
cd cofy_system
python demo/run_demo.py --use-case wbc --scheme sequential            # 1-3 min on a laptop CPU
python demo/run_demo.py --use-case kfm12 --scheme concurrent --keep-open
python demo/run_demo.py --use-case dentex --scheme sequential --scale l --imgsz 640   # the study's model size
```

While it runs, open the **server dashboard** (`http://127.0.0.1:8900`) and the four **client pages**
(`http://127.0.0.1:8901` to `8904`). The script prints the same events in the terminal.

## Run it online, free

- **Google Colab** (free GPU, Google account): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SalvatoreCapuozzo/DAFYOLO/blob/master/cofy_system/demo/COFY_demo.ipynb)
  The notebook installs the code, starts the federation and gives links to the server dashboard and the four
  client pages.
- **GitHub Codespaces** (free monthly hours, GitHub account, CPU): [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/SalvatoreCapuozzo/DAFYOLO)
  Wait for the setup to finish (a few minutes), then run
  `cd cofy_system && python demo/run_demo.py --use-case wbc --keep-open`;
  the dashboards open from the *Ports* tab (8900 is the server, 8901–8904 the nodes).
- **Your own machine**: see below.

Both online options need nothing installed locally. A free service cannot keep a permanent public demo
running, because every visitor starts a real training run; each person runs their own copy instead.

## Run it on your machine

```bash
git clone https://github.com/SalvatoreCapuozzo/DAFYOLO.git && cd DAFYOLO/cofy_system
pip install -r requirements.txt "ultralytics==8.4.84"
python demo/run_demo.py --use-case wbc --scheme sequential
```

## What to look for

- **Ownership.** Each node owns a disjoint set of classes (printed at join, and in
  `data/<use case>/classes.yaml`). Its images are labelled only for those classes, even when they show
  objects of other nodes' classes. Only a class's owner changes that class's output rows: the server
  takes them from its own registry, never from what an upload claims.
- **Sequential scheme.** The server schedules the turns, in a new random order every cycle. Only the
  node whose turn it is may download and upload; the others are told to wait (shown as `waiting`).
- **Concurrent scheme.** Every node trains as soon as it can, and each upload is merged on arrival,
  weighted by how stale it is.
- **Calibration.** After training, each node sends the BatchNorm statistics of the final model on its
  own images, then re-fits only its own class rows. In the concurrent scheme the server releases an
  updated model after every contribution, without waiting for everyone.
- **No data on the server.** The demo starts the server in developer mode only so that it can score
  the model on a small evaluation set for you to see. In a real deployment the server holds no data
  and never evaluates.

With six images per node and one or two cycles, the accuracy printed at the end only shows that the
pipeline works; the study's results come from the full datasets (see the paper).

## Options

| Option | Default | Meaning |
|---|---|---|
| `--use-case` | `wbc` | `kfm12` (parasite eggs), `wbc` (white blood cells) or `dentex` (dental radiographs) |
| `--scheme` | `sequential` | `sequential` (server-scheduled turns) or `concurrent` (merged on arrival) |
| `--cycles`, `--local-epochs` | 3, 10 | training cycles per node, and epochs per cycle |
| `--lr0` | 0.01 | learning rate; higher than the study's 0.0008, because a demo trains few images for few steps |
| `--scale`, `--imgsz` | `n`, 320 | model size and input size; the study used `l` at 640 |
| `--cpu` | off | train on the CPU even if a GPU is available |
| `--port` | 8900 | server port; the clients take the next four |
| `--keep-open` | off | leave the dashboards running at the end (Ctrl-C to stop) |

Output goes to `runs/demo/<use case>_<scheme>/`: the room config, one log per process, the server's
persisted state and its audit trail, and the final and calibrated models.

## Requirements

The project environment (`requirements.txt`: PyTorch, `ultralytics==8.4.84`, `fastapi`, `uvicorn`,
`requests`). The first run downloads the COCO-pretrained YOLO26 weights from Ultralytics.

## Data

`data/` holds a few images per node from each use case, re-encoded at a smaller size, with their
labels. Their sources and licences are in [`data/README.md`](data/README.md).
`make_demo_data.py` rebuilds them from the study's splits (maintainers only).
