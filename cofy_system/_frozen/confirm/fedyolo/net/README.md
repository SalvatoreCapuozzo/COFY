# fedyolo.net — real client-server federation

Everything else in this project (`fedyolo.simulate`, `FedServer`,
`AsyncFedServer`) runs a federation inside **one Python process**. This
subsystem runs the same config-driven federation as a real distributed
system: a server process with a dashboard, and separate client processes
(one per node, each with its own "personal room" page) that pull the global
model, train on their own machine's data, and push their update back over
HTTP or HTTPS. `federation.mode` decides sync vs. async, as everywhere else.
Clients use the same local training (`run_node_round`) and the server the
same merge code as the in-process runs.

Scope: a few trusted machines on one local network (or over a VPN). It is
not meant to be exposed to the internet.

## Quick start — one machine, simulating N nodes

```bash
python -m fedyolo.net.server_app --port 8900 [--config-dir configs]
python -m fedyolo.net.client_app --port 8901   # one per node
python -m fedyolo.net.client_app --port 8902
```

Open `http://127.0.0.1:8900` for the server dashboard. Create a room
**from a config file** (dropdown of `--config-dir`, with a preview) or
**from scratch** (dynamic, see below). The room password shows on the room
card.

Open `http://127.0.0.1:8901`, `8902`, ... for each client. Enter the server
URL, room ID and password, fetch the room, (dynamic rooms only) declare
your classes, and point the dataset field at a local `data.yaml` (browse to
one, or build one from a folder laid out as `images/{train,val}`,
`labels/{train,val}`). Join — the server assigns the next free node slot.
Press **Start federation** on the server once the nodes you want have
joined, then **Start training** on each client.

## Several machines on the same network

`server_app` binds `0.0.0.0`, so it is reachable on the LAN. On each client
machine, set "Server URL" to the server's LAN address
(`http://192.168.x.x:8900`). Each machine only needs its own node's data.
One node per machine removes the one-GPU limit of the simulation: nodes
really train at the same time, and async updates really arrive stale.

## Security

- **Admin actions** (dashboard, creating/deleting rooms, reading passwords,
  checkpoints, rollback, shutdown) are allowed from the server machine
  itself, or with the **admin token** printed at startup (also saved in
  `<state-dir>/admin_token`, mode 600). A dashboard opened from another
  machine asks for it once. Behind a reverse proxy on the same machine,
  every request looks local: start with `--require-admin-token`.
- **Nodes** need the room password **and** their own node token (issued at
  join) for every call; one node can't act as another.
- **Updates are plain tensors**: both sides load with
  `torch.load(weights_only=True)`, so a peer can't execute code by sending
  a crafted file. The server checks every update against the room's model
  (same keys, shapes, no NaN/Inf), caps its size (`--max-upload-mb`, 2048),
  and uses its own record of which classes each node owns.
- A node's momentum buffers never leave its machine (smaller uploads).
- **HTTPS**: `--ssl-certfile cert.pem --ssl-keyfile key.pem` on the server;
  clients use `https://...` URLs. For a private/self-signed certificate,
  start clients with `--ca-cert cert.pem`. Self-signed example:

  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=<server-ip>" \
      -addext "subjectAltName=IP:<server-ip>" -keyout key.pem -out cert.pem
  ```

## Robustness

- **Retries**: every client call is retried with back-off (2 s up to 60 s)
  on network errors, timeouts or a restarting server; the node keeps going.
- **Heartbeats**: clients report every 20 s while running. A node silent for
  `--heartbeat-timeout` seconds (180) is marked `lost`.
- **Sync rounds don't hang on a lost node**: once every missing node is
  silent past the heartbeat timeout, the round is aggregated with the nodes
  that reported; classes owned only by absent nodes keep their previous
  weights. `--sync-round-timeout S` (> 0) additionally caps how long a
  round waits even for a node that is still alive.
- **Async rooms** finish once every node is done or gone.
- **Idempotent uploads**: a retried push (same `push_id`) is answered from
  the record, never aggregated twice.
- **Reclaiming a node**: a client that was restarted can take its node back
  by name ("Reclaim node" on the client page) once the old session has been
  silent past the heartbeat timeout; the old token stops working.
- **Dataset checks at join**: a `data.yaml` whose `path:` points to another
  machine is repaired (a corrected copy in `~/.fedyolo_client/data/`; the
  original is untouched) when its train/val folders sit next to it; class
  names must match the node's classes exactly. Problems are reported at
  join, not at the first training cycle.

## Persistence

Rooms, passwords, node tokens (hashed), counters, checkpoint history and
unfinished sync rounds are saved under `--state-dir`
(`runs/net_server_state`) after every change and restored at startup:
restarting the server resumes every room, and connected clients carry on
after their retries succeed. Not persisted: the async pull cache and the
CA2FL cache (async rooms rebuild them as nodes submit).

## Evaluation — developer mode only

In real use the server never holds labeled data, so **server-side
evaluation is off**. Starting the server with `--developer-mode` enables it
(and warns at startup); rooms then evaluate every `eval_interval` rounds /
submissions and at the end, in a background thread, against:

1. a dataset path entered at room creation (the field only appears in
   developer mode), else
2. the config's own `eval_data_yaml`, else
3. the nodes' validation splits, when the server can reach them (testing
   everything on one machine).

## Static vs. dynamic rooms

- **Static** — loaded from a config YAML: `nodes`/`global_classes` fixed and
  server-authoritative.
- **Dynamic** — built with a class ceiling (`max_classes`), model arch/imgsz
  and federation hyperparameters. Each client declares its class names on
  join; global ids are assigned in order of first appearance. **Sync**
  dynamic rooms freeze the node/class set at Start; **async** ones stay open
  (new nodes can join, nodes can declare more classes between cycles) and
  never auto-finish — close them with **Finish room**.

## Checkpoints

Every round (sync) / submission (async) is checkpointed, last **10** kept
per room. **Serve this** pins which checkpoint new pulls receive, without
touching training; **Rollback here** resets the room's live state to that
checkpoint and deletes everything newer.

## Merge rules and training options

Sync rooms pass every merge option of the config through (`merge_rule`
mean/ties/fisher, `normalize_steps`, `weight_basis`); async rooms use
`AsyncFedServer` itself (`async_rule`, `async_mix`, CA2FL/FedBuff).
Pseudo-labeling works as in the simulation: the teacher is the previous
round's global model (sync) or the pulled model (async).

## Layout

- `protocol.py` — wire format: encode/decode the global model and node
  updates (plain tensors, `weights_only=True`, validation).
- `rooms.py` — `Room`: static/dynamic config, node registry and tokens,
  sync barrier with lost-node handling, async wrapper over
  `AsyncFedServer`, checkpoints/rollback, persistence, developer-mode eval.
- `server_app.py` — FastAPI server: admin/node auth, REST + WebSocket,
  dashboard (`static/server_dashboard.html`), HTTPS, startup restore.
- `client_app.py` — FastAPI client: one node's pull→train→push loop with
  retries and heartbeats, dataset checks, the personal-room page
  (`static/client_dashboard.html`).
