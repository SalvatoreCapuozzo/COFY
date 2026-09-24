# fedyolo.net — real client-server federation

Everything else in this project (`fedyolo.simulate`, `FedServer`,
`AsyncFedServer`) runs a federation inside **one Python process**. This
subsystem is additive, not a replacement: it runs the exact same
config-driven federation as a real distributed system — a server process
with a dashboard reachable by IP, and separate client processes — each
with its own GUI ("personal room") — that pull the global model, train
locally on their own machine's own dataset, and push results back over
HTTP. `federation.mode` decides sync vs. async, same as everywhere else.

## Quick start — everything on one Mac (simulating N nodes)

```bash
python -m fedyolo.net.server_app --port 8900 [--config-dir configs]
python -m fedyolo.net.client_app --port 8901   # one per node
python -m fedyolo.net.client_app --port 8902
```

Open `http://127.0.0.1:8900` for the server dashboard. Create a room
either **from a config file** (pick one from the dropdown — lists
`--config-dir`, default `configs/`, with a preview) or **build one from
scratch** (see "Dynamic rooms" below). The password needed by clients to
join/push shows on the room card, with a refresh button.

Open `http://127.0.0.1:8901`, `8902`, ... for each client's own "personal
room". Enter the server URL, room ID, and password, fetch the room, then
(dynamic rooms only) declare your own classes. Point the dataset field at
a local `data.yaml` — either browse to an existing one, or build one from
a raw folder (the page illustrates the expected `images/{train,val}`,
`labels/{train,val}` layout). Join — the server auto-assigns you the next
free node slot, you don't pick one. Hit **Start federation** on the
server once the nodes you want have joined, then **Start training** on
each client.

## Two Macs on the same Wi-Fi

Same as above, except: `server_app` already binds `0.0.0.0`, so it's
reachable from the LAN automatically. On machine B, point `client_app`'s
"Server URL" field at machine A's LAN IP (`http://192.168.x.x:8900`)
instead of `127.0.0.1`. Each machine only ever needs local disk access to
whichever node's data it's training.

## Static vs. dynamic rooms

- **Static** — loaded from an existing config YAML, exactly like every
  other run in this project: `nodes`/`global_classes` fixed and
  server-authoritative.
- **Dynamic** — built from scratch with just a class ceiling
  (`max_classes`), model arch/imgsz, and federation hyperparameters. No
  predefined nodes or classes: each client declares its own class names
  on join; the server assigns global ids in order of first appearance
  across registrations, reusing an id if another node already claimed
  that name. **Sync** dynamic rooms freeze the node/class set the moment
  you hit Start (a barrier needs a fixed, known participant set); **async**
  ones stay open indefinitely — a new node can join mid-run, and an
  already-joined node can declare *more* classes between cycles via the
  client page's "declare more classes" field. A class added late starts
  training from its untouched random init — it simply needs its own
  rounds/cycles to catch up, same as it would centralized. Because async
  dynamic rooms have no fixed participant set, they never auto-finish —
  close one explicitly with the server dashboard's **Finish room** button
  when you're done.

## Checkpoints

Every round (sync) / submission (async) is checkpointed, named by
timestamp + room id, with the last **10** kept per room (older ones
auto-pruned — this project has hit real "No space left on device" crashes
from unbounded retention before). The server dashboard lists them per
room: **Serve this** non-destructively pins which checkpoint new `/pull`
calls hand out (training keeps going on the live state regardless);
**Rollback here** is destructive — resets the room's live state to that
checkpoint and deletes everything newer, for undoing a bad stretch of
training.

## Server-side eval

Static rooms keep the same eval fedyolo.simulate always does (pools each
node's own local val split — the server has that on disk in every setup
this project has actually run). Dynamic rooms have no such local access
by default (the whole point is that data stays on each client's own
machine); pass an **eval dataset yaml** at room creation only if the
server operator happens to have one it can reach, and eval runs against
that instead. Left blank, eval is skipped cleanly rather than assumed.

## What's real vs. simplified here

- Sync and async both reuse the exact aggregation code the in-process
  path uses (`fedyolo.model.aggregate`, `AsyncFedServer`), including for
  dynamic rooms — `aggregate_per_class`'s own docstring already handles
  "classes with zero contributing nodes" (an as-yet-unclaimed slot) by
  keeping them at random init, so no core aggregation changes were
  needed.
- Pseudo-labeling isn't wired through the network protocol — every
  config in this project has it disabled.
- Password auth is a shared secret per room (required for join/push, not
  for pull), appropriate for a trusted local network. No TLS. Scoped
  deliberately to "one Mac, or a couple on the same Wi-Fi" — the natural
  next step before ever exposing this beyond a LAN.
- The local file-browser endpoint (`client_app`'s `/api/browse`) is safe
  specifically because `client_app` binds `127.0.0.1` by default — one
  user, one machine, never exposed to the network.

## Layout

- `rooms.py` — `Room`: static/dynamic config handling, node/class
  registration, the sync round barrier, checkpoint history + rollback,
  password, and a thin wrapper over `AsyncFedServer` for async.
- `server_app.py` — FastAPI app: room registry (list/create/delete),
  config browsing, checkpoint endpoints, REST + WebSocket protocol, the
  server dashboard (`static/server_dashboard.html`), shutdown.
- `client_app.py` — FastAPI app: one node's local pull→train→push loop,
  local file browser + dataset builder, the "personal room" GUI
  (`static/client_dashboard.html`), shutdown.
