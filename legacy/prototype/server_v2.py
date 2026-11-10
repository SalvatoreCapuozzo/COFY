import os
import time
import json
import torch
import torch.nn as nn
import shutil
import copy
from datetime import datetime
from ultralytics import YOLO

from server_paths import UPLOAD_DIR, GLOBAL_MODEL_DIR, PROCESSED_DIR
# --- Server Config ---
# Paths come from server_paths.yaml (macos_laptop / linux_gpu_box), picked
# automatically by OS or via DAFYOLO_SERVER_PROFILE -- see server_paths.py.

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ==============================================================================
# SERVER STRATEGY OVERVIEW
#
# FedHead (RECOMMENDED — new in v2):
#   The core idea: backbone is treated as a frozen shared foundation;
#   only the detect head is federated. Within the detect head:
#     • cv3 FINAL 1x1 conv  →  per-class injection (target_id slot only)
#     • cv3 intermediate convs → incremental FedAvg across clients
#     • cv2 convs (box regression, class-agnostic) → incremental FedAvg
#     • DFL / anchor layers  → skip (shared, never modified by frozen clients)
#   Backbone and neck layers → NEVER modified after initialisation.
#
#   Why FedHead works for extreme non-IID:
#   With clients trained via HeadOnlyTrainer (backbone frozen), all clients
#   operate in the SAME COCO-pretrained feature space. The detect head's
#   intermediate layers receive identical input distributions across clients,
#   so FedAvg on those layers is mathematically sound. The final cv3 layer
#   carries class-specific logits and is injected surgically per class.
#
# Stitch (conservative fallback):
#   Only injects the cv3 final layer. Zero interference with ALL other layers.
#   Use this if clients are training their full detect head OR if FedAvg on
#   intermediate layers causes regression.
#
# TIES-Merging, FedAvg (full model), DFKD — kept for comparison but are
#   fundamentally incompatible with extreme non-IID setups where each client
#   holds a disjoint class set. Their backbone averaging destroys the shared
#   feature space.
# ==============================================================================


class FLServer:
    def __init__(self, strategy='fedhead'):
        self.global_model = None
        self.registry = {}      # class_name → global class ID
        self.nc = 0
        self.strategy = strategy
        self.merge_counts = {}  # tracks how many clients have contributed to each layer group
        self.bootstrap_server()

    # ------------------------------------------------------------------
    # BOOTSTRAP
    # ------------------------------------------------------------------

    def bootstrap_server(self):
        """Restores existing model, or rebuilds it from the archive vault."""
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")

        if os.path.exists(global_model_path):
            print(f"Loading existing global model from {global_model_path}...")
            self.global_model = YOLO(global_model_path)
            self.registry = {v: k for k, v in self.global_model.model.names.items()}
            self.nc = len(self.registry)
            print(f"Restored Global Classes: {self.registry}")
        else:
            processed_metas = sorted(
                [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_meta.json')],
                key=lambda x: os.path.getmtime(os.path.join(PROCESSED_DIR, x))
            )
            if processed_metas:
                print(f"Rebuilding from {len(processed_metas)} archived updates "
                      f"[strategy: {self.strategy.upper()}]...")
                for meta_file in processed_metas:
                    try:
                        with open(os.path.join(PROCESSED_DIR, meta_file), 'r') as f:
                            meta = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    class_name = meta.get('class_name')
                    weights_file = meta_file.replace('_meta.json', '_weights.pt')
                    weights_path = os.path.join(PROCESSED_DIR, weights_file)
                    if class_name and os.path.exists(weights_path):
                        self.merge_client(weights_path, class_name)
            else:
                print(f"Server Initialized [{self.strategy.upper()}]. Waiting for first client...")

    # ------------------------------------------------------------------
    # HEAD EXPANSION
    # ------------------------------------------------------------------

    def _expand_classification_head(self):
        """Expands cv3 (and one2one_cv3 if present) from nc to nc+1 classes."""
        print(f"Expanding classification head: {self.nc} → {self.nc + 1} classes...")
        head = self.global_model.model.model[-1]

        cv3_lists = []
        if hasattr(head, 'cv3'):
            cv3_lists.append(head.cv3)
        if hasattr(head, 'one2one_cv3'):
            cv3_lists.append(head.one2one_cv3)

        for cv3_module in cv3_lists:
            for i in range(len(cv3_module)):
                seq = cv3_module[i]
                last_idx = len(seq) - 1
                last_layer = seq[last_idx]
                old_conv = last_layer if isinstance(last_layer, nn.Conv2d) else last_layer.conv

                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=self.nc + 1,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=(old_conv.bias is not None)
                ).to(old_conv.weight.device)

                with torch.no_grad():
                    # Preserve existing class weights exactly
                    new_conv.weight[:self.nc] = old_conv.weight
                    # Initialise new class slot with small noise (not zero)
                    nn.init.normal_(new_conv.weight[self.nc:], std=0.01)
                    if old_conv.bias is not None:
                        new_conv.bias[:self.nc] = old_conv.bias
                        nn.init.zeros_(new_conv.bias[self.nc:])

                if isinstance(last_layer, nn.Conv2d):
                    seq[last_idx] = new_conv
                else:
                    seq[last_idx].conv = new_conv

        self.nc += 1
        self.global_model.model.nc = self.nc
        if hasattr(head, 'nc'):
            head.nc = self.nc
        if hasattr(head, 'no'):
            head.no += 1

    # ------------------------------------------------------------------
    # FIRST CLIENT INITIALIZATION — always from COCO pretrained backbone
    # ------------------------------------------------------------------

    def _init_from_first_client(self, client_weights_path, class_name):
        """
        ROOT CAUSE FIX #4: Backbone source on first client.

        The merge direction must be: start from the CLIENT model, overwrite its
        backbone+neck with COCO pretrained weights.

        Why NOT the other way around:
        When nc=1, yolo26n builds cv3 intermediate convs with 64 channels.
        When nc=80 (base COCO), cv3 intermediates are 80 channels.
        These shapes are structurally incompatible — you cannot load a nc=1
        client head into a nc=80 base model even with strict=False, because
        strict=False skips missing keys but still raises on shape mismatches.

        Correct approach: use the CLIENT model as the container (its cv3
        dimensions already match nc=1), and replace only backbone+neck keys
        whose shapes match the COCO base. The detect head stays as-is from
        the client.

        This guarantees:
          • Backbone = pure COCO pretrained (neutral, multi-class feature space)
          • Head     = first client's trained single-class output (correct dims)
        """
        print(f"First client '{class_name}': Patching client backbone with COCO weights...")
        client_model = YOLO(client_weights_path)
        base_model   = YOLO("yolo26n.pt")

        client_sd = client_model.model.state_dict()
        base_sd   = base_model.model.state_dict()

        # Use client model's layer count to find detect head prefix
        detect_idx    = len(client_model.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'

        # Build merged state dict:
        #   backbone+neck keys  →  take from COCO base (same shapes, not part of head)
        #   detect head keys    →  keep client's trained weights (nc=1 dimensions)
        merged_sd = {}
        for key, client_tensor in client_sd.items():
            if key.startswith(detect_prefix):
                # Detect head: keep client's trained weights exactly
                merged_sd[key] = client_tensor.clone()
            elif key in base_sd and base_sd[key].shape == client_tensor.shape:
                # Backbone/neck: replace with COCO pretrained (same shape guaranteed
                # because only the head differs between nc=1 and nc=80)
                merged_sd[key] = base_sd[key].clone()
            else:
                # Fallback: keep client weight (handles any unexpected key)
                merged_sd[key] = client_tensor.clone()

        client_model.model.load_state_dict(merged_sd, strict=True)

        self.global_model = client_model
        self.nc = 1
        self.registry[class_name] = 0
        self.global_model.model.names = {0: class_name}
        self.merge_counts['head_intermediate'] = 1
        self._save_model()

    # ------------------------------------------------------------------
    # CORE MERGE ENTRY POINT
    # ------------------------------------------------------------------

    def merge_client(self, client_weights_path, class_name):
        print(f"\n--- Processing client: '{class_name}' [{self.strategy.upper()}] ---")

        if self.global_model is None:
            self._init_from_first_client(client_weights_path, class_name)
            return

        # Register new class and expand head if needed
        if class_name not in self.registry:
            self.registry[class_name] = self.nc
            self.global_model.model.names[self.nc] = class_name
            self._expand_classification_head()
            print(f"New class '{class_name}' → Global ID {self.registry[class_name]}")

        target_id = self.registry[class_name]
        print(f"Merging local ID 0 → global ID {target_id} ...")

        global_sd = self.global_model.model.state_dict()
        client_sd = YOLO(client_weights_path).model.state_dict()

        detect_idx = len(self.global_model.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'

        # Dispatch to selected strategy
        if self.strategy == 'fedhead':
            self._merge_fedhead(global_sd, client_sd, target_id, detect_prefix)
        elif self.strategy == 'stitch':
            self._merge_stitch(global_sd, client_sd, target_id)
        elif self.strategy == 'ties':
            self._merge_ties(global_sd, client_sd, target_id, detect_prefix)
        elif self.strategy == 'fedavg':
            self._merge_fedavg(global_sd, client_sd, target_id, detect_prefix)
        elif self.strategy == 'dfkd':
            self._merge_dfkd(global_sd, client_sd, target_id, detect_prefix, client_weights_path)

        self.global_model.model.load_state_dict(global_sd)
        self._save_model()

    # ==================================================================
    # STRATEGY 1: FedHead  ← RECOMMENDED
    # ==================================================================

    def _merge_fedhead(self, global_sd, client_sd, target_id, detect_prefix):
        """
        FedHead aggregation for frozen-backbone clients:

        Rule 1 — Backbone + neck (keys NOT starting with detect_prefix):
            NEVER TOUCH. These are pure COCO pretrained weights.

        Rule 2 — cv3 / one2one_cv3 FINAL 1x1 (shape mismatch: global has nc rows,
            client has 1 row):
            Inject client's single row into the target_id slot.

        Rule 3 — All other detect head layers (cv3 intermediate, cv2, cv3 early):
            Incremental FedAvg. Since all clients trained with a frozen COCO backbone,
            these layers all received identical COCO-level input features and should
            converge toward similar values. Averaging them improves generalisation.

        Rule 4 — DFL / anchor layers ('dfl', 'stride'):
            Skip — these are constant anchor parameters, never changed by any client.
        """
        n = self.merge_counts.get('head_intermediate', 1)  # clients already merged
        alpha = 1.0 / (n + 1)   # running-average weight for the new client
        self.merge_counts['head_intermediate'] = n + 1

        skip_keys = {'dfl', 'stride', 'anchors'}

        for key in global_sd.keys():
            if key not in client_sd:
                continue

            # RULE 1: Backbone/neck — never touch
            if not key.startswith(detect_prefix):
                continue

            # RULE 4: DFL / anchor constants — skip
            if any(s in key for s in skip_keys):
                continue

            is_cv3_branch = any(x in key for x in ['cv3.', 'one2one_cv3.'])
            g_shape = global_sd[key].shape
            c_shape = client_sd[key].shape

            # RULE 2: cv3 final layer — class-specific injection
            if (is_cv3_branch and
                    ('2.weight' in key or '2.bias' in key) and
                    len(g_shape) > 0 and g_shape[0] == self.nc and
                    len(c_shape) > 0 and c_shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()

            # RULE 3: Other detect head layers — incremental FedAvg
            elif g_shape == c_shape:
                global_sd[key] = (1 - alpha) * global_sd[key] + alpha * client_sd[key]

    # ==================================================================
    # STRATEGY 2: Stitch (conservative, zero interference)
    # ==================================================================

    def _merge_stitch(self, global_sd, client_sd, target_id):
        """
        Injects ONLY the cv3 final 1x1 conv for the new class.
        All other weights (backbone, neck, cv3 intermediate, cv2) are untouched.

        Best choice when: clients trained their full detect head (not backbone-frozen),
        and you want to avoid any cross-client interference in the intermediate layers.
        The backbone must already be COCO-neutral (guaranteed by _init_from_first_client).
        """
        for key in global_sd.keys():
            if key not in client_sd:
                continue
            if (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                    ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and
                    client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()

    # ==================================================================
    # STRATEGY 3: TIES-Merging (Task Vector)
    # ==================================================================

    def _merge_ties(self, global_sd, client_sd, target_id, detect_prefix):
        """
        TIES: trims the task vector (keep top-30% magnitudes) then adds to global.
        NOTE: In the frozen-backbone regime, applying TIES to backbone layers is
        a no-op because client backbone weights == global backbone weights (no task
        vector). It is still applied to detect head layers for correctness.
        """
        n = len(self.registry)
        alpha = 1.0 / n

        for key in global_sd.keys():
            if key not in client_sd:
                continue
            if (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                    ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and
                    client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()
            elif global_sd[key].shape == client_sd[key].shape:
                if any(x in key for x in ['cv2', 'dfl']):
                    continue
                task_vector = client_sd[key].float() - global_sd[key].float()
                num_elements = task_vector.numel()
                if num_elements > 100:
                    keep_ratio = 0.30
                    k_index = max(1, int(num_elements * (1 - keep_ratio)))
                    threshold = torch.kthvalue(task_vector.abs().flatten(), k_index).values
                    task_vector = task_vector * (task_vector.abs() >= threshold)
                global_sd[key] = (global_sd[key].float() + alpha * task_vector).to(global_sd[key].dtype)

    # ==================================================================
    # STRATEGY 4: FedAvg (full model blend)
    # ==================================================================

    def _merge_fedavg(self, global_sd, client_sd, target_id, detect_prefix):
        """Standard FedAvg blended across ALL compatible layers."""
        n = len(self.registry)
        alpha = 1.0 / n

        for key in global_sd.keys():
            if key not in client_sd:
                continue
            if (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                    ('2.weight' in key or '2.bias' in key) and
                    global_sd[key].shape[0] == self.nc and
                    client_sd[key].shape[0] == 1):
                global_sd[key][target_id] = client_sd[key][0].clone()
            elif global_sd[key].shape == client_sd[key].shape:
                if any(x in key for x in ['cv2', 'dfl']):
                    continue
                global_sd[key] = ((1 - alpha) * global_sd[key].float() +
                                  alpha * client_sd[key].float()).to(global_sd[key].dtype)

    # ==================================================================
    # STRATEGY 5: Data-Free Knowledge Distillation (DFKD)
    # ==================================================================

    def _merge_dfkd(self, global_sd, client_sd, target_id, detect_prefix, client_weights_path):
        """
        Improved DFKD:
          1. Inject cv3 final layer (same as stitch).
          2. Run model inversion on synthetic noise to generate pseudo-images
             that activate the CLIENT model strongly.
          3. Distil the client (teacher) and existing global (teacher) into the
             new global (student) using those pseudo-images.

        Increased iterations (50 inversion, 200 distillation) vs original (20/50).
        """
        # Step 1 — inject the classification output
        for key in global_sd.keys():
            if key in client_sd:
                if (any(x in key for x in ['cv3.', 'one2one_cv3.']) and
                        ('2.weight' in key or '2.bias' in key) and
                        global_sd[key].shape[0] == self.nc and
                        client_sd[key].shape[0] == 1):
                    global_sd[key][target_id] = client_sd[key][0].clone()
        self.global_model.model.load_state_dict(global_sd)

        device = next(self.global_model.model.parameters()).device
        client_model = YOLO(client_weights_path)
        client_model.model.eval().to(device)
        for p in client_model.model.parameters():
            p.requires_grad = False

        global_teacher = copy.deepcopy(self.global_model.model).eval().to(device)
        for p in global_teacher.parameters():
            p.requires_grad = False

        # Hooks — capture feature pyramids entering the Detect head
        client_feats, global_feats, student_feats = [], [], []

        def _hook(store):
            def fn(mod, inp, out):
                store.clear()
                x = inp[0] if isinstance(inp[0], (list, tuple)) else [inp[0]]
                store.extend([t.detach() if t.requires_grad else t for t in x])
            return fn

        h1 = client_model.model.model[-1].register_forward_hook(_hook(client_feats))
        h2 = global_teacher.model[-1].register_forward_hook(_hook(global_feats))
        h3 = self.global_model.model.model[-1].register_forward_hook(_hook(student_feats))

        print("[DFKD] Inverting client model to generate pseudo-images (50 steps)...")
        dummy = torch.randn(4, 3, 640, 640, device=device, requires_grad=True)
        opt_noise = torch.optim.Adam([dummy], lr=0.05)
        for _ in range(50):
            opt_noise.zero_grad()
            client_model.model(dummy.clamp(-3, 3))
            loss = -sum(f.mean() for f in client_feats)
            loss.backward()
            opt_noise.step()
        dummy = dummy.detach()

        print("[DFKD] Distilling knowledge (200 steps)...")
        self.global_model.model.train()
        for m in self.global_model.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()   # protect BN statistics

        opt_student = torch.optim.Adam(self.global_model.model.parameters(), lr=0.0003)
        criterion = nn.MSELoss()
        n = len(self.registry)
        alpha = 1.0 / n

        for _ in range(200):
            opt_student.zero_grad()
            with torch.no_grad():
                client_model.model(dummy)
                global_teacher(dummy)
            self.global_model.model(dummy)
            loss = sum(
                criterion(sf, cf) * alpha + criterion(sf, gf) * (1 - alpha)
                for sf, cf, gf in zip(student_feats, client_feats, global_feats)
            )
            loss.backward()
            opt_student.step()

        h1.remove()
        h2.remove()
        h3.remove()

        # Reload updated state dict from the distilled student
        global_sd.update(self.global_model.model.state_dict())

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def _save_model(self):
        out_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        self.global_model.save(out_path)
        print(f"✅ Global model saved → {out_path}")
        print(f"   Classes: {self.registry}")


# ==============================================================================
# SERVER MAIN LOOP
# ==============================================================================

def run_server():
    print("=" * 52)
    print("  🏆  DAFYOLO FEDERATED LEARNING SERVER (v2)  🏆  ")
    print("=" * 52)
    print("\nSelect aggregation strategy:")
    print("  [1] FedHead  — Head FedAvg + class injection  ← RECOMMENDED")
    print("  [2] Stitch   — Final layer injection only (zero interference)")
    print("  [3] TIES     — Task-vector trimming + injection")
    print("  [4] FedAvg   — Full-model mathematical blend")
    print("  [5] DFKD     — Data-Free Knowledge Distillation [experimental]")
    print("=" * 52)

    choice = input("Choice (1–5): ").strip()
    strategy_map = {'1': 'fedhead', '2': 'stitch', '3': 'ties', '4': 'fedavg', '5': 'dfkd'}
    strategy = strategy_map.get(choice, 'fedhead')

    print(f"\nBooting with [{strategy.upper()}] strategy...\n")
    server = FLServer(strategy=strategy)
    print(f"\nWatching {UPLOAD_DIR} for incoming uploads...\n")

    while True:
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")

        # Hot-reload watchdog
        if server.global_model is not None and not os.path.exists(global_model_path):
            print("\n[WATCHDOG] Global model deleted from disk — rebuilding from vault...")
            server.global_model = None
            server.registry = {}
            server.nc = 0
            server.merge_counts = {}
            server.bootstrap_server()

        meta_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('_meta.json')]
        if meta_files:
            print(f"[DEBUG] Incoming: {meta_files}")

        for meta_file in meta_files:
            meta_path = os.path.join(UPLOAD_DIR, meta_file)
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except json.JSONDecodeError:
                print(f"[DEBUG] JSON still writing: {meta_file}, skipping tick.")
                continue

            client_id = meta.get('client_id')
            class_name = meta.get('class_name')
            if not client_id or not class_name:
                continue

            weights_path = os.path.join(UPLOAD_DIR, f"{client_id}_weights.pt")
            if not os.path.exists(weights_path):
                continue

            print(f"[DEBUG] Processing '{class_name}'...")
            server.merge_client(weights_path, class_name)

            # Archive
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(meta_path, os.path.join(PROCESSED_DIR, f"{client_id}_{ts}_meta.json"))
            shutil.move(weights_path, os.path.join(PROCESSED_DIR, f"{client_id}_{ts}_weights.pt"))
            print(f"📦 Archived {client_id}\n")

        time.sleep(5)


if __name__ == "__main__":
    run_server()
