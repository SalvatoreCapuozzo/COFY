import os
import time
import json
import torch
import torch.nn as nn
import shutil
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# YOLO26 C3k2 AND SPPF COMPATIBILITY PATCH
# ==============================================================================
import ultralytics.nn.modules.block as block

orig_c3k2_init = block.C3k2.__init__
orig_sppf_init = block.SPPF.__init__

def patched_c3k2_init(self, *args, **kwargs):
    args = list(args)
    if len(args) == 6 and isinstance(args[5], bool):
        shortcut = args[5]
        args[5] = 1  
        args.append(shortcut) 
    orig_c3k2_init(self, *args, **kwargs)

def patched_sppf_init(self, c1, c2, k=5, *args, **kwargs):
    orig_sppf_init(self, c1, c2, k)

block.C3k2.__init__ = patched_c3k2_init
block.SPPF.__init__ = patched_sppf_init

# Anchored to this script's own directory (not the caller's CWD) so it
# resolves the same way regardless of where server_updated.py is launched
# from -- and matches client_updated.py's LOCAL_SERVER_BASE_DIR, which is
# anchored to ITS OWN script directory the same way, for local-mode use.
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_node")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
GLOBAL_MODEL_DIR = os.path.join(BASE_DIR, "global_model")
ARCHIVE_DIR = os.path.join(GLOBAL_MODEL_DIR, "archive")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_models")

for d in [UPLOAD_DIR, GLOBAL_MODEL_DIR, ARCHIVE_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

class FLServer:
    def __init__(self, strategy='fedcon'):
        self.global_model = None
        self.registry = {}
        self.nc = 0
        self.strategy = strategy
        self.total_samples = 0
        self.class_merge_counts = {} 
        self._write_server_info()
        self.bootstrap_server()

    def _write_server_info(self):
        info = {"strategy": self.strategy, "boot_time": datetime.now().strftime("%Y%m%d_%H%M%S")}
        with open(os.path.join(GLOBAL_MODEL_DIR, "server_info.json"), 'w') as f: json.dump(info, f)
        print(f"📡 Broadcasted server info: {info}")

    def bootstrap_server(self):
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        if os.path.exists(global_model_path):
            self.global_model = YOLO(global_model_path)
            self.registry = {v: k for k, v in self.global_model.model.names.items()}
            self.nc = len(self.registry)
        else:
            processed_metas = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_meta.json')],
                                     key=lambda x: os.path.getmtime(os.path.join(PROCESSED_DIR, x)))
            if processed_metas:
                for meta_file in processed_metas:
                    try:
                        with open(os.path.join(PROCESSED_DIR, meta_file), 'r') as f: meta = json.load(f)
                    except json.JSONDecodeError: continue
                    class_names = meta.get('class_names', [meta.get('class_name')])
                    num_samples = meta.get('num_samples', 100)
                    weights_path = os.path.join(PROCESSED_DIR, meta_file.replace('_meta.json', '_weights.pt'))
                    if class_names and os.path.exists(weights_path):
                        self.merge_client(weights_path, class_names, num_samples)

    def reset_session(self):
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(global_model_path):
            shutil.move(global_model_path, os.path.join(ARCHIVE_DIR, f"global_model_archived_{ts}.pt"))

        processed_files = os.listdir(PROCESSED_DIR)
        if processed_files:
            session_archive_dir = os.path.join(ARCHIVE_DIR, f"session_{ts}")
            os.makedirs(session_archive_dir, exist_ok=True)
            for file in processed_files:
                shutil.move(os.path.join(PROCESSED_DIR, file), os.path.join(session_archive_dir, file))

        self.global_model, self.registry, self.nc, self.total_samples, self.class_merge_counts = None, {}, 0, 0, {}
        self._write_server_info()
        print("✨ Server memory wiped. Ready for a brand new Federated Learning session!")

    def _expand_classification_head(self):
        head = self.global_model.model.model[-1]
        cv3_lists = []
        if hasattr(head, 'cv3'): cv3_lists.append(head.cv3)
        if hasattr(head, 'one2one_cv3'): cv3_lists.append(head.one2one_cv3)

        for cv3_module in cv3_lists:
            for i in range(len(cv3_module)):
                seq = cv3_module[i]
                last_idx = len(seq) - 1
                last_layer = seq[last_idx]
                old_conv = last_layer if isinstance(last_layer, nn.Conv2d) else last_layer.conv

                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels, out_channels=self.nc + 1,
                    kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                    padding=old_conv.padding, bias=(old_conv.bias is not None)
                ).to(old_conv.weight.device)

                with torch.no_grad():
                    new_conv.weight[:self.nc] = old_conv.weight
                    nn.init.normal_(new_conv.weight[self.nc:], std=0.01)
                    if old_conv.bias is not None:
                        new_conv.bias[:self.nc] = old_conv.bias
                        nn.init.zeros_(new_conv.bias[self.nc:])

                if isinstance(last_layer, nn.Conv2d): seq[last_idx] = new_conv
                else: seq[last_idx].conv = new_conv

        self.nc += 1
        self.global_model.model.nc = self.nc
        if hasattr(head, 'nc'): head.nc = self.nc
        if hasattr(head, 'no'): head.no += 1

    def _init_from_first_client(self, client_weights_path, class_names, num_samples):
        client_model = YOLO(client_weights_path)
        base_model   = YOLO("yolo26n.pt")
        client_sd = client_model.model.state_dict()
        base_sd   = base_model.model.state_dict()
        detect_idx    = len(client_model.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'

        merged_sd = {}
        for key, client_tensor in client_sd.items():
            if key.startswith(detect_prefix): merged_sd[key] = client_tensor.clone()
            elif key in base_sd and base_sd[key].shape == client_tensor.shape: merged_sd[key] = base_sd[key].clone()
            else: merged_sd[key] = client_tensor.clone()

        client_model.model.load_state_dict(merged_sd, strict=True)
        self.global_model = client_model
        
        self.nc = len(class_names)
        self.registry = {c: i for i, c in enumerate(class_names)}
        self.global_model.model.names = {i: c for i, c in enumerate(class_names)}
        for c in class_names: self.class_merge_counts[c] = 1
        
        self.total_samples = num_samples
        self._save_model()

    def _normalize_class_weights(self, weights_tensor):
        """
        Normalize class-specific weights by L2 norm to account for
        different activation scales across clients.
        """
        if len(weights_tensor.shape) == 1:
            # Bias vector - normalize by itself
            norm = torch.norm(weights_tensor, p=2)
            return weights_tensor / (norm + 1e-8)
        else:
            # Weight matrix - normalize per output channel (class)
            # weights_tensor shape: [out_channels, in_channels, ...]
            # Reshape to [out_channels, -1] for per-channel normalization
            original_shape = weights_tensor.shape
            reshaped = weights_tensor.reshape(original_shape[0], -1)
            norms = torch.norm(reshaped, p=2, dim=1, keepdim=True)
            normalized = reshaped / (norms + 1e-8)
            return normalized.reshape(original_shape)

    def merge_client(self, client_weights_path, class_names, num_samples=100):
        print(f"\n--- Processing client containing: {class_names} [{self.strategy.upper()}] ---")
        if self.global_model is None:
            self._init_from_first_client(client_weights_path, class_names, num_samples)
            return

        for c in class_names:
            if c not in self.registry:
                self.registry[c] = self.nc
                self.global_model.model.names[self.nc] = c
                self._expand_classification_head()
            
            if c not in self.class_merge_counts:
                self.class_merge_counts[c] = 0
            self.class_merge_counts[c] += 1

        global_sd = self.global_model.model.state_dict()
        client_sd = YOLO(client_weights_path).model.state_dict()
        
        # Initialize total_samples if needed
        if self.total_samples == 0: 
            self.total_samples = num_samples
        
        # Use proper weighted averaging for ALL strategies (not just YOLOINC)
        alpha = num_samples / (self.total_samples + num_samples)
        
        # Update total for next iteration
        self.total_samples += num_samples

        base_sd = YOLO("yolo26n.pt").model.state_dict() if self.strategy == 'ties' else None

        for key in global_sd.keys():
            if key not in client_sd: continue
            
            # --- 1. CLASSIFICATION HEAD MERGING (with normalization) ---
            if any(x in key for x in ['cv3.', 'one2one_cv3.']) and ('2.weight' in key or '2.bias' in key):
                if len(global_sd[key].shape) > 0 and len(client_sd[key].shape) > 0:
                    if global_sd[key].shape[0] == self.nc and client_sd[key].shape[0] == len(class_names):
                        for local_id, c_name in enumerate(class_names):
                            target_id = self.registry[c_name]
                            
                            # Normalize client weights by L2 norm
                            normalized_client_weight = self._normalize_class_weights(
                                client_sd[key][local_id:local_id+1]
                            ).squeeze(0)
                            
                            if self.class_merge_counts[c_name] == 1 or self.strategy in ['fedhead', 'stitch']:
                                global_sd[key][target_id] = normalized_client_weight.clone()
                            # FedAvg, YOLOINC, TIES, FedCon, YOLO-PA: Standard averaging with normalized weights
                            elif self.strategy in ['fedavg', 'yoloinc', 'ties', 'fedcon'] or self.strategy.startswith('yolopa_'):
                                normalized_global_weight = self._normalize_class_weights(
                                    global_sd[key][target_id:target_id+1]
                                ).squeeze(0)
                                global_sd[key][target_id] = (
                                    (1 - alpha) * normalized_global_weight.float() + 
                                    alpha * normalized_client_weight.float()
                                )
                                
            # --- 2. BACKBONE / NECK MERGING ---
            elif global_sd[key].shape == client_sd[key].shape:
                if any(x in key for x in ['cv2', 'dfl', 'anchors', 'stride']): continue
                
                if self.strategy == 'ties':
                    if key in base_sd and base_sd[key].shape == client_sd[key].shape:
                        task_vector = client_sd[key].float() - base_sd[key].float()
                        if task_vector.numel() > 100:
                            threshold = torch.kthvalue(task_vector.abs().flatten(), max(1, int(task_vector.numel() * 0.70))).values
                            task_vector = task_vector * (task_vector.abs() >= threshold)
                        global_sd[key] = (global_sd[key].float() + alpha * task_vector).to(global_sd[key].dtype)
                    else:
                        global_sd[key] = ((1 - alpha) * global_sd[key].float() + alpha * client_sd[key].float()).to(global_sd[key].dtype)
                
                # FedCon & YOLO-PA utilizes standard averaging for the backbone!
                elif self.strategy in ['fedavg', 'yoloinc', 'fedcon'] or self.strategy.startswith('yolopa_'):
                    global_sd[key] = ((1 - alpha) * global_sd[key].float() + alpha * client_sd[key].float()).to(global_sd[key].dtype)

        self.global_model.model.load_state_dict(global_sd)
        self._save_model()

    def _save_model(self):
        out_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        self.global_model.save(out_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(ARCHIVE_DIR, f"global_model_{ts}.pt")
        shutil.copy(out_path, archive_path)
        print(f"✅ Global model updated. Classes: {self.registry}")

def run_server():
    print("=" * 52)
    print("  🏆 DAFYOLO SERVER (v6 - Multi-Class Support) 🏆")
    print("=" * 52)
    print("Select Server Merging Strategy:")
    print("  [1] FedHead (Head Injection)")
    print("  [2] Stitch (Zero-Interference Head Stitching)")
    print("  [3] TIES (Task Vector Trimming)")
    print("  [4] FedAvg (Standard Averaging)")
    print("  [5] YOLOINC (Incremental Distillation)")
    print("  [6] FedCon (Federated Contrastive)")
    print("  [7] UltraFlwr YOLO-PA (Partial Aggregation) [NEW]")
    print("  [8] FedProx (Proximal Federated Optimization)")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    strategy_map = {'1': 'fedhead', '2': 'stitch', '3': 'ties', '4': 'fedavg', '5': 'yoloinc', '6': 'fedcon', '8': 'fedprox'}
    
    if choice == '7':
        print("\nSelect YOLO-PA Component(s) to Update:")
        print("  [a] Backbone (Modules 0-9)")
        print("  [b] Neck (Modules 10-22)")
        print("  [c] Head (Module 23)")
        print("  [d] NeckHead (Modules 10-23)")
        print("  [e] BackboneHead (Modules 0-9 + 23)")
        pa_choice = input("Enter choice (a-e): ").strip().lower()
        pa_map = {'a': 'backbone', 'b': 'neck', 'c': 'head', 'd': 'neckhead', 'e': 'backbonehead'}
        selected_strategy = f"yolopa_{pa_map.get(pa_choice, 'neckhead')}"
    else:
        selected_strategy = strategy_map.get(choice, 'fedcon') # Defaults to fedcon if invalid input
        
    server = FLServer(strategy=selected_strategy)
    print(f"\n🚀 Booting Server with [{selected_strategy.upper()}] strategy...")
    print(f"👀 Watching {UPLOAD_DIR} for incoming uploads or commands...\n")

    while True:
        reset_cmd_path = os.path.join(UPLOAD_DIR, "CMD_RESET.json")
        if os.path.exists(reset_cmd_path):
            try:
                with open(reset_cmd_path, 'r') as f: cmd_data = json.load(f)
                if 'strategy' in cmd_data: server.strategy = cmd_data['strategy']
            except Exception: pass
            server.reset_session()
            try: os.remove(reset_cmd_path)
            except OSError: pass
            time.sleep(2)
            continue

        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_model.pt")
        if server.global_model is not None and not os.path.exists(global_model_path):
            server.global_model, server.registry, server.nc, server.total_samples, server.class_merge_counts = None, {}, 0, 0, {}
            server.bootstrap_server()

        meta_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('_meta.json')]
        for meta_file in meta_files:
            meta_path = os.path.join(UPLOAD_DIR, meta_file)
            try:
                with open(meta_path, 'r') as f: meta = json.load(f)
            except json.JSONDecodeError: continue 

            client_id = meta.get('client_id')
            class_names = meta.get('class_names', [meta.get('class_name')])
            num_samples = meta.get('num_samples', 100)
            
            if not client_id or not class_names: continue

            weights_path = os.path.join(UPLOAD_DIR, f"{client_id}_weights.pt")
            if not os.path.exists(weights_path): continue 

            server.merge_client(weights_path, class_names, num_samples)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(meta_path, os.path.join(PROCESSED_DIR, f"{client_id}_{ts}_meta.json"))
            shutil.move(weights_path, os.path.join(PROCESSED_DIR, f"{client_id}_{ts}_weights.pt"))

        time.sleep(3) 

if __name__ == "__main__":
    run_server()