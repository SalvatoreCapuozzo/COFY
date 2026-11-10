import os
import json
import shutil
import paramiko
import yaml
import glob
from datetime import datetime
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from ultralytics.utils.downloads import download
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

# .env lives alongside this script (legacy/.env) -- load explicitly by path
# rather than relying on python-dotenv's CWD-upward search, which breaks as
# soon as this script isn't run with the repo root as CWD.
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'), override=True)

# Force Ultralytics dataset root into the repo workspace for local runs
settings['datasets_dir'] = os.path.abspath(os.path.join(script_dir, 'datasets'))

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

SSH_PORT = 22
SERVER_IP = os.getenv("SERVER_IP", "").strip().replace('"', '').replace("'", "")
SSH_USER = os.getenv("REMOTE_USER", "").strip().replace('"', '').replace("'", "")
SSH_PASSWORD = os.getenv("PASSWORD", "").strip().replace('"', '').replace("'", "")

SERVER_UPLOAD_DIR = "server_node/uploads"
SERVER_DOWNLOAD_DIR = "server_node/global_model"

# --- Connection Mode Globals ---
CONNECTION_MODE = "remote" 
LOCAL_SERVER_BASE_DIR = os.path.abspath(os.path.join(script_dir, "server_node"))

LOCAL_MODELS_DIR = "runs/detect"
DOWNLOADED_MODELS_DIR = "global_models"
os.makedirs(DOWNLOADED_MODELS_DIR, exist_ok=True)


def _ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=SERVER_IP, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD,
        look_for_keys=False, allow_agent=False, timeout=15
    )
    return ssh


class SmartFLTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_train_start", self._apply_strategy_freezing)

    def _apply_strategy_freezing(self, trainer):
        is_round_1 = "global_model" not in str(trainer.args.model)
        strategy = getattr(self, 'strategy', 'fedhead')
        
        # 1. Capture global weights for FedProx proximal regularization
        if strategy == 'fedprox':
            self.global_weights_dict = {
                name: param.clone().detach().to(trainer.device)
                for name, param in trainer.model.named_parameters()
            }
            self.mu = 0.05
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.momentum = 0.1
            for param in trainer.model.parameters():
                param.requires_grad = True
            print(f"\n🧠 [SmartFL] Mode: FEDPROX (Full Model, Proximal Regularization)\n")
            return

        # 2. Capture global weights for contrastive mathematical anchors (FedCon)
        if not is_round_1 and strategy == 'fedcon':
            self.global_weights_dict = {
                name: param.clone().detach().to(trainer.device)
                for name, param in trainer.model.named_parameters()
            }
            self.mu = 0.05               
            self.head_temperature = 0.1  
        
        # 3. UNFREEZE THE NETWORK (FedCon & YOLOINC)
        if strategy in ['yoloinc', 'fedcon']:
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.momentum = 0.1 
            for param in trainer.model.parameters(): 
                param.requires_grad = True
            print(f"\n🧠 [SmartFL] Mode: {strategy.upper()} (Full Model Unfrozen & Learning with Calibrated Loss)\n")
            return
            
        # 3. ULTRAFLWR YOLO-PA (Partial Aggregation)
        if strategy.startswith('yolopa_'):
            parts = strategy.split('_')[1]
            update_backbone = 'backbone' in parts
            update_neck = 'neck' in parts
            update_head = 'head' in parts
            
            detect_idx = len(trainer.model.model) - 1
            
            frozen, unfrozen = 0, 0
            for name, param in trainer.model.named_parameters():
                try:
                    mod_idx = int(name.split('.')[1])
                except (IndexError, ValueError):
                    mod_idx = -1
                    
                is_backbone = 0 <= mod_idx <= 9
                is_neck = 10 <= mod_idx < detect_idx
                is_head = mod_idx == detect_idx
                
                if (is_backbone and update_backbone) or \
                   (is_neck and update_neck) or \
                   (is_head and update_head):
                    param.requires_grad = True
                    unfrozen += 1
                else:
                    param.requires_grad = False
                    frozen += 1
            
            # UltraFlwr locks BN stats for frozen components to prevent distribution shift
            for name, module in trainer.model.named_modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    try:
                        mod_idx = int(name.split('.')[1])
                    except (IndexError, ValueError):
                        mod_idx = -1
                    is_backbone = 0 <= mod_idx <= 9
                    is_neck = 10 <= mod_idx < detect_idx
                    is_head = mod_idx == detect_idx
                    
                    if not ((is_backbone and update_backbone) or \
                           (is_neck and update_neck) or \
                           (is_head and update_head)):
                        module.track_running_stats = False
                        module.eval()
            
            print(f"\n🧠 [SmartFL] Mode: UltraFlwr YOLO-PA ({parts.upper()})")
            print(f"🧠 [SmartFL] Parameters -> Frozen: {frozen} | Trainable: {unfrozen}\n")
            return

        # 4. LEGACY FREEZING (FedHead, Stitch, etc.)
        detect_idx = len(trainer.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'
        
        frozen, unfrozen = 0, 0
        for name, param in trainer.model.named_parameters():
            if not name.startswith(detect_prefix):
                param.requires_grad = False 
                frozen += 1
            else:
                if is_round_1:
                    param.requires_grad = True
                    unfrozen += 1
                else:
                    is_final_cls = ".cv3." in name and ".2." in name
                    if is_final_cls:
                        param.requires_grad = True; unfrozen += 1
                    else:
                        param.requires_grad = False; frozen += 1

        if strategy == 'stitch':
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.track_running_stats = False
                    module.eval()  
            print(f"\n🧠 [SmartFL] Mode: Stitch (BN frozen to prevent feature drift)\n")
        else:
            print(f"\n🧠 [SmartFL] Mode: {'FULL HEAD (Round 1)' if is_round_1 else 'FINAL LAYER ONLY (Round 2+)'}")
            print(f"🧠 [SmartFL] Parameters -> Frozen: {frozen} | Trainable: {unfrozen}\n")

    @torch.no_grad()
    def optimizer_step(self):
        strategy = getattr(self, 'strategy', 'fedhead')
        if strategy == 'fedprox' and hasattr(self, 'global_weights_dict'):
            scale = self.scaler.get_scale()
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.global_weights_dict:
                    global_w = self.global_weights_dict[name].to(param.device)
                    prox_grad = self.mu * (param.data - global_w)
                    param.grad.data.add_(prox_grad * scale)

        if strategy == 'fedcon' and hasattr(self, 'global_weights_dict'):
            scale = self.scaler.get_scale()
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.global_weights_dict:
                    global_w = self.global_weights_dict[name].to(param.device)
                    contrastive_grad = self.mu * (param.data - global_w)
                    if 'cv3' in name or 'one2one_cv3' in name:
                        contrastive_grad *= (1.0 / self.head_temperature)
                    param.grad.data.add_(contrastive_grad * scale)
        super().optimizer_step()




def setup_local_dataset(target_class_names, node_name="interactive", max_images=None):
    if isinstance(target_class_names, str):
        target_class_names = [target_class_names]
        
    model_name = "yolo26n.pt"
    if not os.path.exists(model_name): download(f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}")

    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')
    if not os.path.exists(base_dir) or not any(os.path.exists(os.path.join(base_dir, ldir)) for ldir, _ in [('labels/train', 'images/train'), ('labels/train2012', 'images/train2012'), ('labels', 'images')]):
        os.makedirs(base_dir, exist_ok=True)
        try:
            YOLO(model_name).train(data='VOC.yaml', epochs=0, imgsz=640)
        except Exception as e:
            print(f"❌ Failed to prepare VOC dataset in {base_dir}: {e}")
            raise

    for ldir, idir in [('labels/train', 'images/train'), ('labels/train2012', 'images/train2012'), ('labels', 'images')]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir): break

    voc_classes = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
    
    original_ids = {}
    for c in target_class_names:
        orig_id = next((k for k, v in voc_classes.items() if v.lower() == c.lower()), None)
        if orig_id is not None: original_ids[orig_id] = c
    
    if not original_ids: raise ValueError("None of the specified classes were found in the dataset.")

    local_map = {orig: loc for loc, orig in enumerate(original_ids.keys())}
    out_dir = os.path.abspath(f"./client_{node_name}_data")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)

    for split in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    all_samples = []
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'): continue
        with open(os.path.join(labels_dir, label_file), 'r') as f: lines = f.readlines()
        filtered = []
        for line in lines:
            if not line.strip(): continue
            try:
                cls_id = int(line.split()[0])
                if cls_id in local_map:
                    parts = line.split()
                    parts[0] = str(local_map[cls_id])
                    filtered.append(" ".join(parts) + "\n")
            except ValueError: pass
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(images_dir, img_filename)):
                all_samples.append((label_file, img_filename, filtered))
        if max_images and len(all_samples) >= max_images: break

    split_idx = int(len(all_samples) * 0.8)
    splits = {'train': all_samples[:split_idx], 'val': all_samples[split_idx:]}

    for split, samples in splits.items():
        for label_file, img_filename, filtered in samples:
            with open(os.path.join(out_dir, 'labels', split, label_file), 'w') as f: f.writelines(filtered)
            shutil.copy(os.path.join(images_dir, img_filename), os.path.join(out_dir, 'images', split, img_filename))

    names_dict = {loc: original_ids[orig] for orig, loc in local_map.items()}
    yaml_path = f"client_{node_name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": out_dir, "train": "images/train", "val": "images/val", "nc": len(names_dict), "names": names_dict}, f, sort_keys=False)
        
    return yaml_path, f"client_{node_name}", len(splits['train'])


def train_and_send():
    server_info = fetch_server_info()
    if not server_info: return
    strategy = server_info['strategy']

    classes_input = input("\nEnter classes to train (comma-separated, e.g., 'person, car'): ").strip()
    target_class_names = [c.strip() for c in classes_input.split(',')] if classes_input else ['person']
    
    node_name = input("Enter a name for this client node (e.g., 'node_1', 'interactive'): ").strip() or "interactive"
    
    limit_input = input("Max images? (Enter for ALL): ").strip()
    max_images = int(limit_input) if limit_input.isdigit() else None

    
    
    yaml_path, client_id, num_samples = setup_local_dataset(target_class_names, node_name, max_images)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    client_run_name = f"{client_id}_{strategy}_{ts}"

    if strategy in ['ties', 'fedavg']:
        starting_model = "yolo26n.pt"
        SmartFLTrainer.is_round_1 = True
    else:
        global_model = download_global_model(strategy)
        if global_model:
            starting_model = global_model
            SmartFLTrainer.is_round_1 = False
        else:
            starting_model = "yolo26n.pt"
            SmartFLTrainer.is_round_1 = True

    SmartFLTrainer.strategy = strategy

    print(f"\n🚀 Training starting -> {client_run_name}")
    print(f"\n🚀 Training starting -> {client_run_name}")
    trainer = SmartFLTrainer(overrides=dict(
        model=starting_model, data=yaml_path, epochs=50, imgsz=640,
        name=client_run_name, lr0=0.01, patience=15,
        # ==========================================
        # CRITICAL RAM FIXES FOR FULL-MODEL TRAINING
        # ==========================================
        batch=4,        # Forces a tiny batch size to save VRAM
        workers=2,      # Stops multi-processing from copying the model into System RAM
        cache=False,    # Prevents dataset from being loaded entirely into RAM
        close_mosaic=0  # Disables a highly memory-intensive augmentation
        # ==========================================
    ))
    trainer.train()

    best_weights = os.path.abspath(f"runs/detect/{client_run_name}/weights/best.pt")
    if os.path.exists(best_weights):
        upload_to_server(client_run_name, best_weights, target_class_names, num_samples)


def select_file_interactive(prompt_text, search_pattern):
    files = sorted(glob.glob(search_pattern), key=os.path.getmtime, reverse=True)
    if not files: return None
    print(f"\n{prompt_text}")
    for i, f in enumerate(files): print(f"  [{i+1}] {f}")
    print(f"  [0] Cancel")
    try:
        choice = int(input("Select number: ").strip())
        return files[choice - 1] if choice != 0 else None
    except (ValueError, IndexError): return None


def validate_and_compare():
    print("\n--- Validate and Compare Metrics ---")
    global_model_path = select_file_interactive("Select a Global Model to Validate:", f"{DOWNLOADED_MODELS_DIR}/*.pt")
    if not global_model_path: return

    global_model = YOLO(global_model_path)
    global_names = global_model.names
    print(f"\n🌍 Global Model Classes: {global_names}")

    voc_classes = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
    global_to_voc = {g_id: v_id for g_id, g_name in global_names.items() for v_id, v_name in voc_classes.items() if g_name.lower() == v_name.lower()}

    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')

    for ldir, idir in [('labels/val2012', 'images/val2012'), ('labels/val', 'images/val'), ('labels/train2012', 'images/train2012')]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir): break

    if not os.path.exists(labels_dir):
        print(f"❌ Could not find VOC labels directory in {base_dir}")
        return

    val_dir = os.path.abspath("./global_val_data_manual")
    out_img_dir = os.path.join(val_dir, "images", "val")
    out_lbl_dir = os.path.join(val_dir, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True); os.makedirs(out_lbl_dir, exist_ok=True)

    images_copied = 0
    for label_file in os.listdir(labels_dir)[:500]:
        if not label_file.endswith('.txt'): continue
        with open(os.path.join(labels_dir, label_file), 'r') as f: lines = f.readlines()
        filtered = []
        for line in lines:
            if not line.strip(): continue
            try:
                orig_id = int(line.split()[0])
                for g_id, v_id in global_to_voc.items():
                    if orig_id == v_id:
                        parts = line.split()
                        parts[0] = str(g_id)
                        filtered.append(" ".join(parts) + "\n")
            except ValueError: pass
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            src_img = os.path.join(images_dir, img_filename)
            if os.path.exists(src_img):
                with open(os.path.join(out_lbl_dir, label_file), 'w') as f: f.writelines(filtered)
                shutil.copy(src_img, os.path.join(out_img_dir, img_filename))
                images_copied += 1

    yaml_path = "global_val_manual.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": val_dir, "train": "images/val", "val": "images/val", "nc": len(global_names), "names": global_names}, f, sort_keys=False)

    print(f"\n⚙️ Running Global Model Validation ({images_copied} images)...")
    global_metrics = global_model.val(data=yaml_path, split='val', verbose=False)

    local_nodes_input = input("\nEnter local node names to compare (comma-separated, e.g. 'node_0, interactive'): ").strip()
    local_nodes = [c.strip() for c in local_nodes_input.split(',')] if local_nodes_input else []
    local_results = {}
    
    for node in local_nodes:
        pattern = f"{LOCAL_MODELS_DIR}/client_{node}_*/weights/best.pt"
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        local_yaml = f"client_{node}.yaml"
        if matches and os.path.exists(local_yaml):
            latest_model_path = matches[0]
            print(f"⚙️ Running Local Validation for '{node}'...")
            local_model = YOLO(latest_model_path)
            local_metrics = local_model.val(data=local_yaml, split='val', verbose=False)
            local_results[node] = local_metrics.box.map50

    print("\n" + "=" * 50 + "\n🏆 FEDERATED LEARNING METRICS COMPARISON 🏆\n" + "=" * 50)
    for node, map50 in local_results.items():
        print(f"\n[ LOCAL MODEL ('{node}') ]\n  - Overall mAP@50:      {map50:.3f}")
        
    print(f"\n[ GLOBAL MODEL (Merged Knowledge) ]\n  - Target Model:        {os.path.basename(global_model_path)}\n  - Overall mAP@50:      {global_metrics.box.map50:.3f}\n  - Classes Known:       {len(global_names)}\n\n[ GLOBAL MODEL Class Breakdown ]")
    for i, class_name in global_names.items():
        try:
            idx = global_metrics.ap_class_index.tolist().index(i)
            print(f"  - {class_name:<15}: mAP@50 = {global_metrics.box.maps[idx]:.3f}")
        except ValueError: print(f"  - {class_name:<15}: No instances found in validation sample.")
    print("=" * 50)


def run_inference():
    print("\n--- Run Inference ---")
    print("1. Select a Global Model")
    print("2. Select a Local Model")
    choice = input("Select: ").strip()

    model_path = None
    if choice == '1':
        model_path = select_file_interactive("Select a Global Model:", f"{DOWNLOADED_MODELS_DIR}/*.pt")
    elif choice == '2':
        model_path = select_file_interactive("Select a Local Model:", f"{LOCAL_MODELS_DIR}/**/best.pt")

    if not model_path: return
    
    img_path = input("\nImage path (Enter for bus.jpg): ").strip() or "bus.jpg"
    if img_path == "bus.jpg" and not os.path.exists(img_path): download("https://ultralytics.com/images/bus.jpg")

    model = YOLO(model_path)
    model.predict(source=img_path, save=True, show=False)
    print("✅ Inference complete. Check runs/detect/predict/")


def upload_to_server(client_id, weights_path, class_names, num_samples=100):
    """Dynamically routes file transfer via SSH or Local OS copy."""
    print(f"\nUploading {weights_path} to server ({CONNECTION_MODE.upper()} mode)...")
    if isinstance(class_names, str): class_names = [class_names]
    meta = {"client_id": client_id, "class_names": class_names, "num_samples": num_samples}
    
    if CONNECTION_MODE == "local":
        upload_dir = os.path.join(LOCAL_SERVER_BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        shutil.copy(weights_path, os.path.join(upload_dir, f"{client_id}_weights.pt"))
        with open(os.path.join(upload_dir, f"{client_id}_meta.json"), "w") as f:
            json.dump(meta, f)
    else:
        ssh = _ssh_connect(); sftp = ssh.open_sftp()
        sftp.put(weights_path, f"{SERVER_UPLOAD_DIR}/{client_id}_weights.pt")
        with open("meta.json", "w") as f: json.dump(meta, f)
        sftp.put("meta.json", f"{SERVER_UPLOAD_DIR}/{client_id}_meta.json")
        sftp.close(); ssh.close()
        if os.path.exists("meta.json"): os.remove("meta.json")
        
    print("✅ Transfer complete!")

def fetch_server_info():
    print(f"\n🔄 Connecting to server ({CONNECTION_MODE.upper()} mode) to handshake...")
    try:
        if CONNECTION_MODE == "local":
            info_path = os.path.join(LOCAL_SERVER_BASE_DIR, "global_model", "server_info.json")
            if not os.path.exists(info_path):
                raise ValueError("Server is not running or hasn't initialized.")
            with open(info_path, "r") as f: return json.load(f)
        else:
            ssh = _ssh_connect(); sftp = ssh.open_sftp()
            try:
                sftp.get(f"{SERVER_DOWNLOAD_DIR}/server_info.json", "local_server_info.json")
                with open("local_server_info.json", "r") as f: info = json.load(f)
                sftp.close(); ssh.close()
                return info
            except FileNotFoundError:
                sftp.close(); ssh.close()
                raise ValueError("Server is not running or hasn't initialized.")
    except Exception as e:
        print(f"❌ Handshake failed: {e}"); return None

def download_global_model(strategy):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(DOWNLOADED_MODELS_DIR, f"global_model_{strategy}_{ts}.pt")
    try:
        if CONNECTION_MODE == "local":
            remote_path = os.path.join(LOCAL_SERVER_BASE_DIR, "global_model", "global_model.pt")
            if not os.path.exists(remote_path): return None
            shutil.copy(remote_path, local_path)
        else:
            ssh = _ssh_connect(); sftp = ssh.open_sftp()
            sftp.get(f"{SERVER_DOWNLOAD_DIR}/global_model.pt", local_path)
            sftp.close(); ssh.close()
            
        if os.path.getsize(local_path) < 1000: os.remove(local_path); return None
        return local_path
    except Exception: return None

def trigger_server_reset():
    print("\n⚠️ WARNING: This will archive the server's current global model.")
    confirm = input("Are you sure you want to reset the server session? (y/N): ").strip().lower()
    if confirm == 'y':
        try:
            cmd_data = {"command": "reset", "timestamp": str(datetime.now())}
            if CONNECTION_MODE == "local":
                upload_dir = os.path.join(LOCAL_SERVER_BASE_DIR, "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                with open(os.path.join(upload_dir, "CMD_RESET.json"), "w") as f:
                    json.dump(cmd_data, f)
            else:
                with open("CMD_RESET.json", "w") as f: json.dump(cmd_data, f)
                ssh = _ssh_connect(); sftp = ssh.open_sftp()
                sftp.put("CMD_RESET.json", f"{SERVER_UPLOAD_DIR}/CMD_RESET.json")
                sftp.close(); ssh.close(); os.remove("CMD_RESET.json")
            print("✅ Reset command sent!")
        except Exception as e: print(f"❌ Failed to send reset command: {e}")


if __name__ == "__main__":
    print("\n=== DAFYOLO Connection Setup ===")
    print("How is your server running?")
    print("  [1] Remote Server (via SSH to Cloud/Remote Machine)")
    print("  [2] Local Server (Running in another terminal on THIS machine)")
    
    conn_choice = input("Select connection mode (1-2): ").strip()
    
    if conn_choice == '2':
        CONNECTION_MODE = 'local'
        print(f"\n📂 The default local server path is: {LOCAL_SERVER_BASE_DIR}")
        local_path = input("Press Enter to use default, or type a custom path: ").strip()
        if local_path: LOCAL_SERVER_BASE_DIR = local_path
        print(f"✅ Running in LOCAL mode targeting: {LOCAL_SERVER_BASE_DIR}")
    else:
        CONNECTION_MODE = 'remote'
        print(f"✅ Running in REMOTE mode targeting: {SERVER_IP}")
        
    while True:
        print("\n=== DAFYOLO Smart Client (v6) ===")
        print("1. Sync & Train (Auto-Configured)")
        print("2. Download Global Model Backup")
        print("3. Run Inference (Interactive Selector)")
        print("4. Validate and Compare Metrics 📊")
        print("5. 🧨 Start New Server Session (Hot Reset)")
        print("6. Exit")
        
        choice = input("Select (1-6): ").strip()
        if choice == '1': train_and_send()
        elif choice == '2': 
            server_info = fetch_server_info()
            if server_info: download_global_model(server_info['strategy'])
        elif choice == '3': run_inference()
        elif choice == '4': validate_and_compare()
        elif choice == '5': trigger_server_reset()
        elif choice == '6': break