import os
import json
import shutil
import paramiko
import yaml
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from ultralytics.utils.downloads import download
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

# .env lives alongside this script (legacy/.env) -- load explicitly by path
# rather than relying on python-dotenv's CWD-upward search.
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, '.env'))

# --- SSH Configuration ---
SERVER_IP = os.getenv("SERVER_IP")
SSH_PORT = 22
SSH_USER = os.getenv("USERNAME")
SSH_PASSWORD = os.getenv("PASSWORD")
# Paths on the SSH server -- from server_paths.yaml (macos_laptop /
# linux_gpu_box), picked automatically by OS or via DAFYOLO_SERVER_PROFILE.
from server_paths import UPLOAD_DIR as SERVER_UPLOAD_DIR, GLOBAL_MODEL_DIR as SERVER_DOWNLOAD_DIR

# ==============================================================================
# ROOT CAUSE FIX #1: Replace FedProxTrainer with HeadOnlyTrainer
#
# Why FedProxTrainer was broken:
#  - `freeze=23` in the training args conflicts with the surgical freeze callback.
#    YOLO's built-in freeze runs BEFORE the on_train_start callback, creating an
#    unpredictable mixed freeze state. Sometimes backbone layers end up trainable,
#    sometimes the head ends up frozen.
#  - FedProx adds a proximal gradient penalty that pulls the head weights back toward
#    the global model. With a frozen backbone this is counter-productive — it
#    actively prevents the head from specializing on its class.
#  - Training the backbone at all is the root cause of low global mAP.
#    Each client's backbone drifts into a class-specialized feature space
#    (car-space, bicycle-space, person-space). Averaging or stitching these
#    divergent backbones onto a single global model corrupts the feature extractor
#    for ALL classes simultaneously.
#
# The fix: The COCO-pretrained backbone already contains rich detectors for car,
# bicycle, and person. We only need to teach the detection HEAD to use those
# features for a single-class output. By freezing the backbone completely on every
# client, all clients share the SAME feature space, so the server can inject
# per-class head weights cleanly without any backbone interference.
# ==============================================================================

class HeadOnlyTrainer(DetectionTrainer):
    """
    Federated Learning trainer that freezes the entire backbone + neck and trains
    ONLY the Detect head (the last module in the model's Sequential stack).

    This is the correct approach for extreme non-IID FL where each client holds
    a disjoint subset of classes. The shared COCO-pretrained backbone guarantees
    a common feature space that the server-side stitch aggregation can exploit.
    """

    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_train_start", self._freeze_backbone_and_neck)

    def _freeze_backbone_and_neck(self, trainer):
        """
        Identifies the final Detect layer index dynamically (works for any YOLO
        variant: yolo11n, yolo26n, yolov8n, etc.) and freezes everything before it.
        """
        # In the training context, trainer.model IS the DetectionModel (nn.Module),
        # so trainer.model.model is the Sequential stack of layers — not .model.model.model.
        detect_idx = len(trainer.model.model) - 1
        detect_prefix = f'model.{detect_idx}.'

        frozen_count = 0
        unfrozen_count = 0
        unfrozen_names = []

        for name, param in trainer.model.named_parameters():
            if name.startswith(detect_prefix):
                param.requires_grad = True
                unfrozen_count += 1
                unfrozen_names.append(name)
            else:
                param.requires_grad = False
                frozen_count += 1

        print(f"\n[HeadOnly] Backbone+Neck FROZEN  : {frozen_count} parameter tensors")
        print(f"[HeadOnly] Detect Head TRAINABLE : {unfrozen_count} parameter tensors")
        print(f"[HeadOnly] Detect head layers: {', '.join(unfrozen_names[:6])}{'...' if len(unfrozen_names) > 6 else ''}\n")


# ==============================================================================
# ROOT CAUSE FIX #2: Dataset preparation — proper validation split
#
# Original bug: val YAML pointed to the training images directory ("images/train").
# This means the model sees val images during training → the local mAP@50 of 0.816
# is inflated by data leakage and early stopping fires too late (or not at all).
# The fix splits the collected images into 80% train / 20% val.
# ==============================================================================

def setup_local_dataset(target_class_name, max_images=None):
    model_name = "yolo26n.pt"
    if not os.path.exists(model_name):
        download(f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}")

    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')
    if not os.path.exists(base_dir):
        try:
            YOLO(model_name).train(data="VOC.yaml", epochs=0, imgsz=640)
        except Exception:
            pass

    # Resolve the labels/images directories (handle different VOC layout variants)
    for ldir, idir in [
        ('labels/train', 'images/train'),
        ('labels/train2012', 'images/train2012'),
        ('labels', 'images'),
    ]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir):
            break

    voc_classes = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

    original_id = next((k for k, v in voc_classes.items() if v.lower() == target_class_name.lower()), None)
    if original_id is None:
        raise ValueError(f"Class '{target_class_name}' not found in VOC class list.")

    client_id = f"client_{target_class_name}"
    out_dir = os.path.abspath(f"./{client_id}_data")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for split in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    # Collect all matching (label_file, filtered_lines) pairs first
    all_samples = []
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        filtered = []
        for line in lines:
            if not line.strip():
                continue
            try:
                if int(line.split()[0]) == original_id:
                    parts = line.split()
                    parts[0] = "0"   # remap to local class 0
                    filtered.append(" ".join(parts) + "\n")
            except ValueError:
                pass
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            src_img = os.path.join(images_dir, img_filename)
            if os.path.exists(src_img):
                all_samples.append((label_file, img_filename, filtered))
        if max_images is not None and len(all_samples) >= max_images:
            break

    if max_images is not None:
        all_samples = all_samples[:max_images]

    # 80/20 train/val split (deterministic)
    split_idx = int(len(all_samples) * 0.8)
    splits = {'train': all_samples[:split_idx], 'val': all_samples[split_idx:]}

    for split, samples in splits.items():
        for label_file, img_filename, filtered in samples:
            with open(os.path.join(out_dir, 'labels', split, label_file), 'w') as f:
                f.writelines(filtered)
            shutil.copy(
                os.path.join(images_dir, img_filename),
                os.path.join(out_dir, 'images', split, img_filename)
            )

    yaml_path = f"{client_id}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({
            "path": out_dir,
            "train": "images/train",
            "val": "images/val",   # FIX: separate validation split
            "nc": 1,
            "names": {0: target_class_name}
        }, f, sort_keys=False)

    print(f"[Dataset] '{target_class_name}': {len(splits['train'])} train, {len(splits['val'])} val images")
    return yaml_path, client_id


def ssh_transfer(client_id, weights_path):
    """Transfers weights and metadata to the server via SFTP."""
    print(f"\nConnecting to server {SERVER_IP} via SSH...")
    transport = paramiko.Transport((SERVER_IP, SSH_PORT))
    transport.connect(username=SSH_USER, password=SSH_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    remote_weights = f"{SERVER_UPLOAD_DIR}/{client_id}_weights.pt"
    print(f"Uploading {weights_path} to {remote_weights}...")
    sftp.put(weights_path, remote_weights)

    meta = {"client_id": client_id, "class_name": client_id.split('_', 1)[1]}
    with open("meta.json", "w") as f:
        json.dump(meta, f)

    remote_meta = f"{SERVER_UPLOAD_DIR}/{client_id}_meta.json"
    print(f"Uploading metadata to {remote_meta}...")
    sftp.put("meta.json", remote_meta)

    sftp.close()
    transport.close()
    print("Transfer complete! The server should process it shortly.")


# ==============================================================================
# ROOT CAUSE FIX #3: Use the global model as the starting point for local training.
#
# Original bug: Every client always started from yolo26n.pt (base COCO pretrained).
# This is fine for round 1, but in a real FL system after round 1 the global model
# accumulates multi-class head knowledge. Starting from the global model means the
# local client adapts its single-class head starting from a good multi-class
# initialisation, which speeds up convergence and improves aggregation quality.
#
# If no global model exists yet (first round), fall back to yolo26n.pt.
# ==============================================================================

def _get_starting_model():
    """Downloads the global model from the server to use as the FL starting point."""
    local_global = "./downloaded_global_model.pt"
    if os.path.exists(local_global):
        print(f"[FL] Using existing global model as starting point: {local_global}")
        return local_global

    print("[FL] Attempting to download global model from server...")
    try:
        transport = paramiko.Transport((SERVER_IP, SSH_PORT))
        transport.connect(username=SSH_USER, password=SSH_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(f"{SERVER_DOWNLOAD_DIR}/global_model.pt", local_global)
        sftp.close()
        transport.close()
        print(f"[FL] Downloaded global model. Will use it as the FL starting point.")
        return local_global
    except Exception as e:
        print(f"[FL] No global model on server yet (this is normal for round 1). Using base yolo26n.pt. ({e})")
        return "yolo26n.pt"


def train_and_send():
    """Full pipeline: extract dataset → train head only → transfer weights."""
    target_class = input("\nEnter the class to train on (e.g., 'person', 'car'): ").strip()

    limit_input = input("How many images? (number, or Enter for ALL): ").strip()
    max_images = int(limit_input) if limit_input.isdigit() else None

    epochs_input = input("Epochs? (Enter for default 50): ").strip()
    num_epochs = int(epochs_input) if epochs_input.isdigit() else 50

    yaml_path, client_id = setup_local_dataset(target_class, max_images=max_images)

    # Attempt to get the current global model; fall back to COCO pretrained
    starting_model = _get_starting_model()

    print(f"\n[FL] Starting HeadOnly training for '{client_id}' ({num_epochs} epochs)...")
    print(f"[FL] Starting weights: {starting_model}")
    print("[FL] Backbone will be FROZEN. Only the Detect head will be trained.\n")

    # ==================================================================
    # KEY HYPERPARAMETER NOTE:
    # With backbone frozen, convergence is fast and we only need to move
    # the detect head. lr0=0.01 is standard; warmup helps stabilise the
    # first few batches. No freeze= arg here — the callback owns all
    # freezing logic so there is no conflict.
    # ==================================================================
    args = dict(
        model=starting_model,
        data=yaml_path,
        epochs=num_epochs,
        imgsz=640,
        name=client_id,
        # NO freeze=N here — HeadOnlyTrainer callback handles all freezing
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        patience=15,   # early stopping: stop if no improvement for 15 epochs
        seed=42,
        deterministic=True,
    )

    trainer = HeadOnlyTrainer(overrides=args)
    trainer.train()

    best_weights = os.path.abspath(f"runs/detect/{client_id}/weights/best.pt")

    if os.path.exists(best_weights):
        print("\n✅ Training complete! Preparing to send weights...")
        try:
            ssh_transfer(client_id, best_weights)
        except Exception as e:
            print(f"\n❌ SSH Transfer failed: {e}")
            print(f"Weights saved locally at: {best_weights}")
    else:
        print(f"\n❌ Could not find weights at {best_weights}")


def send_existing_weights():
    """Transfer already-trained weights without re-training."""
    target_class = input("\nEnter the class name already trained (e.g., 'person', 'car'): ").strip()
    client_id = f"client_{target_class}"
    best_weights = os.path.abspath(f"runs/detect/{client_id}/weights/best.pt")

    if os.path.exists(best_weights):
        print(f"Found existing weights at: {best_weights}")
        try:
            ssh_transfer(client_id, best_weights)
        except Exception as e:
            print(f"\n❌ SSH Transfer failed: {e}")
    else:
        print(f"\n❌ Could not find weights at {best_weights}")


def download_and_inspect_global_model():
    """Connects to server, downloads global_model.pt, and prints its class map."""
    print(f"\nConnecting to server {SERVER_IP} via SSH...")
    try:
        transport = paramiko.Transport((SERVER_IP, SSH_PORT))
        transport.connect(username=SSH_USER, password=SSH_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        remote_global_model = f"{SERVER_DOWNLOAD_DIR}/global_model.pt"
        local_global_model = "./downloaded_global_model.pt"

        print(f"Downloading {remote_global_model}...")
        sftp.get(remote_global_model, local_global_model)
        sftp.close()
        transport.close()
        print("Download complete!\n")

        global_model = YOLO(local_global_model)
        classes = global_model.model.names
        nc = global_model.model.nc

        print(f"Total Global Classes: {nc}")
        for global_id, name in classes.items():
            print(f"  [ID {global_id}] → {name}")
        print(f"\nSaved locally as: {os.path.abspath(local_global_model)}")

    except FileNotFoundError:
        print("\n❌ Global model not found on server. Upload at least one client first.")
    except Exception as e:
        print(f"\n❌ SSH Download failed: {e}")


def run_inference():
    """Run inference with local or global model on a user-supplied image."""
    print("\n--- Run Inference ---")
    print("1. Downloaded Global Model (Combined Knowledge)")
    print("2. Locally Trained Model (Single Class)")
    print("3. Cancel")

    choice = input("Select (1-3): ").strip()

    if choice == '1':
        model_path = "./downloaded_global_model.pt"
        if not os.path.exists(model_path):
            print("❌ No global model found. Use Option 3 first.")
            return
    elif choice == '2':
        target_class = input("Enter the locally trained class name: ").strip()
        model_path = os.path.abspath(f"runs/detect/client_{target_class}/weights/best.pt")
        if not os.path.exists(model_path):
            print(f"❌ No local weights found at {model_path}")
            return
    elif choice == '3':
        return
    else:
        print("Invalid choice.")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    img_path = input("\nImage path (Enter for bus.jpg): ").strip() or "bus.jpg"
    if img_path == "bus.jpg" and not os.path.exists(img_path):
        download("https://ultralytics.com/images/bus.jpg")

    if not os.path.exists(img_path):
        print(f"❌ Image '{img_path}' not found.")
        return

    results = model.predict(source=img_path, save=True, show=False, conf=0.25)
    result = results[0]
    boxes = result.boxes
    print(f"\n✅ Found {len(boxes)} objects.")

    if len(boxes) > 0:
        detections = {}
        for box in boxes:
            cls_id = int(box.cls[0].item())
            name = model.names.get(cls_id, f"ID {cls_id}")
            detections[name] = detections.get(name, 0) + 1
        for name, count in detections.items():
            print(f"  - {name}: {count}")

    if result.save_dir:
        print(f"Annotated image: {os.path.join(result.save_dir, os.path.basename(img_path))}")


def validate_and_compare():
    """Build a joint validation set and compare Global vs Local model metrics."""
    global_model_path = "./downloaded_global_model.pt"
    if not os.path.exists(global_model_path):
        print("❌ No global model found. Use Option 3 first.")
        return

    global_model = YOLO(global_model_path)
    global_names = global_model.names
    print(f"Global Model Classes: {global_names}")

    voc_classes = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

    global_to_voc = {
        g_id: v_id
        for g_id, g_name in global_names.items()
        for v_id, v_name in voc_classes.items()
        if g_name.lower() == v_name.lower()
    }

    datasets_root = settings['datasets_dir']
    base_dir = os.path.join(datasets_root, 'VOC')

    for ldir, idir in [
        ('labels/val2012', 'images/val2012'),
        ('labels/val', 'images/val'),
        ('labels/train2012', 'images/train2012'),
    ]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir):
            break

    val_dir = os.path.abspath("./global_val_data")
    out_img_dir = os.path.join(val_dir, "images", "val")
    out_lbl_dir = os.path.join(val_dir, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    images_copied = 0
    for label_file in os.listdir(labels_dir)[:500]:
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        filtered = []
        for line in lines:
            if not line.strip():
                continue
            try:
                orig_id = int(line.split()[0])
                for g_id, v_id in global_to_voc.items():
                    if orig_id == v_id:
                        parts = line.split()
                        parts[0] = str(g_id)
                        filtered.append(" ".join(parts) + "\n")
            except ValueError:
                pass
        if filtered:
            img_filename = label_file.replace('.txt', '.jpg')
            src_img = os.path.join(images_dir, img_filename)
            if os.path.exists(src_img):
                with open(os.path.join(out_lbl_dir, label_file), 'w') as f:
                    f.writelines(filtered)
                shutil.copy(src_img, os.path.join(out_img_dir, img_filename))
                images_copied += 1

    yaml_path = "global_val.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({
            "path": val_dir, "train": "images/val", "val": "images/val",
            "nc": len(global_names), "names": global_names
        }, f, sort_keys=False)

    print(f"Validation set: {images_copied} images")

    print("\n--- Evaluating GLOBAL Model ---")
    global_metrics = global_model.val(data=yaml_path, split='val', verbose=False)

    local_classes_input = input("\nLocal classes to compare (comma-separated, e.g. 'person,car'): ").strip()
    local_classes = [c.strip() for c in local_classes_input.split(',')] if local_classes_input else []

    local_results = {}
    for local_class in local_classes:
        client_id = f"client_{local_class}"
        local_model_path = os.path.abspath(f"runs/detect/{client_id}/weights/best.pt")
        local_yaml = f"{client_id}.yaml"
        if os.path.exists(local_model_path) and os.path.exists(local_yaml):
            local_model = YOLO(local_model_path)
            local_metrics = local_model.val(data=local_yaml, split='val', verbose=False)
            local_results[local_class] = local_metrics.box.map50
        else:
            print(f"⚠️ Local model or YAML for '{local_class}' not found. Skipping.")

    print("\n" + "=" * 50)
    print("🏆 FEDERATED LEARNING METRICS COMPARISON 🏆")
    print("=" * 50)
    for local_class, map50 in local_results.items():
        print(f"\n[ LOCAL MODEL ('{local_class}' only) ]")
        print(f"  - Overall mAP@50:      {map50:.3f}")
        print(f"  - Classes Known:       1 ({local_class})")
    print(f"\n[ GLOBAL MODEL (Merged Knowledge) ]")
    print(f"  - Overall mAP@50:      {global_metrics.box.map50:.3f}")
    print(f"  - Classes Known:       {len(global_names)}")
    print("\n[ GLOBAL MODEL Class Breakdown ]")
    for i, class_name in global_names.items():
        try:
            idx = global_metrics.ap_class_index.tolist().index(i)
            print(f"  - {class_name:<15}: mAP@50 = {global_metrics.box.maps[idx]:.3f}")
        except ValueError:
            print(f"  - {class_name:<15}: No instances found in validation sample.")
    print("=" * 50)


if __name__ == "__main__":
    while True:
        print("\n=== DAFYOLO Federated Learning Client (v2) ===")
        print("1. Train a new model and send weights to server")
        print("2. Send EXISTING weights to server (skip training)")
        print("3. Download Global Model & Inspect Classes")
        print("4. Run Inference (Local or Global Model)")
        print("5. Validate and Compare Metrics 📊")
        print("6. Exit")

        choice = input("Select an option (1-6): ").strip()
        if choice == '1':
            train_and_send()
        elif choice == '2':
            send_existing_weights()
        elif choice == '3':
            download_and_inspect_global_model()
        elif choice == '4':
            run_inference()
        elif choice == '5':
            validate_and_compare()
        elif choice == '6':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-6.")
