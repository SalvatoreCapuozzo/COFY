import os
import time
import json
import paramiko
import yaml
import glob
import shutil
import subprocess
from datetime import datetime
from ultralytics import settings
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

import client_updated

from client_updated import (
    SERVER_IP, SSH_PORT, SSH_USER, SSH_PASSWORD, SERVER_UPLOAD_DIR,
    setup_local_dataset, upload_to_server, download_global_model,
    LOCAL_MODELS_DIR, DOWNLOADED_MODELS_DIR
)

if not SERVER_IP or not SSH_USER or not SSH_PASSWORD:
    raise ValueError(f"\n❌ CRITICAL: Missing credentials!\nCheck your .env file inside {script_dir}")

#STRATEGIES = ['fedhead', 'stitch', 'ties', 'fedavg', 'yoloinc', 'yolopa_neckhead']
STRATEGIES = ['fedhead', 'fedavg', 'fedprox', 'yoloinc']
MAX_IMAGES = 800 #2000
EPOCHS = 10
RESULTS_FILE = "advanced_experiment_results_260505_1800.txt"

SCENARIOS = {
    "EXTREME_NON_IID": [
        ['person', 'car', 'bicycle', 'motorbike'],
        ['aeroplane', 'bus', 'train', 'boat'],
        ['bird', 'cat', 'dog', 'horse'],
        ['sheep', 'cow', 'bottle', 'chair'],
        ['sofa', 'pottedplant', 'diningtable', 'tvmonitor']
    ],
    "INTERSECTED": [
        ['person', 'car', 'bicycle', 'dog'],
        ['car', 'bus', 'train', 'cat'],
        ['bicycle', 'motorbike', 'train', 'horse'],
        ['person', 'bus', 'aeroplane', 'cow'],
        ['dog', 'cat', 'horse', 'cow']
    ]
}

def trigger_server_reset_headless(strategy):
    print(f"\n📡 Sending command to server: RESET and switch to {strategy.upper()}")
    cmd_data = {"command": "reset", "strategy": strategy, "timestamp": str(datetime.now())}
    try:
        if client_updated.CONNECTION_MODE == "local":
            upload_dir = os.path.join(client_updated.LOCAL_SERVER_BASE_DIR, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            with open(os.path.join(upload_dir, "CMD_RESET.json"), "w") as f:
                json.dump(cmd_data, f)
        else:
            with open("CMD_RESET.json", "w") as f:
                json.dump(cmd_data, f)
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=SERVER_IP, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, look_for_keys=False, allow_agent=False, timeout=15)
            sftp = ssh.open_sftp()
            sftp.put("CMD_RESET.json", f"{SERVER_UPLOAD_DIR}/CMD_RESET.json")
            sftp.close(); ssh.close(); os.remove("CMD_RESET.json")
    except Exception as e: 
        print(f"❌ Failed to send reset command: {e}")

# =====================================================================
# SUBPROCESS WORKER SCRIPTS (Written dynamically to disk)
# =====================================================================
# By writing these out and executing them via `subprocess.run`, we 
# ensure that PyTorch is completely destroyed and purged from RAM 
# by the operating system at the end of every task.

def run_isolated_training(starting_model, yaml_path, client_run_name, strategy):
    """Generates a temporary python script to run training in an isolated process."""
    script_content = f"""
from client_updated import SmartFLTrainer
import os

SmartFLTrainer.is_round_1 = "global_model" not in "{starting_model}"
SmartFLTrainer.strategy = "{strategy}"

trainer = SmartFLTrainer(overrides=dict(
    model="{starting_model}", data="{yaml_path}", epochs={EPOCHS}, imgsz=640,
    name="{client_run_name}", lr0=0.01, patience=15, verbose=False, 
    batch=16, workers=0, plots=False 
))
trainer.train()
"""
    with open("temp_train.py", "w") as f: f.write(script_content)
    # Execute the script and wait for it to finish
    subprocess.run(["python", "temp_train.py"], check=True)
    os.remove("temp_train.py")

def run_isolated_validation(model_path, yaml_path):
    """Generates a temporary python script to run validation and return the map50 score."""
    script_content = f"""
from ultralytics import YOLO
import json

model = YOLO("{model_path}")
metrics = model.val(data="{yaml_path}", split='val', verbose=False, plots=False, workers=0)

output = {{
    "map50": metrics.box.map50,
    "names": model.names,
    "maps": metrics.box.maps.tolist() if hasattr(metrics.box, 'maps') else [],
    "ap_index": metrics.ap_class_index.tolist() if hasattr(metrics, 'ap_class_index') else []
}}

with open("temp_val_results.json", "w") as f: json.dump(output, f)
"""
    with open("temp_val.py", "w") as f: f.write(script_content)
    subprocess.run(["python", "temp_val.py"], check=True)
    os.remove("temp_val.py")
    
    with open("temp_val_results.json", "r") as f: results = json.load(f)
    os.remove("temp_val_results.json")
    return results

# =====================================================================

def train_and_send_headless(target_class_names, strategy, node_name):
    print(f"\n🚀 [AUTO-TRAIN] Preparing dataset for '{node_name}' ({target_class_names}) | Strat: {strategy.upper()}")
    yaml_path, client_id, num_samples = setup_local_dataset(target_class_names, node_name, MAX_IMAGES)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    client_run_name = f"{client_id}_{strategy}_{ts}"

    if strategy in ['ties', 'fedavg']:
        starting_model = "yolo26n.pt"
    else:
        global_model = download_global_model(strategy)
        starting_model = global_model if global_model else "yolo26n.pt"

    # Execute training in an isolated OS process to guarantee 0 memory leaks
    run_isolated_training(starting_model, yaml_path, client_run_name, strategy)

    best_weights = os.path.abspath(f"runs/detect/{client_run_name}/weights/best.pt")
    if os.path.exists(best_weights):
        upload_to_server(client_run_name, best_weights, target_class_names, num_samples)

def validate_and_compare_headless(strategy, scenario_name, log_file, clients):
    print(f"\n📊 Running Evaluation for Strategy: {strategy.upper()} | Scenario: {scenario_name}")
    global_model_path = download_global_model(strategy)
    if not global_model_path: return

    val_dir = os.path.abspath(f"./global_val_data_auto_{scenario_name}")
    out_img_dir = os.path.join(val_dir, "images", "val")
    out_lbl_dir = os.path.join(val_dir, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True); os.makedirs(out_lbl_dir, exist_ok=True)

    voc_classes = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
    
    # We must load the global model briefly to get its class names for the dataset builder
    from ultralytics import YOLO
    temp_model = YOLO(global_model_path)
    global_names = temp_model.names
    del temp_model

    global_to_voc = {g_id: v_id for g_id, g_name in global_names.items() for v_id, v_name in voc_classes.items() if g_name.lower() == v_name.lower()}

    base_dir = os.path.join(settings['datasets_dir'], 'VOC')
    for ldir, idir in [('labels/val2012', 'images/val2012'), ('labels/val', 'images/val')]:
        labels_dir = os.path.join(base_dir, ldir)
        images_dir = os.path.join(base_dir, idir)
        if os.path.exists(labels_dir): break

    if os.path.exists(labels_dir):
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
                            parts = line.split(); parts[0] = str(g_id)
                            filtered.append(" ".join(parts) + "\n")
                except ValueError: pass
            if filtered:
                img_filename = label_file.replace('.txt', '.jpg')
                if os.path.exists(os.path.join(images_dir, img_filename)):
                    with open(os.path.join(out_lbl_dir, label_file), 'w') as f: f.writelines(filtered)
                    shutil.copy(os.path.join(images_dir, img_filename), os.path.join(out_img_dir, img_filename))

    yaml_path = f"global_val_{scenario_name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({"path": val_dir, "train": "images/val", "val": "images/val", "nc": len(global_names), "names": global_names}, f, sort_keys=False)

    # Isolated Validation execution
    print("⚙️ Evaluating Global Model...")
    global_metrics = run_isolated_validation(global_model_path, yaml_path)

    local_results = {}
    for node_idx, class_names in enumerate(clients):
        node_name = f"node_{node_idx}"
        pattern = f"{LOCAL_MODELS_DIR}/client_{node_name}_{strategy}_*/weights/best.pt"
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches:
            print(f"⚙️ Evaluating Local Model: '{node_name}'...")
            local_metrics = run_isolated_validation(matches[0], f"client_{node_name}.yaml")
            local_results[node_name] = local_metrics['map50']

    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"🏆 METRICS | STRATEGY: {strategy.upper()} | SCENARIO: {scenario_name}\n")
        f.write("=" * 50 + "\n")
        for node_name, map50 in local_results.items(): f.write(f"\n[ LOCAL {node_name} ]: mAP@50 = {map50:.3f}\n")
        f.write(f"\n[ GLOBAL MODEL ]: Overall mAP@50 = {global_metrics['map50']:.3f} | Total Classes: {len(global_names)}\n\n")
        for i, class_name in global_names.items():
            try:
                idx = global_metrics['ap_index'].index(int(i))
                f.write(f"  - {class_name:<15}: mAP@50 = {global_metrics['maps'][idx]:.3f}\n")
            except ValueError: f.write(f"  - {class_name:<15}: No instances found.\n")
        f.write("=" * 50 + "\n")

if __name__ == "__main__":
    print("\n=== DAFYOLO Automated Experiments ===")
    print("How is your server running?")
    print("  [1] Remote Server")
    print("  [2] Local Server")
    
    conn_choice = input("Select connection mode (1-2): ").strip()
    if conn_choice == '2':
        client_updated.CONNECTION_MODE = 'local'
        print(f"\n📂 Default local path: {client_updated.LOCAL_SERVER_BASE_DIR}")
        local_path = input("Press Enter to use default, or type custom path: ").strip()
        if local_path: client_updated.LOCAL_SERVER_BASE_DIR = local_path
    else:
        client_updated.CONNECTION_MODE = 'remote'

    with open(RESULTS_FILE, "w") as f:
        f.write(f"DAFYOLO ADVANCED EXPERIMENTS\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for scenario_name, clients in SCENARIOS.items():
        print(f"\n\n{'='*80}\n🚀 STARTING SCENARIO: {scenario_name}\n{'='*80}")
        for strategy in STRATEGIES:
            trigger_server_reset_headless(strategy)
            time.sleep(10) 
            for node_idx, class_names in enumerate(clients):
                train_and_send_headless(class_names, strategy, f"node_{node_idx}")
                time.sleep(5)
            time.sleep(30) 
            validate_and_compare_headless(strategy, scenario_name, RESULTS_FILE, clients)

    print(f"\n🎉 ALL EXPERIMENTS COMPLETE! Check {RESULTS_FILE}.")