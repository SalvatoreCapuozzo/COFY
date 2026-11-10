#!/usr/bin/env python3
"""
Diagnostic script to check if run_experiments.py can communicate with server_updated.py
"""

import os
import sys
import json
import paramiko
from datetime import datetime
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

from client_updated import SERVER_IP, SSH_PORT, SSH_USER, SSH_PASSWORD, SERVER_UPLOAD_DIR
# Diagnosing a *remote* server over SSH -- see check_server_running.py's note
# on why this defaults to linux_gpu_box rather than local-OS auto-detection.
from server_paths import load_profile
_remote = load_profile(os.getenv("DAFYOLO_SERVER_PROFILE", "linux_gpu_box"))
SERVER_GLOBAL_MODEL_DIR = _remote["global_model_dir"]

print("=" * 80)
print("🔍 DAFYOLO Communication Diagnostic")
print("=" * 80)

# Check 1: Credentials
print("\n1️⃣  Checking Credentials...")
print(f"   SERVER_IP: {SERVER_IP if SERVER_IP else '❌ MISSING'}")
print(f"   SSH_USER: {SSH_USER if SSH_USER else '❌ MISSING'}")
print(f"   SSH_PORT: {SSH_PORT if SSH_PORT else '❌ MISSING'}")
print(f"   SSH_PASSWORD: {'***' if SSH_PASSWORD else '❌ MISSING'}")

if not SERVER_IP or not SSH_USER or not SSH_PASSWORD:
    print("\n❌ CRITICAL: Missing credentials!")
    sys.exit(1)

# Check 2: SSH Connection
print("\n2️⃣  Testing SSH Connection...")
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=SERVER_IP, 
        port=SSH_PORT, 
        username=SSH_USER, 
        password=SSH_PASSWORD,
        look_for_keys=False, 
        allow_agent=False, 
        timeout=10
    )
    print(f"   ✅ Successfully connected to {SERVER_IP}:{SSH_PORT}")
    
    # Check 3: Server upload directory exists
    print("\n3️⃣  Checking Server Directories...")
    sftp = ssh.open_sftp()
    try:
        sftp.stat(SERVER_UPLOAD_DIR)
        print(f"   ✅ Upload directory exists: {SERVER_UPLOAD_DIR}")
    except FileNotFoundError:
        print(f"   ❌ Upload directory NOT found: {SERVER_UPLOAD_DIR}")
        print(f"   ⚠️  You need to create it on the server!")
    
    # Check 4: List files in upload directory
    print("\n4️⃣  Files in Server Upload Directory...")
    try:
        files = sftp.listdir(SERVER_UPLOAD_DIR)
        if files:
            print(f"   📁 Found {len(files)} files:")
            for f in files:
                print(f"      - {f}")
        else:
            print(f"   📁 Directory is empty (waiting for uploads)")
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
    
    # Check 5: Check server_info.json
    print("\n5️⃣  Checking Server Info...")
    server_info_path = f"{SERVER_GLOBAL_MODEL_DIR}/server_info.json"
    try:
        with sftp.file(server_info_path, 'r') as f:
            server_info = json.load(f)
            print(f"   ✅ Server info found:")
            print(f"      Strategy: {server_info.get('strategy', 'UNKNOWN')}")
            print(f"      Boot time: {server_info.get('boot_time', 'UNKNOWN')}")
    except Exception as e:
        print(f"   ❌ Server info not found: {e}")
        print(f"      ⚠️  Server might not be running!")
    
    # Check 6: Check if any global model exists
    print("\n6️⃣  Checking Global Model...")
    try:
        sftp.stat(f"{SERVER_GLOBAL_MODEL_DIR}/global_model.pt")
        print(f"   ✅ Global model exists")
    except FileNotFoundError:
        print(f"   ℹ️  No global model yet (expected on first run)")
    
    sftp.close()
    ssh.close()
    
except Exception as e:
    print(f"   ❌ SSH Connection Failed: {e}")
    print(f"\n⚠️  Cannot reach server! Check:")
    print(f"   1. Is the server machine running?")
    print(f"   2. Are credentials in .env correct?")
    print(f"   3. Is network connectivity working?")
    sys.exit(1)

# Check 7: Local configuration
print("\n7️⃣  Local Configuration...")
print(f"   LOCAL_MODELS_DIR: runs/detect")
print(f"   DOWNLOADED_MODELS_DIR: global_models")
print(f"   SERVER_UPLOAD_DIR (used by client): {SERVER_UPLOAD_DIR}")

# Check 8: Server process check
print("\n8️⃣  Server Process Status...")
print(f"   ℹ️  To check if server_updated.py is running on the server, SSH in and run:")
print(f"      ps aux | grep server_updated.py")

print("\n" + "=" * 80)
print("✅ Diagnostics Complete!")
print("=" * 80)

print("\n📋 WHAT TO CHECK:")
print("1. Is server_updated.py running on the server? (check with ps aux)")
print("2. Are files appearing in the upload directory? (check with ls)")
print("3. Are credentials correct in .env?")
print("4. Is there network connectivity between client and server?")
print("5. Is the upload directory readable/writable?")
