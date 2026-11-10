#!/usr/bin/env python3
"""Check if server_updated.py is running on the server"""

import os
import paramiko
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

from client_updated import SERVER_IP, SSH_PORT, SSH_USER, SSH_PASSWORD
# This script SSHes into the remote server to check it, so the path it needs
# is the *remote* box's profile, not local-OS auto-detection (which would
# pick macos_laptop when run from a Mac even though the server is Linux).
# Override with DAFYOLO_SERVER_PROFILE if the server itself isn't the
# linux_gpu_box profile below.
import os
from server_paths import load_profile
SERVER_LOG = load_profile(os.getenv("DAFYOLO_SERVER_PROFILE", "linux_gpu_box"))["server_log"]

print("🔍 Checking if server_updated.py is running...")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    hostname=SERVER_IP, 
    port=SSH_PORT, 
    username=SSH_USER, 
    password=SSH_PASSWORD,
    look_for_keys=False, 
    allow_agent=False
)

# Check for server_updated.py process
stdin, stdout, stderr = ssh.exec_command("ps aux | grep server_updated.py | grep -v grep")
output = stdout.read().decode().strip()

if output:
    print("✅ server_updated.py IS RUNNING on the server!")
    print("\nProcess info:")
    print(output)
else:
    print("❌ server_updated.py is NOT RUNNING on the server!")
    print("\n⚠️  You need to start it with:")
    print("   python server_updated.py")
    print("\n   Or in background:")
    print(f"   nohup python server_updated.py > {SERVER_LOG} 2>&1 &")

# Check the server log if available
print("\n📋 Recent server activity:")
stdin, stdout, stderr = ssh.exec_command(f"tail -20 {SERVER_LOG} 2>/dev/null || echo 'No log file yet'")
log_output = stdout.read().decode()
print(log_output)

ssh.close()
