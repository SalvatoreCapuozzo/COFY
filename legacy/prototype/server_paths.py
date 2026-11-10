"""Loads server_paths.yaml so client.py / client_v2.py / server.py /
server_v2.py / check_server_running.py / diagnose_connection.py get the
right filesystem paths on whichever machine they're launched from, instead
of a path hardcoded for one specific remote Linux box.

Usage:
    from server_paths import UPLOAD_DIR, GLOBAL_MODEL_DIR, PROCESSED_DIR, SERVER_LOG

Override the auto-detected profile with:
    export DAFYOLO_SERVER_PROFILE=linux_gpu_box   # or macos_laptop

Note on macos_laptop's relative paths: UPLOAD_DIR/GLOBAL_MODEL_DIR/PROCESSED_DIR
are anchored to this file's own directory (see _anchor() below), which is
correct for server.py/server_v2.py's direct filesystem use. client.py/
client_v2.py instead push to these paths over SFTP -- that only lines up with
the anchored local path if the SSH target is this same machine/directory
(e.g. SSH into localhost for an all-on-one-Mac setup). client_updated.py's
CONNECTION_MODE="local" bypasses SFTP entirely for that case and is the more
robust choice for solo-machine use.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_CONFIG_PATH = _HERE / "server_paths.yaml"


def _default_profile() -> str:
    return "macos_laptop" if platform.system() == "Darwin" else "linux_gpu_box"


def _anchor(path: str) -> str:
    """A relative path (e.g. macos_laptop's "server_node/uploads") is meant
    relative to THIS file's directory, not whatever the caller's CWD happens
    to be -- resolve it here so scripts work regardless of where they're
    launched from. Absolute paths (e.g. linux_gpu_box's /datadrive/...)
    pass through unchanged."""
    return path if os.path.isabs(path) else str(_HERE / path)


def load_profile(name: str | None = None) -> dict:
    name = name or os.getenv("DAFYOLO_SERVER_PROFILE") or _default_profile()
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    if name not in data:
        raise ValueError(
            f"Unknown DAFYOLO_SERVER_PROFILE '{name}' -- expected one of "
            f"{list(data)} (edit {_CONFIG_PATH.name} to add a new machine)."
        )
    return data[name]


_PROFILE_NAME = os.getenv("DAFYOLO_SERVER_PROFILE") or _default_profile()
_PATHS = load_profile(_PROFILE_NAME)

UPLOAD_DIR = _anchor(_PATHS["upload_dir"])
GLOBAL_MODEL_DIR = _anchor(_PATHS["global_model_dir"])
PROCESSED_DIR = _anchor(_PATHS["processed_dir"])
SERVER_LOG = _anchor(_PATHS["server_log"])
KFM_DATASET_SOURCE = _PATHS["kfm_dataset_source"]  # always absolute in both profiles
