"""この PC の計算装置（NVIDIA の GPU / Mac の GPU / CPU）を調べる。"""

import json
import subprocess
import threading

from . import settings

_info: dict = {"device": None}
_ready = threading.Event()

_CODE = """
import json, torch
info = {"device": "cpu", "name": None, "memory_gb": None}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    info = {"device": "cuda", "name": props.name, "memory_gb": round(props.total_memory / 1024**3, 1),
            "bf16": bool(torch.cuda.is_bf16_supported())}
elif torch.backends.mps.is_available():
    info = {"device": "mps", "name": "Apple Silicon", "memory_gb": None}
print("@@device " + json.dumps(info))
"""


def detect() -> None:
    """PyTorch が使える計算装置を調べる（サーバーの起動時に、別のスレッドで1回だけ）。"""
    try:
        result = subprocess.run(
            [settings.PYTHON, "-c", _CODE], capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        for line in result.stdout.splitlines():
            if line.startswith("@@device "):
                _info.update(json.loads(line[len("@@device "):]))
                break
        else:
            _info["device"] = "unknown"
    except (OSError, subprocess.SubprocessError, ValueError):
        _info["device"] = "unknown"
    finally:
        _ready.set()


def info(wait: float = 0) -> dict:
    if wait:
        _ready.wait(wait)
    return dict(_info)


def device(wait: float = 0) -> str:
    return info(wait).get("device") or "checking"


def recommended_batch_size() -> int:
    """学習のバッチサイズのおすすめ（GPU のメモリに合わせる）。"""
    current = info()
    if current.get("device") == "cuda":
        memory = current.get("memory_gb") or 0
        if memory >= 20:
            return 12
        if memory >= 10:
            return 8
        return 4
    return 4
