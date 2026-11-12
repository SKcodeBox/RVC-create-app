"""テストの共通設定。本物のデータ（rvc-server/data）を触らないよう、一時フォルダを使う。"""

import os
import sys
import tempfile
from pathlib import Path

# rvc_api を読み込む前に、保存先を一時フォルダにする
_DATA = Path(tempfile.mkdtemp(prefix="rvc-studio-test-"))
os.environ["RVC_DATA_DIR"] = str(_DATA)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

from rvc_api import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_data_dir():
    """テストごとに保存先を空にする。"""
    import shutil

    for child in list(_DATA.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    settings.ensure_dirs()
    yield


@pytest.fixture
def data_dir() -> Path:
    return _DATA


def write_tone(path: Path, seconds: float = 1.0, rate: int = 44100, channels: int = 1, freq: float = 220.0, level: float = 0.3) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    tone = (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, rate, subtype="PCM_16")
    return path


@pytest.fixture
def tone():
    return write_tone


def write_checkpoint(path: Path, iteration: int) -> Path:
    """PyTorch のチェックポイントと同じ形（zip の中に data.pkl）の、小さな偽物を書く。"""
    import pickle
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"iteration": iteration, "model": {"w": [1.0] * 100}}, protocol=2))
        archive.writestr("archive/data/0", b"\0" * 4000)
    return path
