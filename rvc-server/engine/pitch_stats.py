"""音声の声の高さ（ピッチ）を調べる。変換のときに「声の高さを自動で合わせる」ために使う。

Applio の仮想環境の Python で、Applio のフォルダを作業フォルダにして動かす（PYTHONPATH に Applio）。
使い方: python pitch_stats.py AUDIO
結果は「@@pitch {JSON}」の1行で出力する（median・p10・p90 は Hz、voiced_seconds は声のある秒数）。
"""

import json
import os
import sys

import numpy as np
import torch

from rvc.lib.predictors.f0 import RMVPE
from rvc.lib.utils import load_audio


def pick_device() -> str:
    requested = os.environ.get("RVC_EVAL_DEVICE")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stats(f0: np.ndarray) -> dict:
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        return {"median": None, "p10": None, "p90": None, "voiced_seconds": round(len(voiced) / 100, 1)}
    return {
        "median": round(float(np.median(voiced)), 2),
        "p10": round(float(np.percentile(voiced, 10)), 2),
        "p90": round(float(np.percentile(voiced, 90)), 2),
        "voiced_seconds": round(len(voiced) / 100, 1),
    }


def main() -> int:
    audio = load_audio(sys.argv[1], 16000)
    model = RMVPE(device=pick_device(), sample_rate=16000, hop_size=160)
    # 長い曲でもメモリが足りなくならないよう、1分ごとに区切って調べる
    step = 16000 * 60
    pieces = [model.get_f0(audio[start : start + step]) for start in range(0, len(audio), step)]
    f0 = np.concatenate(pieces) if pieces else np.zeros(0)
    print("@@pitch " + json.dumps(stats(np.asarray(f0, dtype=np.float64))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
