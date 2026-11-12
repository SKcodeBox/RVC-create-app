"""RVC のモデルファイル（.pth）の情報を調べる。取り込むときに、正しいモデルか確かめるためにも使う。

使い方: python model_info.py MODEL.pth
結果は「@@model {JSON}」の1行で出力する。
"""

import json
import sys

import torch


def main() -> int:
    try:
        checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
        config = list(checkpoint["config"])
        info = {
            "ok": True,
            "sample_rate": int(config[-1]),
            "version": checkpoint.get("version", "v1"),
            "f0": bool(checkpoint.get("f0", 1)),
            "embedder": checkpoint.get("embedder_model") or None,
            "vocoder": checkpoint.get("vocoder", "HiFi-GAN"),
            "epoch": checkpoint.get("epoch"),
            "speakers": int(checkpoint["weight"]["emb_g.weight"].shape[0]),
        }
    except Exception as error:  # 壊れたファイルや、RVC のモデルではないファイル
        info = {"ok": False, "error": str(error)[:300]}
    print("@@model " + json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
