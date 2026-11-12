"""学習の途中で保存したモデルを評価する（RVC Studio のサーバーが、学習と並行して実行する）。

1. 学習に使っていない音声（検証用に取り分けた分）を、そのモデルでどれだけ再現できるかを測る。
   値（メルスペクトログラムの誤差）が小さいほど、初めて聞く言葉や音程にも強い ＝ 変換の品質が良い。
   学習しすぎる（過学習）と、学習用の音声には合っていくのに、この値は悪くなっていく。
2. 決まった話し声（Applio 付属の確認用音声）をそのモデルの声にした「試し聴き用の音声」を作る。
   声の高さは、学習用の音声の高さに合わせる。

Applio の仮想環境の Python で、Applio のフォルダを作業フォルダにして動かす（PYTHONPATH に Applio）。

使い方:
  python evaluate.py --model-dir DIR --weight FILE.pth --epoch N [--sample OUT.flac] [--target-f0 HZ]
結果は「@@eval {JSON}」の1行で出力する。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.train.mel_processing import mel_spectrogram_torch

# 学習のときの mel 誤差と同じ倍率（Applio の設定 c_mel）にして、学習中の値と見比べやすくする
C_MEL = 45.0
# 試し聴き用の音声の長さ（秒）
SAMPLE_SECONDS = 12.0
# 1つの音声の最大フレーム数（学習のときと同じ。約9秒）
MAX_FRAMES = 900


def pick_device() -> torch.device:
    requested = os.environ.get("RVC_EVAL_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator(weight: Path, device: torch.device):
    checkpoint = torch.load(str(weight), map_location="cpu", weights_only=True)
    config = list(checkpoint["config"])
    config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    version = checkpoint.get("version", "v1")
    net_g = Synthesizer(
        *config,
        use_f0=checkpoint.get("f0", 1),
        text_enc_hidden_dim=768 if version == "v2" else 256,
        vocoder=checkpoint.get("vocoder", "HiFi-GAN"),
    )
    del net_g.enc_q
    net_g.load_state_dict(checkpoint["weight"], strict=False)
    net_g.eval().to(device)
    return net_g, int(config[-1])


def coarse_f0(f0: np.ndarray) -> np.ndarray:
    """声の高さ（Hz）を、モデルに渡す 1〜255 の段階に直す（Applio と同じ計算）。"""
    mel_min = 1127 * np.log(1 + 50.0 / 700)
    mel_max = 1127 * np.log(1 + 1100.0 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel = np.where(mel > 0, (mel - mel_min) * 254 / (mel_max - mel_min) + 1, mel)
    mel = np.clip(mel, 1, 255)
    return np.rint(mel).astype(np.int64)


def run_model(net_g, device, phone: np.ndarray, pitch: np.ndarray, pitchf: np.ndarray, seed: int):
    """特徴量（50フレーム/秒）と声の高さ（100フレーム/秒）から音声を作る。"""
    phone = np.repeat(phone, 2, axis=0)
    frames = min(len(phone), len(pitch), len(pitchf))
    phone, pitch, pitchf = phone[:frames], pitch[:frames], pitchf[:frames]
    torch.manual_seed(seed)
    with torch.no_grad():
        audio, *_ = net_g.infer(
            torch.from_numpy(phone).float().unsqueeze(0).to(device),
            torch.LongTensor([frames]).to(device),
            torch.from_numpy(pitch).long().unsqueeze(0).to(device),
            torch.from_numpy(pitchf).float().unsqueeze(0).to(device),
            torch.LongTensor([0]).to(device),
        )
    return audio[0, 0].float().cpu().numpy(), frames


def validation_loss(net_g, device, model_dir: Path, names: list[str], data_config: dict) -> tuple[float | None, int]:
    losses = []
    for number, name in enumerate(names):
        wav_path = model_dir / "sliced_audios" / f"{name}.wav"
        feature_path = model_dir / "extracted" / f"{name}.npy"
        pitch_path = model_dir / "f0" / f"{name}.wav.npy"
        pitchf_path = model_dir / "f0_voiced" / f"{name}.wav.npy"
        if not all(p.is_file() for p in (wav_path, feature_path, pitch_path, pitchf_path)):
            continue
        phone = np.load(feature_path)
        pitch = np.load(pitch_path)
        pitchf = np.load(pitchf_path)
        limit = MAX_FRAMES // 2
        generated, frames = run_model(
            net_g, device, phone[:limit], pitch[:MAX_FRAMES], pitchf[:MAX_FRAMES], seed=1234 + number
        )
        reference, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if reference.ndim > 1:
            reference = reference.mean(axis=1)
        length = min(len(generated), len(reference), frames * data_config["hop_length"])
        if length < data_config["win_length"]:
            continue
        pair = torch.from_numpy(np.stack([reference[:length], generated[:length]])).float()
        mel = mel_spectrogram_torch(
            pair,
            data_config["filter_length"],
            data_config["n_mel_channels"],
            data_config["sample_rate"],
            data_config["hop_length"],
            data_config["win_length"],
            data_config["mel_fmin"],
            data_config["mel_fmax"],
        )
        losses.append(float(torch.mean(torch.abs(mel[0] - mel[1]))) * C_MEL)
    if not losses:
        return None, 0
    return float(np.mean(losses)), len(losses)


def reference_features(model_dir: Path, logs_dir: Path, embedder: str) -> np.ndarray | None:
    """確認用音声の特徴量（Applio 付属のものがなければ、その場で計算して保存しておく）。"""
    bundled = logs_dir / "reference" / embedder / "feats.npy"
    if bundled.is_file():
        return np.load(bundled)
    cached = model_dir / "studio" / f"reference_{embedder}.npy"
    if cached.is_file():
        return np.load(cached)
    try:
        import librosa
        from rvc.lib.utils import load_embedding

        audio, _ = librosa.load(str(logs_dir / "reference" / "reference.wav"), sr=16000)
        audio = audio[: (len(audio) // 320) * 320]
        model = load_embedding(embedder)
        feats = torch.from_numpy(audio).float().unsqueeze(0)
        feats = torch.nn.functional.pad(feats, (40, 40), mode="reflect")
        with torch.no_grad():
            output = model(feats)["last_hidden_state"].squeeze(0).float().cpu().numpy()
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, output)
        return output
    except Exception as error:  # 試し聴き用の音声は、作れなくても評価は続ける
        print(f"確認用音声の特徴量を作れませんでした: {error}", file=sys.stderr)
        return None


def make_sample(net_g, device, sample_rate: int, model_dir: Path, embedder: str, target_f0: float | None, out: Path) -> int:
    """確認用の話し声を、このモデルの声にした音声を書き出す。使った声の高さの変化（半音）を返す。"""
    logs_dir = Path(os.environ.get("APPLIO_LOGS_DIR", model_dir.parent))
    reference_dir = logs_dir / "reference"
    phone = reference_features(model_dir, logs_dir, embedder)
    if phone is None:
        return 0
    pitchf = np.load(reference_dir / "pitch_fine.npy")[:-1].astype(np.float64)
    voiced = pitchf[pitchf > 0]
    shift = 0
    if target_f0 and len(voiced):
        shift = int(round(12 * math.log2(target_f0 / float(np.median(voiced)))))
        shift = max(-24, min(24, shift))
    pitchf = pitchf * (2 ** (shift / 12))
    pitch = coarse_f0(pitchf)
    frames = int(SAMPLE_SECONDS * 100)
    generated, _ = run_model(net_g, device, phone[: frames // 2], pitch[:frames], pitchf[:frames].astype(np.float32), seed=4321)
    fade = min(len(generated), int(sample_rate * 0.05))
    if fade:
        generated[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    peak = float(np.max(np.abs(generated))) if len(generated) else 0.0
    if peak > 0.99:
        generated = generated * (0.99 / peak)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), generated, sample_rate, format="FLAC", subtype="PCM_16")
    return shift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--sample", type=Path, help="試し聴き用の音声の保存先（.flac）")
    parser.add_argument("--target-f0", type=float, help="試し聴き用の音声の声の高さ（Hz）")
    parser.add_argument("--embedder", default="contentvec")
    args = parser.parse_args()

    device = pick_device()
    net_g, sample_rate = load_generator(args.weight, device)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    data_config = config["data"]
    try:
        names = json.loads((args.model_dir / "validation.json").read_text(encoding="utf-8"))["clips"]
    except (OSError, ValueError, KeyError):
        names = []
    loss, clips = validation_loss(net_g, device, args.model_dir, names, data_config) if names else (None, 0)
    result = {"epoch": args.epoch, "weight": args.weight.name, "val_loss": loss, "clips": clips}
    if args.sample:
        try:
            result["sample_shift"] = make_sample(
                net_g, device, sample_rate, args.model_dir, args.embedder, args.target_f0, args.sample
            )
            result["sample"] = args.sample.name if args.sample.is_file() else None
        except Exception as error:
            print(f"試し聴き用の音声を作れませんでした: {error}", file=sys.stderr)
            result["sample"] = None
    print("@@eval " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
