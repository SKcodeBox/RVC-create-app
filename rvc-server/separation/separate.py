"""ボーカル分離: 音声から BGM・伴奏を取り除いて声だけにする（必要なら残響・ハモリも取り除く）。

RVC Studio のサーバーが、専用の仮想環境（rvc-server/.venv-separation）の Python で実行する。
AI モデルは audio-separator（MIT ライセンス）で動かし、使うモデルは models.json で決める。
1つの手順に複数のモデルを指定すると、それぞれの結果を組み合わせる（アンサンブル。精度が上がる）。

使い方:
  python separate.py --models-dir DIR --steps vocals,dereverb [--quality high] --download-only
      必要な AI モデルをダウンロードだけする（ダウンロード済みなら何もしない）
  python separate.py --models-dir DIR --work-dir DIR --steps vocals,dereverb [--keep-stages] IN1 [IN2 ...]
      学習用（--mode dataset）: それぞれの音声を処理し、結果を <work-dir>/final/<番号>.wav（モノラル）に書き出す。
      --keep-stages を付けると、手順ごとの途中結果を <work-dir>/stages/<番号>/<手順>.flac に残す（聴き比べ用）
  python separate.py --mode song --models-dir DIR --work-dir DIR --steps vocals[,dereverb][,karaoke] IN
      曲の変換用: 声と伴奏を分けて、<work-dir>/final/<番号>_lead.wav（メインの声）・_instrumental.wav（伴奏）・
      _backing.wav（ハモリ。karaoke のときだけ）を書き出す。入力は 44.1kHz の WAV にしておくこと
      （伴奏は「元の音声 − 分けた声」で作るので、モデルと同じサンプルレートである必要がある）

サーバーが進み具合を読み取れるよう、次の行を出力する（ほかに audio-separator の進捗バーも出る）:
  @@download <何番目> <数> <モデル>  … モデルのダウンロード中
  @@step <何番目> <数> <手順>        … 手順の開始
  @@model <何番目> <数>              … その手順で使っているモデル（アンサンブルのとき）
  @@file <何番目> <数>               … いまのモデルで処理している音声（長い音声は区切った単位で数える）
  @@result <番号> <声のある秒数> <全体の秒数>
"""

import argparse
import gc
import importlib.util
import json
import logging
import shutil
import sys
import types
import warnings
from pathlib import Path

# pydub は ffmpeg が見つからないと警告を出すが、このスクリプトでは pydub を使わない
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")


def _stand_in_for_onnxruntime() -> None:
    """onnxruntime の代わりの、何もしないモジュールを置く。

    audio-separator は ONNX 形式のモデル（MDX 系）のために、起動時に onnxruntime を読み込む。
    このアプリで使うモデル（models.json の RoFormer・VR 系）は PyTorch だけで動くので、
    onnxruntime（Microsoft へ利用状況を送る処理を含む）は入れていない。読み込みだけ通るようにする。
    """
    if importlib.util.find_spec("onnxruntime") is not None:
        return
    module = types.ModuleType("onnxruntime")
    module.get_available_providers = lambda: ["CPUExecutionProvider"]

    def missing(name: str):
        raise AttributeError(
            f"onnxruntime.{name} は使えません（onnxruntime を入れていないため、ONNX 形式のモデルには対応していません）"
        )

    module.__getattr__ = missing
    sys.modules["onnxruntime"] = module


_stand_in_for_onnxruntime()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from audio_separator.separator import Separator  # noqa: E402
from audio_separator.separator.ensembler import Ensembler  # noqa: E402

CONFIG = json.loads((Path(__file__).with_name("models.json")).read_text(encoding="utf-8"))
STEPS: dict = CONFIG["steps"]
SAMPLE_RATE = 44100

# Applio の前処理（無音で切り分ける処理）が無音とみなす大きさ（dB）。これより大きい部分を「声がある」と数える
SILENCE_DB = -42.0
# これより長い音声は、メモリが足りなくならないよう区切って処理する（秒）
CHUNK_SECONDS = 600.0
# 区切ったときに最後に残る部分がこれより短ければ、1つ前とまとめる（秒）
MIN_TAIL_SECONDS = 30.0


def step_setting(step: str, quality: str) -> dict:
    """その手順の設定（指定の精度がなければ、いちばん精度の高い設定）。"""
    qualities = STEPS[step]["quality"]
    return qualities.get(quality) or qualities["high"]


def step_models(step: str, quality: str) -> list[dict]:
    return list(step_setting(step, quality)["models"])


class WavOnlySeparator(Separator):
    """WAV の読み書きだけで使う Separator。

    audio-separator は起動時に ffmpeg の有無を確かめ、なければ止まる。
    ffmpeg は pydub で読み書きするときだけ使うもので、use_soundfile=True で書き出し、
    長い音声の区切り（pydub を使う chunk_duration）も使わなければ不要。
    """

    def check_ffmpeg_installed(self):
        pass


def say(*parts) -> None:
    print("@@" + " ".join(str(part) for part in parts), flush=True)


def make_separator(models_dir: Path, output_dir: Path) -> WavOnlySeparator:
    return WavOnlySeparator(
        log_level=logging.WARNING,
        model_file_dir=str(models_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        use_soundfile=True,
        sample_rate=SAMPLE_RATE,
        # 音量を変えない（伴奏を「元の音声 − 声」で作るため。音割れするときだけ下げる）
        normalization_threshold=1.0,
    )


def load(separator: WavOnlySeparator, model: dict) -> None:
    """モデルを読み込む（処理の区切りの重なりなど、モデルごとの設定も反映する）。"""
    separator.arch_specific_params["MDXC"]["overlap"] = model.get("overlap")
    separator.output_single_stem = model["stem"]
    separator.load_model(model_filename=model["file"])


def release(separator: WavOnlySeparator) -> None:
    """次のモデルを読み込む前に、いまのモデルのメモリを空ける。"""
    separator.model_instance = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_input(path: Path, parts_dir: Path, seconds: float) -> list[Path]:
    """音声を audio-separator で扱える WAV にする（長い音声は区切る）。

    - audio-separator は入力と同じ形式で書き出そうとするため、MP3・OGG をそのまま渡すと
      WAV に書き出せずに止まる。そこで WAV にしてから渡す
    - audio-separator にも長い音声を区切る機能があるが、pydub を使うため MP3 などで ffmpeg が必要になる。
      ここでは soundfile で読み書きする
    """
    info = sf.info(str(path))
    rate = info.samplerate
    size = int(seconds * rate)
    if info.frames <= size + MIN_TAIL_SECONDS * rate:
        if info.format == "WAV" and info.subtype in ("PCM_16", "PCM_24", "FLOAT"):
            return [path]  # そのまま使える
        starts = [0]
    else:
        starts = list(range(0, info.frames, size))
        if info.frames - starts[-1] < MIN_TAIL_SECONDS * rate:
            starts.pop()  # 短すぎる最後の部分は、1つ前に含める
    ends = [*starts[1:], info.frames]
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with sf.SoundFile(str(path)) as source:
        for number, (start, end) in enumerate(zip(starts, ends), start=1):
            source.seek(start)
            data = source.read(end - start, dtype="float32", always_2d=True)
            part = parts_dir / f"{number:03d}.wav"
            sf.write(str(part), data, rate, subtype="FLOAT")
            parts.append(part)
    return parts


def voiced_seconds(mono: np.ndarray, rate: int) -> float:
    """声が入っている（学習に使われる）部分の長さの目安。50ミリ秒ごとの音の大きさで数える。"""
    frame = max(int(rate * 0.05), 1)
    count = len(mono) // frame
    if count == 0:
        return 0.0
    frames = mono[: count * frame].reshape(count, frame).astype(np.float64)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    return float(np.count_nonzero(rms_db > SILENCE_DB) * frame / rate)


def read(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data, rate


def fit(data: np.ndarray, frames: int, channels: int) -> np.ndarray:
    """長さとチャンネル数をそろえる（引き算をするため）。"""
    if data.shape[1] != channels:
        data = np.repeat(data.mean(axis=1, keepdims=True), channels, axis=1)
    if len(data) < frames:
        data = np.concatenate([data, np.zeros((frames - len(data), channels), dtype=data.dtype)])
    return data[:frames]


def combine(paths: list[Path], algorithm: str, target: Path) -> Path:
    """複数のモデルの結果を組み合わせる（audio-separator のアンサンブルと同じ計算）。"""
    waves = []
    rate = SAMPLE_RATE
    frames = None
    for path in paths:
        data, rate = read(path)
        frames = len(data) if frames is None else min(frames, len(data))
        waves.append(data)
    waves = [wave[:frames].T for wave in waves]
    combined = Ensembler(logging.getLogger("ensemble"), algorithm=algorithm).ensemble(waves)
    sf.write(str(target), np.asarray(combined, dtype=np.float32).T, rate, subtype="FLOAT")
    for path in paths:
        path.unlink(missing_ok=True)
    return target


def run_steps(separator, steps: list[str], quality: str, current: list[Path], work_dir: Path, on_step_done) -> list[Path]:
    """手順ごとに、モデルを1回だけ読み込んで、すべての音声を処理する。"""
    count = len(current)
    for number, step in enumerate(steps, start=1):
        setting = step_setting(step, quality)
        models = setting["models"]
        say("step", number, len(steps), step)
        results: list[list[Path]] = []  # results[モデル][音声]
        for model_number, model in enumerate(models, start=1):
            say("model", model_number, len(models))
            load(separator, model)
            results.append([])
            for position, path in enumerate(current):
                say("file", position + 1, count)
                name = f"step{number}_{model_number}_{position + 1}"
                outputs = separator.separate(str(path), custom_output_names={model["stem"]: name})
                # 書き出されるのは残す音の1つだけ。名前は返ってきた一覧で確かめる
                produced = [work_dir / Path(output).name for output in outputs or []]
                produced = [output for output in produced if output.is_file()]
                if len(produced) != 1:
                    raise RuntimeError(f"分離した音声が書き出されませんでした: {outputs}")
                results[-1].append(produced[0].rename(work_dir / f"{name}.wav"))
            release(separator)
        previous = list(current)
        for position, path in enumerate(current):
            pieces = [outputs[position] for outputs in results]
            if len(pieces) > 1:
                merged = combine(pieces, setting.get("ensemble", "avg_wave"), work_dir / f"step{number}_{position + 1}.wav")
            else:
                merged = pieces[0]
            current[position] = merged
        on_step_done(step, previous, current)
    return current


def finish_mono(pieces: list[Path], target: Path) -> tuple[float, float]:
    """分離した音声（区切った場合は順につなぐ）をモノラル・16bit で保存する。

    音量は上げない（声がほとんどない音声だと、取り切れなかった小さな雑音まで大きくしてしまうため）。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    rate = sf.info(str(pieces[0])).samplerate
    voiced = total = 0.0
    with sf.SoundFile(str(target), "w", samplerate=rate, channels=1, subtype="PCM_16") as out:
        for piece in pieces:
            data, _ = read(piece)
            mono = np.clip(data.mean(axis=1), -1.0, 1.0)
            out.write(mono)
            voiced += voiced_seconds(mono, rate)
            total += len(mono) / rate
    return voiced, total


def concat(pieces: list[Path], target: Path, subtype: str = "FLOAT", mono: bool = False, fmt: str = "WAV") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    first = sf.info(str(pieces[0]))
    channels = 1 if mono else first.channels
    with sf.SoundFile(str(target), "w", samplerate=first.samplerate, channels=channels, subtype=subtype, format=fmt) as out:
        for piece in pieces:
            data, _ = read(piece)
            out.write(data.mean(axis=1) if mono else data)


def run_dataset(args, separator, steps: list[str], work_dir: Path) -> int:
    # 処理の単位: (元のファイルの番号, 音声)。長い音声は区切って、複数の単位にする
    units: list[tuple[int, Path]] = []
    for index, path in enumerate(args.files, start=1):
        for part in prepare_input(path, work_dir / "parts" / str(index), args.chunk_seconds):
            units.append((index, part))
    current = [part for _, part in units]
    owners = [owner for owner, _ in units]

    def keep(name: str, paths: list[Path]) -> None:
        for index in range(1, len(args.files) + 1):
            pieces = [paths[k] for k, owner in enumerate(owners) if owner == index]
            concat(pieces, work_dir / "stages" / str(index) / f"{name}.flac", subtype="PCM_16", mono=True, fmt="FLAC")

    if args.keep_stages:
        keep("original", current)

    def on_step_done(step: str, previous: list[Path], now: list[Path]) -> None:
        if args.keep_stages:
            keep(step, now)
        for path in previous:
            if work_dir in path.parents and "parts" not in path.parts:
                path.unlink(missing_ok=True)  # 前の手順の途中結果

    current = run_steps(separator, steps, args.quality, current, work_dir, on_step_done)
    for index in range(1, len(args.files) + 1):
        pieces = [current[k] for k, owner in enumerate(owners) if owner == index]
        voiced, total = finish_mono(pieces, work_dir / "final" / f"{index}.wav")
        say("result", index, f"{voiced:.1f}", f"{total:.1f}")
    return 0


def run_song(args, separator, steps: list[str], work_dir: Path) -> int:
    """曲を「メインの声」「伴奏」「ハモリ」に分ける。伴奏とハモリは、引き算で作る（何も失わないように）。"""
    units: list[tuple[int, Path]] = []
    for index, path in enumerate(args.files, start=1):
        if sf.info(str(path)).samplerate != SAMPLE_RATE:
            print(f"入力は {SAMPLE_RATE}Hz にしてください: {path}", file=sys.stderr)
            return 2
        for part in prepare_input(path, work_dir / "parts" / str(index), args.chunk_seconds):
            units.append((index, part))
    current = [part for _, part in units]
    owners = [owner for owner, _ in units]
    extras: dict[str, list[Path]] = {"instrumental": [], "backing": []}

    def subtract(step_input: list[Path], step_output: list[Path], name: str) -> None:
        paths = []
        for position, (source, separated) in enumerate(zip(step_input, step_output)):
            mix, rate = read(source)
            part, _ = read(separated)
            rest = mix - fit(part, len(mix), mix.shape[1])
            target = work_dir / f"{name}_{position + 1}.wav"
            sf.write(str(target), rest, rate, subtype="FLOAT")
            paths.append(target)
        extras[name] = paths

    def on_step_done(step: str, previous: list[Path], now: list[Path]) -> None:
        if step == "vocals":
            subtract(previous, now, "instrumental")
        elif step == "karaoke":
            subtract(previous, now, "backing")
        for path in previous:
            if work_dir in path.parents and "parts" not in path.parts:
                path.unlink(missing_ok=True)

    current = run_steps(separator, steps, args.quality, current, work_dir, on_step_done)
    for index in range(1, len(args.files) + 1):
        positions = [k for k, owner in enumerate(owners) if owner == index]
        concat([current[k] for k in positions], work_dir / "final" / f"{index}_lead.wav")
        for name, paths in extras.items():
            if paths:
                concat([paths[k] for k in positions], work_dir / "final" / f"{index}_{name}.wav")
        total = sum(sf.info(str(current[k])).duration for k in positions)
        say("result", index, "0", f"{total:.1f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--steps", required=True, help="vocals,dereverb,karaoke のうち使うもの")
    parser.add_argument("--quality", default="high", choices=CONFIG["qualities"])
    parser.add_argument("--mode", default="dataset", choices=["dataset", "song"])
    parser.add_argument("--keep-stages", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--chunk-seconds", type=float, default=CHUNK_SECONDS,
        help="これより長い音声は区切って処理する（秒）",
    )
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    requested = [step for step in args.steps.split(",") if step]
    steps = [step for step in CONFIG["order"] if step in requested]
    unknown = set(requested) - set(STEPS)
    if unknown or not steps:
        parser.error(f"使えない手順です: {', '.join(sorted(unknown)) or '(なし)'}")
    if args.mode == "song" and "vocals" not in steps:
        parser.error("曲の変換では vocals（BGM・伴奏の除去）が必要です")
    args.models_dir.mkdir(parents=True, exist_ok=True)

    if args.download_only:
        separator = make_separator(args.models_dir, args.models_dir)
        models = [model["file"] for step in steps for model in step_models(step, args.quality)]
        models = list(dict.fromkeys(models))
        for number, model in enumerate(models, start=1):
            say("download", number, len(models), model)
            _, model_type, _, model_path, _ = separator.download_model_files(model)
            if model_type in ("VR", "MDX"):
                # この種類のモデルは、設定（UVR の一覧）を初めて使うときにも取りに行くので、ここで取っておく
                separator.load_model_data_using_hash(model_path)
        return 0

    if not args.work_dir or not args.files:
        parser.error("--work-dir と、処理する音声ファイルを指定してください")
    for path in args.files:
        if not path.is_file():
            parser.error(f"ファイルが見つかりません: {path}")

    work_dir = args.work_dir
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    separator = make_separator(args.models_dir, work_dir)
    try:
        if args.mode == "song":
            return run_song(args, separator, steps, work_dir)
        return run_dataset(args, separator, steps, work_dir)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
