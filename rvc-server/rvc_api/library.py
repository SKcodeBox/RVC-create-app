"""声のモデル（学習したもの・取り込んだもの）と学習用データの管理。

モデルは data/models/<モデル名>/ に置かれる（Applio の logs フォルダと同じ構成）。
  - <モデル名>_<エポック>e_<ステップ>s.pth … 変換に使うモデル本体（保存した回数だけある）
  - <モデル名>.index                        … 声の特徴を検索するためのインデックス
  - G_*.pth / D_*.pth                        … 学習を再開するためのチェックポイント
  - sliced_audios/ f0/ extracted/ など       … 学習の途中のファイル（続きから学習するときに使う）
  - studio/                                  … このアプリが作るファイル（評価結果・試し聴き用の音声）
学習用の音声は data/datasets/<モデル名>/ に置かれる。
  - .stages/<元の名前>/<手順>.flac            … 下ごしらえの途中経過（聴き比べ用）
"""

import collections
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from . import audio, evaluation, settings

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
WEIGHT_RE = re.compile(r"^(?P<name>.+)_(?P<epoch>\d+)e_(?P<step>\d+)s\.pth$")
META_FILE = "rvc_studio.json"
STAGES_DIR = ".stages"
STAGE_ORDER = ("original", "vocals", "dereverb", "karaoke")
# 学習の途中のファイル（続きから学習するとき・検証のときに使う）
TRAINING_DIRS = ("sliced_audios", "sliced_audios_16k", "f0", "f0_voiced", "extracted")
# 声の特徴を取り出すモデル（Applio が対応しているもの）
EMBEDDERS = ("contentvec", "spin", "spin-v2", "japanese-hubert-base", "chinese-hubert-base", "korean-hubert-base")
# Windows でファイル名・フォルダ名に使えない名前（大文字・小文字は区別しない）
WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}
# 取り込んだモデルの声の高さの目安（Hz）。Applio の「声の高さの自動調整」の初期値と同じ
VOICE_TYPES = {"low": 155.0, "high": 255.0}


class LibraryError(Exception):
    """利用者に見せるメッセージを持つエラー。"""


def validate_name(name: str) -> str:
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise LibraryError(
            "モデル名は、半角英数字で始まる40文字以内で入力してください"
            "（使える文字: 半角英数字・ハイフン「-」・アンダーバー「_」）。"
        )
    if name.lower() in settings.RESERVED_NAMES:
        raise LibraryError(f"「{name}」はシステムで使う名前のため使えません。")
    if name.lower() in WINDOWS_RESERVED:
        # Windows ではこの名前のフォルダを作れない（Mac で作ったモデルを Windows に移すときも困る）
        raise LibraryError(f"「{name}」は Windows で使えない名前のため、別の名前にしてください。")
    return name


def validate_embedder(embedder: str | None) -> str:
    embedder = (embedder or "contentvec").strip()
    if embedder not in EMBEDDERS:
        raise LibraryError("声の特徴を取り出すモデルの指定が正しくありません。")
    return embedder


def resolve_model(name: str) -> str:
    """すでにあるモデルの名前を確かめる（models フォルダの外は指せないようにする）。"""
    name = (name or "").strip()
    root = settings.MODELS_DIR.resolve()
    if (
        not name
        or name.startswith(".")
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or name in settings.RESERVED_NAMES
        or (root / name).resolve().parent != root
        or not (root / name).is_dir()
    ):
        raise LibraryError(f"「{name}」というモデルは見つかりません。")
    # Mac・Windows は大文字と小文字を区別しないので、フォルダの本当の名前にそろえる
    # （"Voice" と "voice" を別のモデルとして扱い、学習中のモデルを消せてしまうことを防ぐ）
    entries = os.listdir(root)
    if name not in entries:
        actual = next((entry for entry in entries if entry.lower() == name.lower()), None)
        if actual is None:
            raise LibraryError(f"「{name}」というモデルは見つかりません。")
        name = actual
    return name


def model_dir(name: str) -> Path:
    return settings.MODELS_DIR / name


def dataset_dir(name: str) -> Path:
    return settings.DATASETS_DIR / name


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_meta(name: str) -> dict:
    return _read_json(model_dir(name) / META_FILE)


def write_meta(name: str, **fields) -> None:
    meta = read_meta(name)
    meta.update(fields)
    path = model_dir(name) / META_FILE
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def list_weights(name: str) -> list[dict]:
    """変換に使えるモデルファイル（.pth）の一覧。古い順。"""
    directory = model_dir(name)
    weights = []
    for path in directory.glob("*.pth"):
        if path.name.startswith(("G_", "D_")):
            continue
        match = WEIGHT_RE.match(path.name)
        stat = path.stat()
        weights.append(
            {
                "file": path.name,
                "epoch": int(match["epoch"]) if match else None,
                "step": int(match["step"]) if match else None,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
            }
        )
    weights.sort(key=lambda w: (w["epoch"] or 0, w["modified_at"]))
    return weights


def find_index(name: str) -> Path | None:
    directory = model_dir(name)
    preferred = directory / f"{name}.index"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        directory.glob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def has_checkpoint(name: str) -> bool:
    return any(model_dir(name).glob("G_*.pth"))


class _Placeholder:
    """チェックポイントの中身（テンソルなど）を読まずに済ませるための仮の値。"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


class _HeaderUnpickler(pickle.Unpickler):
    """PyTorch のチェックポイントの目次（data.pkl）だけを、安全に読む。

    テンソルの本体は読まず、知らないクラスはすべて仮の値にする（任意のコードは実行されない）。
    """

    SAFE = {("collections", "OrderedDict"): collections.OrderedDict}

    def find_class(self, module, name):
        return self.SAFE.get((module, name), _Placeholder)

    def persistent_load(self, pid):
        return None


_checkpoint_cache: dict[str, tuple[tuple[float, int], int | None]] = {}


def _checkpoint_iteration(path: Path) -> int | None:
    """チェックポイントに記録されたエポック数。壊れている（保存の途中で止まったなど）なら None。"""
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = (stat.st_mtime, stat.st_size)
    cached = _checkpoint_cache.get(str(path))
    if cached and cached[0] == signature:
        return cached[1]
    iteration = None
    try:
        # 書きかけのファイルは、末尾にある zip の目次がないので開けない
        with zipfile.ZipFile(path) as archive:
            member = next(
                (info for info in archive.infolist() if info.filename == "data.pkl" or info.filename.endswith("/data.pkl")),
                None,
            )
            if member is not None and member.file_size < 64 * 1024 * 1024:
                data = _HeaderUnpickler(io.BytesIO(archive.read(member))).load()
                value = data.get("iteration") if isinstance(data, dict) else None
                iteration = int(value) if isinstance(value, int) else None
    except (OSError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError, ValueError, TypeError, AttributeError):
        iteration = None
    _checkpoint_cache[str(path)] = (signature, iteration)
    return iteration


def checkpoint_state(name: str) -> dict:
    """学習を再開するためのファイル（G_*.pth と D_*.pth）の状態。

    exists: ファイルがある / ok: 両方とも読める / epoch: 記録されたエポック数（再開するとこの次から）
    """
    directory = model_dir(name)
    generators = sorted(directory.glob("G_*.pth"))
    discriminators = sorted(directory.glob("D_*.pth"))
    if not generators and not discriminators:
        return {"exists": False, "ok": False, "epoch": None}
    g_epoch = _checkpoint_iteration(generators[-1]) if generators else None
    d_epoch = _checkpoint_iteration(discriminators[-1]) if discriminators else None
    ok = g_epoch is not None and d_epoch is not None
    return {"exists": True, "ok": ok, "epoch": g_epoch if ok else None}


def checkpoint_epoch(name: str) -> int:
    """学習を再開できるチェックポイントのエポック数（なければ 0）。"""
    state = checkpoint_state(name)
    if state["ok"] and state["epoch"]:
        return state["epoch"]
    return 0


def steps_per_epoch(weights: list[dict]) -> int | None:
    for weight in reversed(weights):
        if weight["epoch"] and weight["step"]:
            return max(1, round(weight["step"] / weight["epoch"]))
    return None


def recommended_weight(name: str, weights: list[dict] | None = None) -> dict | None:
    """変換に使うのにおすすめのモデルファイル（検証用の誤差がいちばん小さいもの）。"""
    weights = weights if weights is not None else list_weights(name)
    by_epoch = {w["epoch"]: w for w in weights if w["epoch"] is not None}
    best = evaluation.best_evaluation(
        evaluation.read_evaluations(model_dir(name)), set(by_epoch)
    )
    if best is None:
        return None
    return {**by_epoch[best["epoch"]], "val_loss": best["val_loss"]}


def target_pitch(name: str, meta: dict | None = None) -> float | None:
    """このモデルの声の高さの目安（Hz）。学習用の音声から計算したもの、または取り込むときに選んだもの。"""
    meta = meta if meta is not None else read_meta(name)
    pitch = meta.get("pitch") or {}
    if pitch.get("median"):
        return float(pitch["median"])
    voice_type = meta.get("voice_type")
    return VOICE_TYPES.get(voice_type)


def model_summary(name: str) -> dict | None:
    directory = model_dir(name)
    if not directory.is_dir():
        return None
    meta = read_meta(name)
    weights = list_weights(name)
    index = find_index(name)
    config = _read_json(directory / "config.json")
    info = _read_json(directory / "model_info.json")
    stat = directory.stat()
    created_at = meta.get("created_at") or getattr(stat, "st_birthtime", stat.st_mtime)
    recommended = recommended_weight(name, weights)
    evaluations = evaluation.read_evaluations(directory)
    checkpoint = checkpoint_state(name)
    weight_epochs = max((w["epoch"] or 0 for w in weights), default=0)
    return {
        "name": name,
        "source": meta.get("source", "trained"),
        "created_at": created_at,
        "weights": weights,
        "latest_weight": weights[-1]["file"] if weights else None,
        "recommended_weight": recommended["file"] if recommended else None,
        "recommended_epoch": recommended["epoch"] if recommended else None,
        "evaluated": len(evaluations),
        "index_file": index.name if index else None,
        "sample_rate": (config.get("data") or {}).get("sample_rate")
        or meta.get("sample_rate"),
        # Applio の前処理が書き出す学習用音声の合計秒数
        "dataset_duration": info.get("total_seconds"),
        # 再開用のファイルの方が新しいことがある（変換用のモデルを書き出す前に止まった場合）
        "trained_epochs": max(weight_epochs, checkpoint["epoch"] or 0),
        "can_resume": checkpoint["ok"] and (directory / "filelist.txt").is_file(),
        # 再開用のファイルが壊れている（保存の途中で止まったなど）。続きから学習はできない
        "checkpoint_broken": checkpoint["exists"] and not checkpoint["ok"],
        # 最後の学習が最後まで終わったか（キャンセル・中断・失敗なら False。以前の版で作ったモデルは True）
        "finished": meta.get("source") == "imported" or meta.get("finished_at", True) is not None,
        # 学習の前に行った音声の下ごしらえ（"vocals" など。rvc_api/separation.py を参照）
        "prep": meta.get("prep") or [],
        "embedder": meta.get("embedder") or info.get("embedder_model") or "contentvec",
        "pretrained": meta.get("pretrained"),
        "early_stopped_at": meta.get("early_stopped_at"),
        "target_pitch": target_pitch(name, meta),
        "voice_type": meta.get("voice_type"),
    }


def list_models() -> list[dict]:
    if not settings.MODELS_DIR.is_dir():
        return []
    models = []
    for directory in settings.MODELS_DIR.iterdir():
        if (
            not directory.is_dir()
            or directory.name.startswith(".")
            or directory.name in settings.RESERVED_NAMES
        ):
            continue
        summary = model_summary(directory.name)
        if summary:
            models.append(summary)
    models.sort(key=lambda m: m["created_at"], reverse=True)
    return models


def _inside(directory: Path, filename: str) -> Path:
    """フォルダの中のファイルを安全に取り出す（フォルダの外は指せない）。"""
    if not filename or "\x00" in filename:
        raise LibraryError("ファイルが見つかりません。")
    directory = directory.resolve()
    path = (directory / filename).resolve()
    if path.parent != directory or not path.is_file():
        raise LibraryError("ファイルが見つかりません。")
    return path


def file_in_model(name: str, filename: str) -> Path:
    path = _inside(model_dir(name), filename)
    if path.suffix not in (".pth", ".index") or path.name.startswith(("G_", "D_")):
        raise LibraryError("このファイルはダウンロードできません。")
    return path


def sample_file(name: str, epoch: int) -> Path:
    path = evaluation.sample_path(model_dir(name), epoch)
    if not path.is_file():
        raise LibraryError("試し聴き用の音声が見つかりません。")
    return path


def delete_model(name: str) -> None:
    for directory in (model_dir(name), dataset_dir(name)):
        if directory.is_dir():
            shutil.rmtree(directory)


def delete_weight(name: str, filename: str) -> None:
    """途中で保存したモデルファイルを1つ消す。"""
    path = _inside(model_dir(name), filename)
    weights = list_weights(name)
    target = next((w for w in weights if w["file"] == path.name), None)
    if target is None:
        raise LibraryError("このファイルは削除できません。")
    if len(weights) == 1:
        raise LibraryError("最後の1つのモデルファイルは削除できません（モデルごと削除してください）。")
    if has_checkpoint(name) and target["file"] == weights[-1]["file"]:
        raise LibraryError(
            "最新のモデルファイルは「続きから学習」に必要なため削除できません。"
            "続きから学習しない場合は、先に「学習を完了にする」を使ってください。"
        )
    path.unlink()
    if target["epoch"] is not None:
        evaluation.forget_epochs(model_dir(name), {target["epoch"]})


def folder_size(path: Path) -> int:
    """ファイルまたはフォルダの合計サイズ（バイト）。"""
    return _size(path)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for file in files:
            try:
                total += (Path(root) / file).stat().st_size
            except OSError:
                pass
    return total


def storage(name: str) -> dict:
    """モデルが使っている保存容量（バイト）の内訳。"""
    directory = model_dir(name)
    weights = list_weights(name)
    recommended = recommended_weight(name, weights)
    keep = {w["file"] for w in weights[-1:]} | ({recommended["file"]} if recommended else set())
    data = dataset_dir(name)
    return {
        "weights": sum(w["size"] for w in weights),
        "extra_weights": sum(w["size"] for w in weights if w["file"] not in keep),
        "extra_weight_count": sum(1 for w in weights if w["file"] not in keep),
        "checkpoints": sum(_size(p) for p in [*directory.glob("[GD]_*.pth"), *directory.glob("*.pth.tmp")]),
        "training_files": sum(
            _size(directory / folder) for folder in (*TRAINING_DIRS, "eval") if (directory / folder).exists()
        ),
        "index": sum(_size(p) for p in directory.glob("*.index")),
        "samples": _size(evaluation.studio_dir(directory)) if evaluation.studio_dir(directory).exists() else 0,
        "dataset": sum(p.stat().st_size for p in dataset_paths(name)),
        "stages": _size(data / STAGES_DIR) if (data / STAGES_DIR).exists() else 0,
    }


def cleanup(name: str, weights: bool = False, checkpoints: bool = False, stages: bool = False) -> dict:
    """保存容量を空ける。weights: おすすめと最新以外のモデルファイル、checkpoints: 続きから学習するための
    ファイル（チェックポイントと学習の途中のファイル）、stages: 下ごしらえの途中経過の音声。"""
    directory = model_dir(name)
    freed = 0
    if weights:
        all_weights = list_weights(name)
        recommended = recommended_weight(name, all_weights)
        keep = {w["file"] for w in all_weights[-1:]} | ({recommended["file"]} if recommended else set())
        removed = set()
        for weight in all_weights:
            if weight["file"] in keep:
                continue
            (directory / weight["file"]).unlink(missing_ok=True)
            freed += weight["size"]
            if weight["epoch"] is not None:
                removed.add(weight["epoch"])
        if removed:
            evaluation.forget_epochs(directory, removed)
    if checkpoints:
        for path in [*directory.glob("[GD]_*.pth"), *directory.glob("*.pth.tmp")]:
            freed += _size(path)
            path.unlink(missing_ok=True)
        for folder in TRAINING_DIRS:
            if (directory / folder).exists():
                freed += _size(directory / folder)
                shutil.rmtree(directory / folder, ignore_errors=True)
        write_meta(name, finalized_at=time.time())
    if stages:
        folder = dataset_dir(name) / STAGES_DIR
        if folder.exists():
            freed += _size(folder)
            shutil.rmtree(folder, ignore_errors=True)
    return {"freed": freed}


def import_model(name: str, pth_upload, index_upload=None, voice_type: str | None = None,
                 embedder: str | None = None) -> dict:
    """他で作られた RVC モデル（.pth と .index）を取り込む。"""
    name = validate_name(name)
    if model_dir(name).exists():
        raise LibraryError(f"「{name}」という名前のモデルはすでにあります。")
    if not (pth_upload.filename or "").lower().endswith(".pth"):
        raise LibraryError("モデルファイルは .pth を選んでください。")
    if index_upload is not None and not (index_upload.filename or "").lower().endswith(
        ".index"
    ):
        raise LibraryError("インデックスファイルは .index を選んでください。")
    if voice_type and voice_type not in VOICE_TYPES:
        raise LibraryError("声の高さの指定が正しくありません。")

    directory = model_dir(name)
    directory.mkdir(parents=True)
    try:
        _save_upload(pth_upload, directory / f"{name}.pth")
        if index_upload is not None:
            _save_upload(index_upload, directory / f"{name}.index")
        info = inspect_weight(directory / f"{name}.pth")
        if info is not None and not info.get("ok"):
            raise LibraryError("RVC のモデルファイルとして読み込めませんでした。ファイルを確認してください。")
        info = info or {}
        detected = info.get("embedder")
        write_meta(
            name,
            source="imported",
            created_at=time.time(),
            voice_type=voice_type or None,
            sample_rate=info.get("sample_rate"),
            embedder=validate_embedder(embedder) if embedder else (detected if detected in EMBEDDERS else "contentvec"),
        )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return model_summary(name)


def inspect_weight(path: Path) -> dict | None:
    """モデルファイルの情報を調べる（engine/model_info.py）。調べられなければ None。"""
    try:
        result = subprocess.run(
            [settings.PYTHON, str(settings.ENGINE_DIR / "model_info.py"), str(path)],
            capture_output=True, text=True, timeout=180, env=settings.applio_env(),
            cwd=settings.APPLIO_DIR if settings.APPLIO_DIR.is_dir() else None,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("@@model "):
            try:
                return json.loads(line[len("@@model "):])
            except ValueError:
                return None
    return None


def _save_upload(upload, target: Path) -> None:
    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    if target.stat().st_size == 0:
        raise LibraryError(f"「{upload.filename}」が空のファイルです。")


def save_dataset_files(name: str, uploads) -> list[dict]:
    """学習用の音声を保存する。読めない形式は WAV に変換する。"""
    directory = dataset_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    try:
        for upload in uploads:
            filename = upload.filename or "audio.wav"
            audio.check_extension(filename)
            target = audio.unique_path(directory, audio.safe_filename(filename))
            _save_upload(upload, target)
            saved.append(target)
            saved[-1] = audio.make_readable(target, mono=False)
    except Exception:
        for path in saved:
            path.unlink(missing_ok=True)
        raise
    return dataset_files(name)


def dataset_paths(name: str) -> list[Path]:
    """学習用の音声ファイル（名前順）。"""
    directory = dataset_dir(name)
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in audio.NATIVE_EXTS
    ]


def stages_of(name: str, stem: str) -> list[str]:
    folder = dataset_dir(name) / STAGES_DIR / stem
    if not folder.is_dir():
        return []
    present = {path.stem for path in folder.glob("*.flac")}
    return [stage for stage in STAGE_ORDER if stage in present]


def dataset_files(name: str) -> list[dict]:
    return [
        {
            "file": path.name,
            "size": path.stat().st_size,
            "duration": audio.duration_of(path),
            # 下ごしらえの途中経過（聴き比べ用）。"original" は下ごしらえの前
            "stages": stages_of(name, path.stem),
        }
        for path in dataset_paths(name)
    ]


def dataset_file(name: str, filename: str) -> Path:
    """学習用の音声を1つ安全に取り出す（フォルダの外は指せない）。"""
    path = _inside(dataset_dir(name), filename)
    if path.suffix.lower() not in audio.NATIVE_EXTS:
        raise LibraryError("ファイルが見つかりません。")
    return path


def stage_file(name: str, filename: str, stage: str) -> Path:
    """下ごしらえの途中経過の音声を1つ安全に取り出す。"""
    if stage not in STAGE_ORDER or "\x00" in filename:
        raise LibraryError("ファイルが見つかりません。")
    root = (dataset_dir(name) / STAGES_DIR).resolve()
    folder = (root / Path(filename).stem).resolve()
    if folder.parent != root:
        raise LibraryError("ファイルが見つかりません。")
    return _inside(folder, f"{stage}.flac")


def ensure_reference_files() -> None:
    """学習中の確認用音声（Applio 付属）を models フォルダにコピーしておく。"""
    source = settings.APPLIO_DIR / "logs" / "reference"
    target = settings.MODELS_DIR / "reference"
    if source.is_dir() and not target.exists():
        shutil.copytree(source, target)
