"""学習しすぎ（過学習）を見分けるための仕組み。

- 学習用の音声の一部（数%）を「検証用」として取り分け、学習には使わない
- 途中のモデルが保存されるたびに、検証用の音声をどれだけ再現できるかを測る（engine/evaluate.py）
  あわせて、試し聴き用の音声（決まった話し声をそのモデルの声にしたもの）を作る
- 検証用の誤差がしばらく良くならなければ、学習を自動で止める（設定で切り替え可能）
- いちばん誤差が小さかったエポックを「おすすめ」として、変換のときに最初から選ぶ
"""

import importlib.util
import json
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import settings

VALIDATION_FILE = "validation.json"
STUDIO_DIR = "studio"  # このアプリが作るファイル（評価結果・試し聴き用の音声）を置くフォルダ
EVAL_FILE = "evaluation.json"
SAMPLES_DIR = "samples"

# 検証用に取り分ける量（学習用の音声の約4%。最低3個・最大30個）
VALIDATION_RATIO = 0.04
MIN_VALIDATION = 3
MAX_VALIDATION = 30
# これより少ない音声（3秒ごとに区切った数）しかないときは取り分けない（学習用が減りすぎるため）
MIN_CLIPS_FOR_VALIDATION = 20

# 自動で止める条件: 誤差が「これまでで一番良い値」より 1% 以上良くならないまま、
# PATIENCE_EPOCHS エポック（保存の間隔の5回分より短くはしない）たったら止める
MIN_IMPROVEMENT = 0.01
PATIENCE_EPOCHS = 50
# 学習の最初のうちは止めない
MIN_EPOCHS = 40

CLIP_RE = re.compile(r"^(\d+)_(\d+)_(\d+)$")


def studio_dir(model_dir: Path) -> Path:
    return model_dir / STUDIO_DIR


def _clip_key(name: str):
    match = CLIP_RE.match(name)
    return tuple(int(part) for part in match.groups()) if match else (10**9, 0, 0, name)


def split_validation(model_dir: Path) -> list[str]:
    """学習用の一覧（filelist.txt）から検証用の音声を取り分ける。取り分けた名前を返す。

    すでに取り分けてある（続きから学習する）場合は何もしない。
    """
    validation_path = model_dir / VALIDATION_FILE
    if validation_path.is_file():
        try:
            return json.loads(validation_path.read_text(encoding="utf-8"))["clips"]
        except (OSError, ValueError, KeyError):
            return []
    filelist = model_dir / "filelist.txt"
    lines = filelist.read_text(encoding="utf-8").splitlines()
    sliced = (model_dir / "sliced_audios").resolve()
    own: dict[str, str] = {}
    for line in lines:
        wav = Path(line.split("|", 1)[0])
        if wav.parent.resolve() == sliced:
            own[wav.stem] = line
    names = sorted(own, key=_clip_key)
    count = len(names)
    if count < MIN_CLIPS_FOR_VALIDATION:
        chosen: list[str] = []
    else:
        size = min(MAX_VALIDATION, max(MIN_VALIDATION, round(count * VALIDATION_RATIO)))
        # 音声全体からまんべんなく選ぶ（同じ部分ばかりにならないように）
        positions = np.linspace(0, count - 1, size + 2)[1:-1]
        chosen = sorted({names[int(round(p))] for p in positions}, key=_clip_key)
    held = {own[name] for name in chosen}
    kept = [line for line in lines if line not in held]
    filelist.write_text("\n".join(kept) + "\n", encoding="utf-8")
    validation_path.write_text(
        json.dumps({"clips": chosen, "total_clips": count}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chosen


def dataset_pitch(model_dir: Path) -> dict | None:
    """学習用の音声の声の高さ（Hz）。特徴の抽出で作られる f0_voiced/*.npy から計算する。"""
    values = []
    for path in (model_dir / "f0_voiced").glob("*.npy"):
        try:
            f0 = np.load(path)
        except (OSError, ValueError):
            continue
        values.append(f0[f0 > 0])
    if not values:
        return None
    voiced = np.concatenate(values)
    if len(voiced) < 100:
        return None
    return {
        "median": round(float(np.median(voiced)), 2),
        "p10": round(float(np.percentile(voiced, 10)), 2),
        "p90": round(float(np.percentile(voiced, 90)), 2),
    }


# ------------------------------------------------------------------ 評価結果の保存と読み出し


def eval_path(model_dir: Path) -> Path:
    return studio_dir(model_dir) / EVAL_FILE


def sample_path(model_dir: Path, epoch: int) -> Path:
    return studio_dir(model_dir) / SAMPLES_DIR / f"epoch_{epoch:05d}.flac"


def read_evaluations(model_dir: Path) -> list[dict]:
    """評価結果（エポックの順）。"""
    try:
        data = json.loads(eval_path(model_dir).read_text(encoding="utf-8"))
        items = data.get("evaluations") if isinstance(data, dict) else None
    except (OSError, ValueError):
        items = None
    items = [item for item in (items or []) if isinstance(item, dict) and "epoch" in item]
    return sorted(items, key=lambda item: item["epoch"])


def write_evaluations(model_dir: Path, items: list[dict]) -> None:
    path = eval_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps({"evaluations": sorted(items, key=lambda i: i["epoch"])}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def forget_epochs(model_dir: Path, epochs: set[int]) -> None:
    """モデルファイルを消したエポックの評価結果と試し聴き用の音声を消す。"""
    items = read_evaluations(model_dir)
    write_evaluations(model_dir, [item for item in items if item["epoch"] not in epochs])
    for epoch in epochs:
        sample_path(model_dir, epoch).unlink(missing_ok=True)


def best_evaluation(items: list[dict], available_epochs: set[int] | None = None) -> dict | None:
    """検証用の誤差がいちばん小さいもの（モデルファイルが残っているものの中から）。"""
    scored = [
        item for item in items
        if item.get("val_loss") is not None
        and (available_epochs is None or item["epoch"] in available_epochs)
    ]
    return min(scored, key=lambda item: item["val_loss"]) if scored else None


def should_stop(items: list[dict], save_every: int, min_epochs: int = MIN_EPOCHS) -> tuple[bool, int | None]:
    """自動で止めるべきか。(止めるか, 最後に「はっきり良くなった」エポック) を返す。"""
    scored = [item for item in items if item.get("val_loss") is not None]
    if not scored:
        return False, None
    best_value = None
    best_epoch = None
    for item in scored:
        if best_value is None or item["val_loss"] < best_value * (1 - MIN_IMPROVEMENT):
            best_value, best_epoch = item["val_loss"], item["epoch"]
    latest = scored[-1]["epoch"]
    patience = max(PATIENCE_EPOCHS, save_every * 5)
    stop = latest >= min_epochs and best_epoch is not None and latest - best_epoch >= patience
    return stop, best_epoch


# ------------------------------------------------------------------ 学習と並行して評価する


class Evaluator:
    """学習中に保存されたモデルを、順番に評価する（学習のスレッドとは別のスレッドで動かす）。"""

    def __init__(self, job, jm, name: str, model_dir: Path, save_every: int,
                 embedder: str, target_f0: float | None, on_result) -> None:
        self.job = job
        self.jm = jm
        self.name = name
        self.model_dir = model_dir
        self.save_every = save_every
        self.embedder = embedder
        self.target_f0 = target_f0
        self.on_result = on_result
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        # 学習のプロセスが終わったら立てる（それ以降は、まだないモデルファイルを長く待たない）
        self.stopped = threading.Event()
        self._done = {item["epoch"] for item in read_evaluations(model_dir)}
        self._thread = threading.Thread(target=self._run, name=f"eval-{name}", daemon=True)
        self._thread.start()

    def submit(self, epoch: int) -> None:
        with self._lock:
            if epoch in self._done:
                return
            self._done.add(epoch)
        self._queue.put(epoch)

    def finish(self, timeout: float = 600) -> None:
        """残っている評価が終わるのを待つ。"""
        self._queue.put(None)
        self._thread.join(timeout)

    def _weight_for(self, epoch: int, wait_seconds: float) -> Path | None:
        """そのエポックのモデルファイルが書き終わるのを待つ。"""
        deadline = time.time() + wait_seconds
        stopped_at = None
        last_size = -1
        while time.time() < deadline:
            if self.job.cancel_event.is_set():
                return None
            if self.stopped.is_set():
                stopped_at = stopped_at or time.time()
                if time.time() - stopped_at > 10:
                    deadline = 0
            matches = sorted(self.model_dir.glob(f"{self.name}_{epoch}e_*s.pth"))
            if matches:
                size = matches[-1].stat().st_size
                if size > 0 and size == last_size:
                    return matches[-1]
                last_size = size
            time.sleep(2)
        return None

    def _run(self) -> None:
        while True:
            epoch = self._queue.get()
            if epoch is None:
                return
            weight = self._weight_for(epoch, 300)
            if weight is None:
                continue
            try:
                result = self.evaluate(weight, epoch)
            except Exception as error:  # 評価に失敗しても、学習は続ける
                self.job.add_log(f"[RVC Studio] {epoch}エポックのモデルを評価できませんでした: {error}")
                continue
            if result is not None:
                self.on_result(result)

    def evaluate(self, weight: Path, epoch: int) -> dict | None:
        found: dict = {}

        def on_line(line: str) -> None:
            if line.startswith("@@eval "):
                found.update(json.loads(line[len("@@eval "):]))

        sample = sample_path(self.model_dir, epoch)
        command = [
            settings.PYTHON, str(settings.ENGINE_DIR / "evaluate.py"),
            "--model-dir", str(self.model_dir),
            "--weight", str(weight),
            "--epoch", str(epoch),
            "--sample", str(sample),
            "--embedder", self.embedder,
        ]
        if self.target_f0:
            command += ["--target-f0", str(self.target_f0)]
        code = self.jm.run(self.job, command, on_line=on_line, helper=True, log=False)
        if code != 0 or not found:
            return None
        found["evaluated_at"] = time.time()
        with self._lock:
            items = [item for item in read_evaluations(self.model_dir) if item["epoch"] != epoch]
            items.append(found)
            write_evaluations(self.model_dir, items)
        return found


# ------------------------------------------------------------------ 学習中の記録（TensorBoard）


_curve_cache: dict[str, tuple[float, dict]] = {}

CURVE_TAGS = {
    "loss/g/mel": "mel",
    "loss/g/total": "generator",
    "loss/d/adv": "discriminator",
}


# TensorBoard の記録のうち、これより大きいもの（試し聴きの音声・画像）は読まずに飛ばす。
# 数値（誤差）の記録は1件 100 バイトほど。音声が記録の大半を占め、全部読むと数秒かかるため
_LARGE_RECORD = 16 * 1024


def _scalar_events(path: Path):
    """TensorBoard の記録ファイル（TFRecord 形式）から、小さな記録（数値）だけを (タグ, ステップ, 値) で取り出す。"""
    from tensorboard.compat.proto import event_pb2

    with path.open("rb") as handle:
        while True:
            header = handle.read(12)  # 長さ（8バイト）と、その検査用の値（4バイト）
            if len(header) < 12:
                return
            length = int.from_bytes(header[:8], "little")
            if length > _LARGE_RECORD:
                handle.seek(length + 4, 1)
                continue
            data = handle.read(length)
            handle.read(4)
            if len(data) < length:  # 学習中で、書きかけの記録
                return
            try:
                event = event_pb2.Event.FromString(data)
            except Exception:  # 壊れた記録があれば、そこで読むのをやめる
                return
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    yield value.tag, event.step, value.simple_value


def training_curve(model_dir: Path, steps_per_epoch: int | None) -> dict:
    """Applio が学習中に記録した値（TensorBoard の形式）を、エポックごとの値にして返す。"""
    log_dir = model_dir / "eval"
    files = sorted(log_dir.glob("events.out.tfevents.*")) if log_dir.is_dir() else []
    if not files:
        return {}
    stamp = sum(path.stat().st_size for path in files) + len(files)
    key = str(model_dir)
    cached = _curve_cache.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    if importlib.util.find_spec("tensorboard") is None:
        return {}
    series: dict[str, dict[int, float]] = {name: {} for name in CURVE_TAGS.values()}
    for path in files:
        try:
            for tag, step, value in _scalar_events(path):
                name = CURVE_TAGS.get(tag)
                if name is None:
                    continue
                epoch = round(step / steps_per_epoch) if steps_per_epoch else step
                series[name][int(epoch)] = float(value)
        except OSError:
            continue
    curve = {
        name: [[epoch, value] for epoch, value in sorted(points.items()) if np.isfinite(value)]
        for name, points in series.items()
        if points
    }
    _curve_cache[key] = (stamp, curve)
    return curve
