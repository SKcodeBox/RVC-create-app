"""学習と変換の手順。Applio のスクリプトを順番に実行する。

Applio の core.py は途中で失敗しても終了コード 0 を返すことがあるため、
学習では各スクリプトを直接呼び、できあがったファイルがあるかどうかで成否を判断する。
"""

import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

from . import audio, evaluation, library, mixing, pretrains, separation, settings, system
from . import sources as url_sources
from .jobs import Job, JobCancelled, JobFailed, JobManager, stop_main_process

PYTHON = settings.PYTHON

# yt-dlp の進捗（例: "[download]  45.3% of 12.34MiB ..."）
DOWNLOAD_PERCENT_RE = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
# 動画の音声の雑音を減らす強さ（dB）。大きいほど強く減らす（Applio の初期値は 12）。
# Applio の DeepFilterNet を使う。空調音などの雑音は約20dB減るが、BGM（音楽）は数dBしか減らない
CLEANUP_STRENGTH = 20
CLEANUP_CODE = (
    "import sys\n"
    "from rvc.lib.tools.audio_cleanup import cleanup_file\n"
    "cleanup_file(sys.argv[1], sys.argv[2], float(sys.argv[3]))\n"
)

# 学習ログの読み取り（例: "my-voice | epoch=3 | step=120 | batch=5/40"）
BATCH_RE = re.compile(r"\| epoch=(\d+) \| step=(\d+) \| batch=(\d+)/(\d+)")
# 1エポック終わるごとのログ（例: "... | time=12:00:00 | training_speed=0:00:35"）
EPOCH_RE = re.compile(
    r"\| epoch=(\d+) \| step=(\d+) \| time=[^|]*\| training_speed=(\d+):(\d+):(\d+)"
)
PERCENT_RE = re.compile(r"(\d+)%\|")
OOM_MARKERS = ("out of memory", "mps backend out of memory")
# 学習用の音声がこれより長いと、NVIDIA の GPU のメモリに学習データを置かない（秒）
CACHE_IN_GPU_MAX_SECONDS = 30 * 60

# 段階ごとの表示名と、全体の進み具合に占めるおおよその割合
STAGES = {
    # 学習
    "download": ("動画から音声を取り出しています", 0.06),
    "sepmodels": ("BGM・伴奏を取り除くための AI モデルをダウンロードしています（初回のみ）", 0.03),
    "separate": ("BGM・伴奏と声を分けています", 0.08),
    "cleanup": ("雑音を減らしています", 0.06),
    "pretrained": ("事前学習モデルをダウンロードしています（初回のみ）", 0.04),
    "preprocess": ("音声を整えて切り分けています", 0.05),
    "extract": ("声の特徴を取り出しています", 0.10),
    "train": ("学習しています", 0.80),
    "evaluate": ("最後に保存したモデルを評価しています", 0.01),
    "index": ("検索用インデックスを作っています", 0.03),
    # 変換
    "prepare": ("音声を読み込んでいます", 0.02),
    "song_separate": ("曲を声と伴奏に分けています", 0.50),
    "pitch": ("声の高さを調べています", 0.04),
    "convert": ("変換しています", 0.30),
    "convert_backing": ("ハモリを変換しています", 0.15),
    "polish": ("変換した声の雑音を減らしています", 0.04),
    "mix": ("声と伴奏を合わせています", 0.04),
    "export": ("書き出しています", 0.02),
}


def applio_script(*parts: str) -> str:
    return str(settings.APPLIO_DIR.joinpath(*parts))


class StageTracker:
    """いまの段階と、全体の進み具合（0〜1）を管理する。"""

    def __init__(self, job: Job, keys: list[str]) -> None:
        self.job = job
        total = sum(STAGES[key][1] for key in keys) or 1
        self.spans: dict[str, tuple[float, float]] = {}
        start = 0.0
        for key in keys:
            width = STAGES[key][1] / total
            self.spans[key] = (start, width)
            start += width

    def enter(self, key: str) -> None:
        self.job.stage = key
        self.job.stage_label = STAGES[key][0]
        self.update(0.0)

    def update(self, fraction: float) -> None:
        span = self.spans.get(self.job.stage or "")
        if span is None:
            return
        start, width = span
        fraction = min(max(fraction, 0.0), 1.0)
        self.job.progress = round(start + width * fraction, 4)

    def percent_reader(self):
        """進捗バー（"45%|..."）の数字を読み取って反映する。"""

        def on_line(line: str) -> None:
            match = PERCENT_RE.search(line)
            if match:
                self.update(int(match[1]) / 100)

        return on_line


# ------------------------------------------------------------------ 学習


def run_training(job: Job, jm: JobManager) -> dict:
    params = job.params
    name = params["name"]
    sample_rate = int(params["sample_rate"])
    total_epochs = int(params["total_epochs"])
    batch_size = int(params["batch_size"])
    save_every = int(params["save_every"])
    resume = bool(params.get("resume"))
    embedder = params.get("embedder") or "contentvec"
    pretrained = params.get("pretrained") or pretrains.STANDARD
    early_stop = bool(params.get("early_stop", True))
    quality = params.get("separation_quality") or separation.DEFAULT_QUALITY

    exp_dir = library.model_dir(name)
    data_dir = library.dataset_dir(name)
    cores = str(settings.worker_cores())
    pretrained_g, pretrained_d = pretrains.paths(pretrained, sample_rate)
    has_pretrained = pretrained_g.is_file() and pretrained_d.is_file()
    video_sources = [] if resume else (params.get("sources") or [])
    # 音声の下ごしらえ（BGM・伴奏の除去など。rvc_api/separation.py を参照）
    prep = [] if resume else list(params.get("prep") or [])
    model_steps = [step for step in separation.model_steps() if step in prep]
    to_download = [step for step in model_steps if not separation.is_downloaded(step, quality)]
    denoise = separation.DENOISE in prep

    keys = []
    if video_sources:
        keys.append("download")
    if to_download:
        keys.append("sepmodels")
    if model_steps:
        keys.append("separate")
    if denoise:
        keys.append("cleanup")
    if not has_pretrained:
        keys.append("pretrained")
    if not resume:
        keys += ["preprocess", "extract"]
    keys += ["train", "evaluate", "index"]
    stages = StageTracker(job, keys)

    # 0. 学習用の音声を用意する（動画の取り込みと、音声の下ごしらえ）
    if video_sources:
        stages.enter("download")
        _download_sources(job, jm, stages, name, video_sources)
    voiced = None
    if model_steps:
        if to_download:
            stages.enter("sepmodels")
            _download_separation_models(job, jm, stages, to_download, quality)
        stages.enter("separate")
        voiced = _separate_dataset(job, jm, stages, name, model_steps, quality)
    if denoise:
        stages.enter("cleanup")
        _denoise_dataset(job, jm, stages, name)
    if video_sources or prep:
        dataset = library.dataset_files(name)
        total_seconds = sum(item["duration"] or 0 for item in dataset)
        job.params["dataset_files"] = len(dataset)
        job.params["dataset_duration"] = round(total_seconds, 1)
        if voiced is not None:
            job.params["voiced_duration"] = round(voiced, 1)
            if voiced < 10:
                raise JobFailed(
                    f"BGM・伴奏を取り除いたところ、声がほとんど残りませんでした（約{voiced:.0f}秒）。"
                    "声がはっきり入っている音声を使ってください。"
                )
        elif total_seconds < 10:
            raise JobFailed(
                "取り込んだ音声が短すぎます。合計10秒以上が必要です（10分以上がおすすめです）。"
            )

    # 1. 事前学習モデル（初めて使うものだけダウンロード）
    if not has_pretrained:
        stages.enter("pretrained")
        _download_pretrained(job, jm, stages, pretrained, sample_rate)

    if not resume:
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 2. 前処理: 音声を整えて短く切り分ける
        stages.enter("preprocess")
        code = jm.run(
            job,
            [
                PYTHON,
                applio_script("rvc", "train", "preprocess", "preprocess.py"),
                str(exp_dir),
                str(data_dir),
                str(sample_rate),
                cores,
                "Automatic",  # 無音の部分で自動的に切り分ける
                "False",  # 低音カットのフィルターは使わない
                "3.0",  # 切り分ける長さ（秒）
                "0.3",  # 前後の重なり（秒）
                "post",  # 音量をそろえる
            ],
            on_line=stages.percent_reader(),
        )
        if code != 0 or not any((exp_dir / "sliced_audios").glob("*.wav")):
            raise JobFailed(
                "音声の前処理に失敗しました。音声に声が入っているか（無音や、ごく小さい音になっていないか）、"
                "ファイルが壊れていないか確認してください。"
            )

        # 3. 特徴の抽出: 声の高さ（ピッチ）と声の特徴を取り出す
        stages.enter("extract")
        code = jm.run(
            job,
            [
                PYTHON,
                applio_script("rvc", "train", "extract", "extract.py"),
                str(exp_dir),
                "rmvpe",  # ピッチの検出方法
                cores,
                "0",  # GPU を使う（Mac では MPS）
                str(sample_rate),
                embedder,  # 声の特徴を取り出すモデル
                "None",
                "2",  # 無音の学習データを2つ混ぜる
            ],
            on_line=stages.percent_reader(),
        )
        # Applio は特徴の抽出に失敗しても「成功」と表示して先へ進むため、結果のファイルで確かめる
        sliced = len(list((exp_dir / "sliced_audios").glob("*.wav")))
        extracted = len(list((exp_dir / "extracted").glob("*.npy")))
        if code != 0 or not (exp_dir / "filelist.txt").is_file() or extracted == 0:
            raise JobFailed("声の特徴の抽出に失敗しました。詳しくはログを確認してください。")
        if extracted < sliced:
            job.add_log(
                f"[RVC Studio] 注意: 特徴を取り出せたのは {sliced} 個中 {extracted} 個です。"
            )

        # 4. 学習しすぎを見分けるために、検証用の音声を取り分ける。声の高さも記録しておく
        held = evaluation.split_validation(exp_dir)
        pitch = evaluation.dataset_pitch(exp_dir)
        library.write_meta(name, pitch=pitch, validation_clips=len(held))
        if held:
            job.add_log(
                f"[RVC Studio] 学習しすぎを見分けるため、音声のうち {len(held)}個（約{len(held) * 3}秒）を"
                "検証用に取り分けました（学習には使いません）。"
            )
        else:
            job.add_log(
                "[RVC Studio] 音声が短いため、検証用の音声は取り分けません。"
                "学習しすぎの自動停止と、おすすめのエポックの判定は行いません（試し聴きで選んでください）。"
            )

    # 5. 学習
    # 前回、保存の途中で止まったときの書きかけのファイルを消す
    for leftover in exp_dir.glob("*.pth.tmp"):
        leftover.unlink(missing_ok=True)
    if resume and not library.checkpoint_state(name)["ok"]:
        raise JobFailed(
            "再開用のファイルが壊れているため、続きから学習できません（前回、保存の途中で止まった可能性があります）。"
            "今あるモデルはそのまま変換に使えます。"
        )
    start_epoch = library.checkpoint_epoch(name) if resume else 0
    validation_clips = int(library.read_meta(name).get("validation_clips") or 0)
    use_early_stop = early_stop and validation_clips > 0
    job.detail.update(
        {
            "start_epoch": start_epoch,
            "epoch": start_epoch,
            "current_epoch": start_epoch + 1,
            "total_epochs": total_epochs,
            "early_stop": use_early_stop,
            "validation_clips": validation_clips,
        }
    )
    _publish_evaluations(job, exp_dir)
    stages.enter("train")
    epoch_seconds: deque[int] = deque(maxlen=5)
    state = {"oom": False, "too_little": False, "stop_requested": False, "stopped": False, "stop_epoch": None,
             "restarted": False, "started": False}
    span = max(total_epochs - start_epoch, 1)

    def on_result(result: dict) -> None:
        items = _publish_evaluations(job, exp_dir)
        value = result.get("val_loss")
        best = evaluation.best_evaluation(items)
        if value is not None:
            note = "（これまでで一番良い値）" if best and best["epoch"] == result["epoch"] else ""
            job.add_log(f"[RVC Studio] {result['epoch']}エポック: 検証用の誤差 {value:.3f}{note}")
        if use_early_stop and not state["stop_requested"]:
            stop, last_good = evaluation.should_stop(items, save_every)
            if stop:
                state.update(stop_requested=True, stop_epoch=result["epoch"])
                job.add_log(
                    f"[RVC Studio] {last_good}エポックのあと、検証用の誤差がはっきり良くならなくなりました。"
                    "これ以上は学習しすぎになるため、学習を止めます。"
                )

    evaluator = evaluation.Evaluator(
        job, jm, name, exp_dir, save_every, embedder, library.target_pitch(name), on_result
    )
    if resume:
        # 前回、評価が終わる前に止まったエポックがあれば、ここで評価する
        evaluated = {item["epoch"] for item in evaluation.read_evaluations(exp_dir)}
        for weight in library.list_weights(name):
            if weight["epoch"] and weight["epoch"] not in evaluated:
                evaluator.submit(weight["epoch"])

    def on_train_line(line: str) -> None:
        if resume and line.startswith("Loaded pretrained (G)") and not state["restarted"]:
            # Applio は再開用のファイルを読めないと、何も言わずに最初から学習し直してしまうので止める
            state["restarted"] = True
            stop_main_process(job)
            return
        if any(marker in line.lower() for marker in OOM_MARKERS):
            state["oom"] = True
        if "Not enough data present in the training set" in line:
            state["too_little"] = True
        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match[1])
            epoch_seconds.append(
                int(match[3]) * 3600 + int(match[4]) * 60 + int(match[5])
            )
            average = sum(epoch_seconds) / len(epoch_seconds)
            job.detail.update(
                {
                    "epoch": epoch,
                    "current_epoch": min(epoch + 1, total_epochs),
                    "step": int(match[2]),
                    "batch": None,
                    "num_batches": None,
                    "epoch_seconds": round(average, 1),
                    "eta_seconds": round(average * max(total_epochs - epoch, 0)),
                }
            )
            stages.update((epoch - start_epoch) / span)
            saved = epoch % save_every == 0 or epoch == total_epochs
            if saved:
                evaluator.submit(epoch)
            elif state["stop_requested"] and not state["stopped"]:
                # 保存しないエポックの終わりは、ファイルを書いている途中ではないので安全に止められる
                _stop_training(job, state)
            return
        match = BATCH_RE.search(line)
        if match:
            if not state["started"]:
                # 学習が実際に進み始めたら「最後まで終わっていない」印を付ける（終わったら時刻を入れる）。
                # 途中で止まったモデルを見分けるため。始まる前に失敗したときは、前の状態のままにする
                state["started"] = True
                library.write_meta(name, finished_at=None)
            epoch, step, batch, num_batches = (int(value) for value in match.groups())
            job.detail.update(
                {
                    "epoch": epoch - 1,
                    "current_epoch": epoch,
                    "step": step,
                    "batch": batch,
                    "num_batches": num_batches,
                }
            )
            stages.update((epoch - 1 - start_epoch + batch / num_batches) / span)
            if state["stop_requested"] and not state["stopped"]:
                # 次のエポックの学習が始まっている ＝ 前のエポックの保存は終わっている
                _stop_training(job, state)

    duration = (library.model_summary(name) or {}).get("dataset_duration")
    cache_in_gpu = (
        system.device() == "cuda"
        and duration is not None
        and float(duration) <= CACHE_IN_GPU_MAX_SECONDS
    )
    try:
        jm.run(
            job,
            [
                PYTHON,
                applio_script("rvc", "train", "train.py"),
                name,
                str(save_every),
                str(total_epochs),
                str(pretrained_g),
                str(pretrained_d),
                "0",  # GPU を使う（Mac では MPS）
                str(batch_size),
                str(sample_rate),
                "True",  # 再開用のチェックポイントは最新のものだけ残す
                "True",  # 保存のたびに変換用のモデルを書き出す
                # NVIDIA の GPU なら、学習データを GPU のメモリに置いて速くする
                "True" if cache_in_gpu else "False",
                "False",  # 前回の学習結果を消さない（続きから学習できるように）
                "HiFi-GAN",
                "False",
            ],
            on_line=on_train_line,
        )
    finally:
        evaluator.stopped.set()

    if state["restarted"]:
        evaluator.finish(timeout=5)
        raise JobFailed(
            "再開用のファイルを読み込めなかったため、学習を止めました（そのまま進めると、最初から学習し直しになるため）。"
            "今あるモデルはそのまま変換に使えます。"
        )
    weights = library.list_weights(name)
    if state["stopped"]:
        if not weights:
            raise JobFailed("学習を自動で止めましたが、保存されたモデルがありませんでした。")
        library.write_meta(name, early_stopped_at=weights[-1]["epoch"])
    else:
        if not [w for w in weights if w["epoch"] == total_epochs]:
            evaluator.finish(timeout=5)
            if state["too_little"]:
                raise JobFailed(
                    "学習用の音声が少なすぎて、学習を始められませんでした。音声を増やすか、バッチサイズを小さくしてください。"
                )
            if state["oom"]:
                raise JobFailed(
                    "メモリが足りなくなり、学習が止まりました。バッチサイズを小さくして（例: 2）もう一度お試しください。"
                )
            hint = (
                "途中まで保存されているので、モデル一覧の「続きから学習」で再開できます。"
                if library.checkpoint_epoch(name)
                else ""
            )
            raise JobFailed(f"学習が途中で止まりました。詳しくはログを確認してください。{hint}")
        library.write_meta(name, early_stopped_at=None)

    # 6. 最後に保存したモデルの評価が終わるのを待つ
    stages.enter("evaluate")
    evaluator.finish()
    _publish_evaluations(job, exp_dir)

    # 7. インデックス作成（声の特徴を検索して、より本人らしい声にするためのもの）
    stages.enter("index")
    jm.run(
        job,
        [
            PYTHON,
            applio_script("rvc", "train", "process", "extract_index.py"),
            str(exp_dir),
            "Auto",
        ],
    )
    index = library.find_index(name)
    if index is None:
        job.message = "インデックスの作成に失敗しました（モデルはそのまま使えます）。"

    weights = library.list_weights(name)
    recommended = library.recommended_weight(name, weights)
    chosen = recommended or weights[-1]
    library.write_meta(name, finished_at=time.time())
    if state["stopped"]:
        job.message = (
            f"学習しすぎになる前に、{weights[-1]['epoch']}エポックで自動的に止めました。"
            + (job.message or "")
        )
        # これ以上学習すると悪くなるだけなので、続きから学習するためのファイル（約1.3GB）は残さない
        freed = library.cleanup(name, checkpoints=True)["freed"]
        if freed:
            job.add_log(
                f"[RVC Studio] 学習しすぎで止めたため、続きから学習するためのファイル（約{freed / 1e9:.1f}GB）を削除しました。"
            )
    return {
        "model": name,
        "weight": chosen["file"],
        "index": index.name if index else None,
        "recommended_epoch": recommended["epoch"] if recommended else None,
        "early_stopped_at": weights[-1]["epoch"] if state["stopped"] else None,
    }


def _stop_training(job: Job, state: dict) -> None:
    state["stopped"] = True
    stop_main_process(job)


def _publish_evaluations(job: Job, exp_dir: Path) -> list[dict]:
    """評価結果を、画面に出すための値（job.detail）に入れる。"""
    items = evaluation.read_evaluations(exp_dir)
    best = evaluation.best_evaluation(items)
    job.detail["evaluations"] = [
        {"epoch": item["epoch"], "val_loss": item.get("val_loss"), "sample": bool(item.get("sample"))}
        for item in items
    ]
    job.detail["best_epoch"] = best["epoch"] if best else None
    return items


def _download_pretrained(job: Job, jm: JobManager, stages: StageTracker, pretrained: str, sample_rate: int) -> None:
    if pretrained == pretrains.STANDARD:
        code = jm.run(
            job,
            [
                PYTHON,
                applio_script("rvc", "lib", "tools", "prerequisites_download.py"),
                "--vocoder", "HiFi-GAN",
                "--sample-rate", str(sample_rate),
            ],
        )
        if code != 0 or not pretrains.is_downloaded(pretrained, sample_rate):
            raise JobFailed(
                "事前学習モデルのダウンロードに失敗しました。インターネット接続を確認して、もう一度お試しください。"
            )
        return
    entry = pretrains.CATALOG[pretrained]
    job.add_log(f"[RVC Studio] 事前学習モデル「{entry['name']}」（{sample_rate // 1000}kHz）をダウンロードします（約1.3GB・初回のみ）")
    try:
        pretrains.download(pretrained, sample_rate, stages.update, job.cancel_event.is_set)
    except pretrains.PretrainedError as error:
        if job.cancel_event.is_set():
            raise JobCancelled()
        raise JobFailed(str(error))


def _download_sources(
    job: Job, jm: JobManager, stages: StageTracker, name: str, video_sources: list[dict]
) -> list[Path]:
    """動画の音声をダウンロードし、指定の範囲を学習用データとして保存する。"""
    work_dir = url_sources.work_dir_for(job.id)
    data_dir = library.dataset_dir(name)
    data_dir.mkdir(parents=True, exist_ok=True)
    count = len(video_sources)
    saved: list[Path] = []
    recorded: list[dict] = []
    # YouTube の仕様変更に追いつくため、yt-dlp を最新にしておく（1日に1回まで）
    url_sources.update_ytdlp(log=job.add_log)
    try:
        for number, source in enumerate(video_sources, start=1):
            stem = f"{number:02d}"
            job.add_log(f"[RVC Studio] {number}/{count} 本目を取り込みます: {source['url']}")

            def on_line(line: str, number: int = number) -> None:
                match = DOWNLOAD_PERCENT_RE.search(line)
                if match:
                    stages.update((number - 1 + float(match[1]) / 100) / count)

            command = url_sources.download_command(source["url"], str(work_dir / f"{stem}.%(ext)s"))
            code = jm.run(job, command, on_line=on_line)
            downloaded = url_sources.find_download(work_dir, stem)
            if (code != 0 or downloaded is None) and url_sources.update_may_help(list(job.logs)):
                # YouTube の仕様変更の可能性があるので、yt-dlp を更新してもう一度だけ試す
                if url_sources.update_ytdlp(force=True, log=job.add_log):
                    code = jm.run(job, command, on_line=on_line)
                    downloaded = url_sources.find_download(work_dir, stem)
            if code != 0 or downloaded is None:
                raise JobFailed(f"{number}本目: {url_sources.explain_failure(list(job.logs))}")
            info = url_sources.read_info(work_dir, stem)
            target = audio.unique_path(data_dir, f"video_{stem}.wav")
            try:
                seconds = url_sources.to_dataset_wav(
                    downloaded, target, source["start"], source["end"]
                )
            except (url_sources.SourceError, audio.AudioError) as error:
                raise JobFailed(f"{number}本目: {error}")
            title = info.get("title") or source["url"]
            job.add_log(f"[RVC Studio] 取り込みました: {title}（{seconds:.0f}秒）")
            saved.append(target)
            recorded.append({**source, "title": info.get("title"), "seconds": round(seconds, 1)})
            stages.update(number / count)
    finally:
        url_sources.remove_work_dir(job.id)
    # どの動画から作ったモデルかを記録しておく
    library.write_meta(name, sources=recorded)
    return saved


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    minutes, rest = divmod(total, 60)
    if minutes >= 60:
        return f"{minutes // 60}時間{minutes % 60}分"
    return f"{minutes}分{rest}秒" if minutes else f"{rest}秒"


def _run_separation(job: Job, jm: JobManager, command: list[str], on_line) -> int:
    """separation/separate.py を、ボーカル分離用の仮想環境の Python で動かす。"""
    return jm.run(
        job, command, on_line=on_line, env=settings.separation_env(), cwd=settings.SERVER_DIR
    )


def _explain_separation_failure(logs: list[str]) -> str:
    text = "\n".join(logs[-60:]).lower()
    if any(marker in text for marker in OOM_MARKERS):
        return (
            "メモリが足りなくなり、BGM・伴奏を取り除く処理が止まりました。"
            "ほかのアプリを閉じてから、もう一度お試しください。"
        )
    if any(marker in text for marker in ("failed to download", "connectionerror", "max retries")):
        return (
            "BGM・伴奏を取り除くための AI モデルをダウンロードできませんでした。"
            "インターネット接続を確認して、もう一度お試しください。"
        )
    return "BGM・伴奏を取り除く処理に失敗しました。詳しくはログを確認してください。"


def _download_separation_models(
    job: Job, jm: JobManager, stages: StageTracker, steps: list[str], quality: str
) -> None:
    """ボーカル分離の AI モデルをダウンロードする（初めて使うときだけ）。"""
    job.add_log(
        "[RVC Studio] BGM・伴奏を取り除くための AI モデルをダウンロードします"
        f"（初回のみ・約{separation.download_mb(steps, quality)}MB）"
    )
    position = {"number": 1, "count": len(steps)}

    def on_line(line: str) -> None:
        parts = line.split()
        if parts[0] == "@@download":
            position.update(number=int(parts[1]), count=int(parts[2]))
            stages.update((position["number"] - 1) / position["count"])
            return
        match = PERCENT_RE.search(line)
        if match:
            stages.update((position["number"] - 1 + int(match[1]) / 100) / position["count"])

    code = _run_separation(job, jm, separation.download_command(steps, quality), on_line)
    if code != 0 or not all(separation.is_downloaded(step, quality) for step in steps):
        raise JobFailed(_explain_separation_failure(list(job.logs)))


def _separation_reader(job: Job, stages: StageTracker, files: int, results: dict):
    """separate.py の進み具合（@@step など）を読み取る。"""
    position = {"step": 1, "steps": 1, "model": 1, "models": 1, "file": 1, "files": files, "label": ""}

    def update(fraction: float) -> None:
        in_model = (position["file"] - 1 + fraction) / position["files"]
        in_step = (position["model"] - 1 + in_model) / position["models"]
        stages.update((position["step"] - 1 + in_step) / position["steps"])
        label = position["label"]
        if label and position["models"] > 1:
            label += f"（{position['model']}/{position['models']}つ目の AI）"
        if label and position["files"] > 1:
            label += f"（{position['file']}/{position['files']}）"
        if label:
            job.stage_label = label

    def on_line(line: str) -> None:
        parts = line.split()
        if parts[0] == "@@step":
            position.update(
                step=int(parts[1]), steps=int(parts[2]), model=1, models=1, file=1,
                label=separation.label(parts[3]),
            )
            update(0)
        elif parts[0] == "@@model":
            position.update(model=int(parts[1]), models=int(parts[2]), file=1)
            update(0)
        elif parts[0] == "@@file":
            position.update(file=int(parts[1]), files=int(parts[2]))
            update(0)
        elif parts[0] == "@@result":
            results[int(parts[1])] = (float(parts[2]), float(parts[3]))
        else:
            match = PERCENT_RE.search(line)
            if match:
                update(int(match[1]) / 100)

    return on_line


def _separate_dataset(
    job: Job, jm: JobManager, stages: StageTracker, name: str, steps: list[str], quality: str
) -> float:
    """BGM・伴奏などを取り除き、学習用の音声を処理したものに置き換える。声が入っている秒数を返す。

    手順ごとの途中経過も残しておき、モデル一覧で聴き比べられるようにする。
    """
    files = library.dataset_paths(name)
    if not files:
        raise JobFailed("学習用の音声が見つかりません。")
    job.add_log(
        f"[RVC Studio] 音声 {len(files)}個を下ごしらえします: "
        + " → ".join(separation.name(step) for step in steps)
        + ("（BGM・伴奏の除去は高精度: 2つの AI の組み合わせ）" if "vocals" in steps and quality == "high" else "")
    )
    work_dir = settings.UPLOADS_DIR / f"separate-{job.id}"
    results: dict[int, tuple[float, float]] = {}
    data_dir = library.dataset_dir(name)
    stages_root = data_dir / library.STAGES_DIR
    try:
        code = _run_separation(
            job, jm,
            separation.separate_command(steps, files, work_dir, quality, keep_stages=True),
            _separation_reader(job, stages, len(files), results),
        )
        if code != 0 or len(results) != len(files):
            raise JobFailed(_explain_separation_failure(list(job.logs)))
        # 処理した音声（WAV）で、元の音声を置き換える。途中経過は .stages/<名前>/ に移す
        for number, original in enumerate(files, start=1):
            original.unlink(missing_ok=True)
            target = audio.unique_path(data_dir, f"{original.stem}.wav")
            shutil.move(str(work_dir / "final" / f"{number}.wav"), target)
            kept = work_dir / "stages" / str(number)
            if kept.is_dir():
                destination = stages_root / target.stem
                shutil.rmtree(destination, ignore_errors=True)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(kept), destination)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    voiced = sum(seconds for seconds, _ in results.values())
    total = sum(length for _, length in results.values())
    job.add_log(
        f"[RVC Studio] 声が入っている部分: 約{_format_seconds(voiced)}（全体 {_format_seconds(total)}）"
    )
    return voiced


def _denoise_dataset(job: Job, jm: JobManager, stages: StageTracker, name: str) -> None:
    """雑音（空調音・物音など）を減らす（Applio の DeepFilterNet）。"""
    paths = library.dataset_paths(name)
    for number, path in enumerate(paths, start=1):
        code = jm.run(job, [PYTHON, "-c", CLEANUP_CODE, str(path), "speech", str(CLEANUP_STRENGTH)])
        if code != 0:
            raise JobFailed(
                "雑音を減らす処理に失敗しました。音声の下ごしらえで「雑音を減らす」を使わない設定にすると、"
                "この処理をせずに学習できます。"
            )
        # 処理した結果は WAV で書き出されるので、拡張子を合わせる
        if path.suffix.lower() != ".wav":
            path.rename(audio.unique_path(path.parent, f"{path.stem}.wav"))
        stages.update(number / len(paths))


# ------------------------------------------------------------------ 変換


def auto_pitch_shift(source_hz: float, target_hz: float, octave_only: bool) -> int:
    """元の声の高さを、モデルの声の高さに近づけるための変化（半音）。

    曲（伴奏あり）はキーが変わらないよう、オクターブ（12半音）単位にする。
    """
    semitones = 12 * math.log2(target_hz / source_hz)
    shift = round(semitones / 12) * 12 if octave_only else round(semitones)
    return int(max(-24, min(24, shift)))


def _pitch_stats(job: Job, jm: JobManager, path: Path) -> dict | None:
    found: dict = {}

    def on_line(line: str) -> None:
        if line.startswith("@@pitch "):
            found.update(json.loads(line[len("@@pitch "):]))

    code = jm.run(job, [PYTHON, str(settings.ENGINE_DIR / "pitch_stats.py"), str(path)], on_line=on_line, log=False)
    return found if code == 0 and found else None


def _index_for(name: str, work_dir: Path) -> str:
    """変換に渡すインデックスのパス。

    Applio はパスの中の "trained" を "added" に置き換えてしまうため、含まれていれば別の場所にコピーを作る。
    （Windows ではシンボリックリンクに管理者の権限が要るため、ハードリンクかコピーにする）
    """
    index = library.find_index(name)
    if index is None:
        return ""
    if "trained" not in str(index):
        return str(index)
    for folder in (work_dir, Path(tempfile.gettempdir()) / "rvc-studio-index"):
        if "trained" in str(folder):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / f"voice-{os.getpid()}-{index.stat().st_size}.index"
        link.unlink(missing_ok=True)
        try:
            os.link(index, link)
        except OSError:
            shutil.copy2(index, link)
        return str(link)
    return str(index)


def _infer(job: Job, jm: JobManager, params: dict, model_path: Path, index_arg: str,
           embedder: str, source: Path, target: Path, pitch: int) -> None:
    command = [
        PYTHON,
        applio_script("core.py"),
        "infer",
        "--input-path", str(source),
        "--output-path", str(target),
        "--pth-path", str(model_path),
        "--index-path", index_arg,
        "--pitch", str(int(pitch)),
        "--index-rate", str(params["index_rate"] if index_arg else 0),
        "--volume-envelope", str(params["volume_envelope"]),
        "--protect", str(params["protect"]),
        "--f0-method", params["f0_method"],
        "--export-format", "WAV",
        "--embedder-model", embedder,
    ]
    if params.get("autotune"):
        command += ["--f0-autotune", "--f0-autotune-strength", str(params.get("autotune_strength", 1.0))]
    if params.get("formant"):
        command += [
            "--formant-shifting",
            "--formant-qfrency", str(params.get("formant_quefrency", 1.0)),
            "--formant-timbre", str(params.get("formant_timbre", 1.0)),
        ]
    if params.get("clean_input"):
        command += ["--cleanup-mode", _cleanup_mode(params), "--cleanup-strength", str(params.get("clean_strength", 12))]
    target.unlink(missing_ok=True)
    finished = {"done": False}

    def on_line(line: str) -> None:
        if line.startswith("Conversion completed at"):
            finished["done"] = True

    code = jm.run(job, command, on_line=on_line)
    if code != 0 and finished["done"] and audio.is_readable(target):
        # 変換を終えて書き出した後、プロセスの終わり際に異常終了することがある（ライブラリの後片付けなど）。
        # 結果のファイルは完成しているので、そのまま使う
        job.add_log(f"[RVC Studio] 変換の処理が終わり際に異常終了しました（終了コード {code}）が、結果は書き出せているので続けます。")
        code = 0
    if code != 0 or not target.is_file():
        raise JobFailed("変換に失敗しました。詳しくはログを確認してください。")


def _cleanup_mode(params: dict) -> str:
    # 歌声は "speech" だと消えてしまうことがあるため、曲のときは控えめな "light" にする
    return "light" if params.get("input_kind") == "song" else "speech"


def _polish(job: Job, jm: JobManager, params: dict, path: Path) -> None:
    """変換した声の雑音を減らす（Applio の DeepFilterNet）。"""
    code = jm.run(job, [PYTHON, "-c", CLEANUP_CODE, str(path), _cleanup_mode(params), str(params.get("clean_strength", 12))])
    if code != 0:
        job.add_log("[RVC Studio] 変換した声の雑音を減らす処理に失敗したため、そのままにします。")


# 変換の作業用ファイル（アップロードのフォルダの中）。元の音声（input.*）は聴き比べ用に残す
WORK_LEFTOVERS = ("separated", "converted.wav", "converted_backing.wav", "mix.wav")


def _remove_work_files(work_dir: Path) -> None:
    for leftover in WORK_LEFTOVERS:
        path = work_dir / leftover
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def run_conversion(job: Job, jm: JobManager) -> dict:
    """変換する。成功しても失敗・キャンセルしても、大きな作業用ファイル（分けた声・伴奏など）は消す。"""
    try:
        return _convert(job, jm)
    except BaseException:
        # 書きかけの結果を残さない
        for path in settings.OUTPUTS_DIR.glob(f"{job.id}*"):
            path.unlink(missing_ok=True)
        raise
    finally:
        _remove_work_files(Path(job.params["input_path"]).parent)


def _convert(job: Job, jm: JobManager) -> dict:
    params = job.params
    name = params["model"]
    model_path = library.model_dir(name) / params["weight"]
    if not model_path.is_file():
        raise JobFailed("モデルファイルが見つかりません。削除された可能性があります。")
    input_path = Path(params["input_path"])
    if not input_path.is_file():
        raise JobFailed("変換する音声ファイルが見つかりません。")
    work_dir = input_path.parent
    fmt = params.get("export_format") or "wav"
    song = params.get("input_kind") == "song"
    auto = params.get("pitch_mode") == "auto"
    meta = library.read_meta(name)
    summary = library.model_summary(name) or {}
    embedder = summary.get("embedder") or "contentvec"
    target_hz = library.target_pitch(name, meta)
    if auto and target_hz is None:
        raise JobFailed(
            "このモデルは声の高さがわからないため、自動で合わせられません。"
            "モデル一覧で「声の高さ」を設定するか、ピッチを手動で指定してください。"
        )
    index_arg = _index_for(name, work_dir) if float(params["index_rate"]) > 0 else ""
    harmony = params.get("harmony") or "together"
    quality = params.get("separation_quality") or separation.DEFAULT_QUALITY

    keys: list[str] = []
    song_steps: list[str] = []
    if song:
        song_steps = separation.parse_song_steps(bool(params.get("dereverb", True)), harmony != "together")
        keys.append("prepare")
        if any(not separation.is_downloaded(step, quality) for step in song_steps):
            keys.append("sepmodels")
        keys.append("song_separate")
    if auto:
        keys.append("pitch")
    keys.append("convert")
    if song and harmony == "convert":
        keys.append("convert_backing")
    if params.get("clean_output"):
        keys.append("polish")
    keys.append("mix" if song else "export")
    stages = StageTracker(job, keys)

    lead = input_path
    original_lead = instrumental = backing = None
    if song:
        stages.enter("prepare")
        mix_path = work_dir / "mix.wav"
        try:
            audio.ffmpeg_to_wav(input_path, mix_path, sample_rate=44100, channels=2)
        except audio.AudioError as error:
            raise JobFailed(str(error))
        to_download = [step for step in song_steps if not separation.is_downloaded(step, quality)]
        if to_download:
            stages.enter("sepmodels")
            _download_separation_models(job, jm, stages, to_download, quality)
        stages.enter("song_separate")
        sep_dir = work_dir / "separated"
        results: dict = {}
        code = _run_separation(
            job, jm, separation.song_command(song_steps, mix_path, sep_dir, quality),
            _separation_reader(job, stages, 1, results),
        )
        final = sep_dir / "final"
        lead = final / "1_lead.wav"
        instrumental = final / "1_instrumental.wav"
        if code != 0 or not lead.is_file() or not instrumental.is_file():
            raise JobFailed(_explain_separation_failure(list(job.logs)))
        original_lead = lead
        if harmony != "together":
            backing = final / "1_backing.wav"
            if not backing.is_file():
                raise JobFailed(_explain_separation_failure(list(job.logs)))
        mix_path.unlink(missing_ok=True)

    pitch = int(params["pitch"])
    source_hz = None
    if auto:
        stages.enter("pitch")
        stats = _pitch_stats(job, jm, lead)
        source_hz = (stats or {}).get("median")
        if source_hz:
            pitch = auto_pitch_shift(source_hz, target_hz, octave_only=song)
            job.add_log(
                f"[RVC Studio] 声の高さ: 元の声 {source_hz:.0f}Hz → モデル {target_hz:.0f}Hz。"
                f"ピッチを {pitch:+d} にします" + ("（曲なので、キーが変わらないようオクターブ単位）" if song else "")
            )
        else:
            pitch = 0
            job.add_log("[RVC Studio] 声の高さを調べられなかったため、ピッチは 0 のまま変換します。")
    job.detail["pitch_used"] = pitch

    stages.enter("convert")
    converted = work_dir / "converted.wav"
    _infer(job, jm, params, model_path, index_arg, embedder, lead, converted, pitch)
    if params.get("clean_output"):
        stages.enter("polish")
        _polish(job, jm, params, converted)

    converted_backing = None
    if song and harmony == "convert" and backing is not None:
        stages.enter("convert_backing")
        converted_backing = work_dir / "converted_backing.wav"
        _infer(job, jm, params, model_path, index_arg, embedder, backing, converted_backing, pitch)

    output_name = f"{job.id}.{fmt}"
    output_path = settings.OUTPUTS_DIR / output_name
    result: dict = {"output": output_name, "format": fmt, "pitch_used": pitch, "pitch_auto": auto,
                    "source_pitch": source_hz, "target_pitch": target_hz if auto else None, "stems": None}
    if song:
        stages.enter("mix")
        backing_for_mix = converted_backing if harmony == "convert" else (backing if harmony == "keep" else None)
        song_audio, vocals, music, rate, info = mixing.mix_song(
            converted, original_lead, instrumental, backing_for_mix,
            reverb=float(params.get("reverb", 0.3)),
            vocal_gain_db=float(params.get("vocal_gain_db", 0.0)),
            instrumental_gain_db=float(params.get("instrumental_gain_db", 0.0)),
        )
        audio.write_audio(output_path, song_audio, rate, fmt)
        stems = {}
        for stem, data in (("vocals", vocals), ("instrumental", music)):
            stem_name = f"{job.id}_{stem}.{fmt}"
            audio.write_audio(settings.OUTPUTS_DIR / stem_name, data, rate, fmt)
            stems[stem] = stem_name
        result["stems"] = stems
        result["mix"] = info
    else:
        stages.enter("export")
        data, rate = mixing.read(converted)
        audio.write_audio(output_path, data[:, 0] if data.shape[1] == 1 else data, rate, fmt)
    result["duration"] = audio.duration_of(output_path)
    return result


def cleanup_conversion(job: Job) -> None:
    """変換の履歴を消すときに、アップロードされた音声と変換結果も消す。"""
    for path in settings.OUTPUTS_DIR.glob(f"{job.id}*"):
        path.unlink(missing_ok=True)
    upload_id = job.params.get("upload_id")
    if upload_id:
        shutil.rmtree(settings.UPLOADS_DIR / upload_id, ignore_errors=True)


def sweep_orphan_files(jobs: list[Job]) -> int:
    """どのジョブの記録にも属さない変換結果と作業用フォルダを消し、空けた容量（バイト）を返す。

    サーバーの起動時（実行中のジョブがないとき）に呼ぶ。履歴が上限（300件）を超えて記録だけ消えた分や、
    前回サーバーが落ちたときに残った作業用ファイルが対象。
    """
    ids = {job.id for job in jobs}
    uploads = {job.params.get("upload_id") for job in jobs if job.kind == "convert"}
    freed = 0
    if settings.OUTPUTS_DIR.is_dir():
        for path in settings.OUTPUTS_DIR.iterdir():
            # 変換結果の名前は "<ジョブID>.wav" か "<ジョブID>_vocals.wav" など
            if path.is_file() and path.name.split(".", 1)[0].split("_", 1)[0] not in ids:
                freed += path.stat().st_size
                path.unlink(missing_ok=True)
    if settings.UPLOADS_DIR.is_dir():
        for path in settings.UPLOADS_DIR.iterdir():
            if path.name not in uploads:
                freed += library.folder_size(path)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
    return freed

