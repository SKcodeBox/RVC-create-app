"""時間のかかる処理（学習・変換）を「ジョブ」として順番に実行する。

- 学習と変換は別々の列に並び、それぞれ1件ずつ実行する（学習中でも変換はできる）
- Applio のスクリプトは別プロセスで動かし、出力をジョブのログとして記録する
- ジョブの状態は data/jobs/<id>.json に保存し、サーバーを再起動しても履歴が残る
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from . import procs, settings

ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
PROGRESS_BAR_RE = re.compile(r"\d+%\|")
ACTIVE = ("queued", "running")
KINDS = ("train", "convert")
MAX_JOBS_KEPT = 300


class JobCancelled(Exception):
    pass


class JobFailed(Exception):
    """利用者に見せるメッセージを持つ失敗。"""


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"  # queued / running / done / error / cancelled / interrupted
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stage: str | None = None
    stage_label: str | None = None
    progress: float | None = None  # 全体の進み具合 0〜1（わからないときは None）
    detail: dict = field(default_factory=dict)
    message: str | None = None
    result: dict | None = None
    pid: int | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=500))
    # 実行中だけ使う値（保存しない）
    proc: subprocess.Popen | None = field(default=None, repr=False)
    # 学習と並行して動かす補助のプロセス（学習中のモデルの評価など）
    helpers: set = field(default_factory=set, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_log(self, line: str) -> None:
        with self.lock:
            # 進捗バーの行が続くときは上書きして、ログが埋まらないようにする
            if (
                self.logs
                and PROGRESS_BAR_RE.search(line)
                and PROGRESS_BAR_RE.search(self.logs[-1])
            ):
                self.logs[-1] = line
            else:
                self.logs.append(line)

    def to_dict(self, log_lines: int | None = 120) -> dict:
        with self.lock:
            logs = list(self.logs)
        if log_lines is not None:
            logs = logs[-log_lines:] if log_lines > 0 else []
        return {
            "id": self.id,
            "kind": self.kind,
            "params": dict(self.params),
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress": self.progress,
            # 学習と並行して動く評価のスレッドも書き換えるので、写しを返す
            "detail": dict(self.detail),
            "message": self.message,
            "result": self.result,
            "logs": logs,
        }


Runner = Callable[[Job, "JobManager"], dict | None]


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._queues: dict[str, queue.Queue] = {kind: queue.Queue() for kind in KINDS}
        self._runners: dict[str, Runner] = {}
        self._last_saved: dict[str, float] = {}
        self._cleanups: dict[str, Callable[[Job], None]] = {}

    # ---- 起動と終了 ----

    def register(
        self, kind: str, runner: Runner, cleanup: Callable[[Job], None] | None = None
    ) -> None:
        self._runners[kind] = runner
        if cleanup:
            self._cleanups[kind] = cleanup

    def start(self) -> None:
        self._load()
        for kind in KINDS:
            threading.Thread(
                target=self._worker, args=(kind,), name=f"jobs-{kind}", daemon=True
            ).start()

    def shutdown(self) -> None:
        """サーバー終了時に、実行中のプロセスを止める。"""
        with self._lock:
            running = [job for job in self._jobs.values() if job.status == "running"]
        for job in running:
            job.cancel_event.set()
            _kill_all(job)
            job.status = "interrupted"
            job.message = "サーバーが停止したため中断しました。"
            job.finished_at = time.time()
            self.save(job, force=True)

    # ---- 参照 ----

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, kind: str | None = None, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if kind is None or j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def active_for_model(self, name: str) -> Job | None:
        """そのモデルを学習中（または学習待ち）のジョブ。"""
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.kind == "train"
                    and job.status in ACTIVE
                    and job.params.get("name") == name
                ):
                    return job
        return None

    def uses_model(self, name: str) -> bool:
        with self._lock:
            return any(
                job.status in ACTIVE
                and (job.params.get("name") == name or job.params.get("model") == name)
                for job in self._jobs.values()
            )

    # ---- 操作 ----

    def submit(self, kind: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self.save(job, force=True)
        self._queues[kind].put(job.id)
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            job.status = "cancelled"
            job.message = "キャンセルしました。"
            job.finished_at = time.time()
            self.save(job, force=True)
        elif job.status == "running":
            job.cancel_event.set()
            _kill_all(job)
        return job

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in ACTIVE:
            return False
        cleanup = self._cleanups.get(job.kind)
        if cleanup:
            cleanup(job)
        with self._lock:
            self._jobs.pop(job_id, None)
        (settings.JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)
        return True

    def run(
        self,
        job: Job,
        command: list[str],
        on_line: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
        cwd=None,
        helper: bool = False,
        log: bool = True,
    ) -> int:
        """スクリプトを実行し、出力をジョブのログに流す。終了コードを返す。

        env と cwd を省略すると、Applio のスクリプト用の設定で動かす。
        helper=True は、ジョブの本体（学習など）と並行して動かす補助の処理。
        log=False なら出力をログに残さない（on_line には渡す）。
        """
        if job.cancel_event.is_set():
            raise JobCancelled()
        process = subprocess.Popen(
            command,
            cwd=cwd or settings.APPLIO_DIR,
            env=env or settings.applio_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # 子プロセスもまとめて止められるよう、別のプロセスグループで動かす
            **procs.popen_options(),
        )
        if helper:
            job.helpers.add(process)
        else:
            job.proc = process
            job.pid = process.pid
        # 起動とほぼ同時にキャンセルされた場合も確実に止める
        if job.cancel_event.is_set():
            _kill(process)
        if not helper:
            self.save(job, force=True)
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = ANSI_RE.sub("", raw).rstrip()
                if not line.strip():
                    continue
                if log:
                    job.add_log(line)
                if on_line:
                    try:
                        on_line(line)
                    except Exception:  # 進捗の読み取りに失敗しても処理は続ける
                        pass
                self.save(job)
        finally:
            process.wait()
            if helper:
                job.helpers.discard(process)
            else:
                job.proc = None
                job.pid = None
        if job.cancel_event.is_set():
            raise JobCancelled()
        return process.returncode

    def save(self, job: Job, force: bool = False) -> None:
        """ジョブの状態をファイルに保存する（頻繁に呼ばれるので間引く）。"""
        now = time.time()
        if not force and now - self._last_saved.get(job.id, 0) < 3:
            return
        self._last_saved[job.id] = now
        data = job.to_dict(log_lines=200)
        data["pid"] = job.pid
        path = settings.JOBS_DIR / f"{job.id}.json"
        temp = path.with_name(f"{job.id}.{threading.get_ident()}.tmp")
        try:
            temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            pass

    # ---- 内部処理 ----

    def _worker(self, kind: str) -> None:
        while True:
            job_id = self._queues[kind].get()
            job = self.get(job_id)
            if job is None or job.status != "queued":
                continue
            job.status = "running"
            job.started_at = time.time()
            self.save(job, force=True)
            try:
                job.result = self._runners[kind](job, self)
                job.status = "done"
                job.progress = 1.0
            except JobCancelled:
                job.status = "cancelled"
                job.message = "キャンセルしました。"
            except JobFailed as error:
                job.status = "error"
                job.message = str(error)
            except Exception as error:
                job.status = "error"
                job.message = f"予期しないエラーが発生しました: {error}"
                job.add_log(traceback.format_exc())
            finally:
                job.finished_at = time.time()
                job.proc = None
                job.pid = None
                self.save(job, force=True)

    def _load(self) -> None:
        """保存されているジョブを読み込む。実行中だったものは「中断」にする。"""
        files = sorted(
            settings.JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime
        )
        for path in files[:-MAX_JOBS_KEPT]:
            path.unlink(missing_ok=True)
        for path in files[-MAX_JOBS_KEPT:]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(
                    id=data["id"],
                    kind=data["kind"],
                    params=data.get("params") or {},
                    status=data.get("status", "error"),
                    created_at=data.get("created_at") or time.time(),
                    started_at=data.get("started_at"),
                    finished_at=data.get("finished_at"),
                    stage=data.get("stage"),
                    stage_label=data.get("stage_label"),
                    progress=data.get("progress"),
                    detail=data.get("detail") or {},
                    message=data.get("message"),
                    result=data.get("result"),
                )
                job.logs.extend(data.get("logs") or [])
            except (OSError, ValueError, KeyError):
                continue
            if job.kind not in KINDS:
                continue
            if job.status in ACTIVE:
                _stop_orphan(data.get("pid"))
                job.status = "interrupted"
                job.message = "サーバーが停止したため中断しました。"
                job.finished_at = job.finished_at or time.time()
                self.save(job, force=True)
            self._jobs[job.id] = job


def _kill(process: subprocess.Popen | None) -> None:
    """プロセスとその子プロセスをまとめて止める。"""
    if process is None or process.poll() is not None:
        return
    procs.kill_tree(process.pid)


def _kill_all(job: Job) -> None:
    """ジョブの本体と、補助のプロセスをすべて止める。"""
    _kill(job.proc)
    for helper in list(job.helpers):
        _kill(helper)


def stop_main_process(job: Job) -> None:
    """ジョブの本体のプロセスだけを止める（キャンセル扱いにはしない。学習の自動停止に使う）。"""
    _kill(job.proc)


def _stop_orphan(pid: int | None) -> None:
    """前回サーバーが落ちたときに残った Applio のプロセスを止める。"""
    if not pid:
        return
    command = procs.command_line(pid)
    # 別のプロセスが同じ番号を使っている可能性があるので、このサーバーが起動したものか確かめる
    # （ジョブのプロセスは必ず rvc-server/.venv の Python をフルパスで起動している）
    if str(settings.APPLIO_DIR) not in command and str(settings.SERVER_DIR) not in command:
        return
    procs.kill_tree(pid)
