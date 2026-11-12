"""子プロセスの起動と停止（Windows と macOS / Linux の違いを吸収する）。

学習などのスクリプトは、さらに子プロセス（データの読み込み担当など）を作る。
止めるときは、子プロセスもまとめて止める。
"""

import subprocess
import threading

import psutil

from . import settings


def popen_options() -> dict:
    """subprocess.Popen に渡す、プロセスをまとめて止められるようにするための設定。"""
    if settings.IS_WINDOWS:
        # 新しいプロセスグループにして、Ctrl+C が二重に届かないようにする。コンソール窓は開かない
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        }
    # 別のセッション（プロセスグループ）で動かす
    return {"start_new_session": True}


def _tree(pid: int) -> list[psutil.Process]:
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []
    try:
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    return [parent, *children]


def kill_tree(pid: int | None, grace_seconds: float = 15.0, wait: bool = False) -> None:
    """プロセスとその子プロセスを止める。まず穏やかに止め、止まらなければ強制終了する。

    wait=False なら、強制終了までの待ち時間はバックグラウンドで待つ。
    """
    if not pid:
        return
    processes = _tree(pid)
    if not processes:
        return
    for process in processes:
        try:
            process.terminate()  # macOS / Linux は SIGTERM、Windows は終了させる
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def finish() -> None:
        _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
        for process in alive:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    if wait:
        finish()
    else:
        threading.Thread(target=finish, daemon=True).start()


def command_line(pid: int) -> str:
    """プロセスの起動コマンド（調べられなければ空文字）。"""
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return ""
