import subprocess
import sys
import time

import psutil

from rvc_api import procs

# 子プロセスを1つ作ってから眠り続けるスクリプト
SCRIPT = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(child.pid, flush=True)
time.sleep(120)
"""


def test_kill_tree_stops_children():
    process = subprocess.Popen(
        [sys.executable, "-c", SCRIPT], stdout=subprocess.PIPE, text=True, **procs.popen_options()
    )
    child_pid = int(process.stdout.readline())
    assert psutil.pid_exists(child_pid)
    procs.kill_tree(process.pid, grace_seconds=5, wait=True)
    process.wait(timeout=10)
    deadline = time.time() + 10
    while time.time() < deadline and psutil.pid_exists(child_pid):
        try:
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.2)
    alive = psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE
    assert not alive


def test_kill_tree_ignores_missing():
    procs.kill_tree(None)
    procs.kill_tree(999_999_999)


def test_command_line():
    assert "python" in procs.command_line(psutil.Process().pid).lower()
    assert procs.command_line(999_999_999) == ""
