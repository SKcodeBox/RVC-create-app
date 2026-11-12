"""動画サイト（YouTube など）の URL から、学習用の音声を取り出す。

ダウンロードには yt-dlp を使う。YouTube の利用規約では、許可なく動画をダウンロードすることは
禁止されているため、画面側で「自分の声、または本人の許可がある」ことの確認を必須にしている。

yt-dlp は YouTube の仕様変更に合わせて頻繁に更新されるため、1日に1回まで自動で最新版にする
（ダウンロードに失敗したときは、その場で更新してからもう一度試す）。
"""

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf

from . import audio, settings

MAX_URLS = 10
# 長すぎる動画（長時間の配信など）は扱わない
MAX_VIDEO_SECONDS = 3 * 60 * 60

_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")

# yt-dlp のエラーを、利用者に分かる言葉に置き換える（上から順に調べるので、具体的なものを先に書く）
_ERROR_HINTS = (
    ("Unsupported URL", "この URL には対応していません。動画のページの URL を貼ってください。"),
    ("Private video", "非公開の動画のため、取り込めませんでした。"),
    ("Video unavailable", "動画を見られませんでした（削除・非公開・地域制限などの可能性があります）。"),
    ("Sign in to confirm", "YouTube へのログインが必要な動画のため、取り込めませんでした。"),
    ("members-only", "メンバー限定の動画のため、取り込めませんでした。"),
    ("JavaScript runtime", "YouTube の読み込みに必要な JavaScript の実行環境（Node.js）が見つかりません。"),
    ("HTTP Error 404", "ページが見つかりませんでした（404）。URL を確認してください。"),
    ("HTTP Error 403", "アクセスを拒否されました（403）。時間をおいてから試すか、npm run rvc:setup で yt-dlp を最新にしてください。"),
    ("HTTP Error 429", "短い時間にアクセスしすぎたため、断られました（429）。しばらく待ってから試してください。"),
    ("Unable to download webpage", "ページを読み込めませんでした。URL とインターネット接続を確認してください。"),
    ("Failed to resolve", "ページを読み込めませんでした。URL とインターネット接続を確認してください。"),
    ("Connection refused", "ページを読み込めませんでした。URL とインターネット接続を確認してください。"),
)


class SourceError(Exception):
    """利用者に見せるメッセージを持つエラー。"""


def parse_time(text) -> float | None:
    """「90」「1:30」「1:02:30」のような時刻を秒に直す。空なら None。"""
    text = str(text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3 or not all(_NUMBER_RE.match(part) for part in parts):
        raise SourceError(f"時刻「{text}」の書き方が正しくありません（例: 90、1:30、1:02:30）。")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_sources(raw: str | None) -> list[dict]:
    """画面から届いた URL の一覧（JSON）を確かめて、扱いやすい形にする。"""
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        raise SourceError("URL の一覧を読み取れませんでした。")
    if not isinstance(items, list):
        raise SourceError("URL の一覧を読み取れませんでした。")
    sources = []
    for number, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SourceError(f"{number}個目の URL が正しくありません（https:// から始まる URL を貼ってください）。")
        start = parse_time(item.get("start"))
        end = parse_time(item.get("end"))
        if start is not None and end is not None and end <= start:
            raise SourceError(f"{number}個目の終了時刻は、開始時刻より後にしてください。")
        sources.append({"url": url, "start": start, "end": end})
    if len(sources) > MAX_URLS:
        raise SourceError(f"URL は {MAX_URLS} 個までです。")
    return sources


def audio_format() -> str:
    """ダウンロードする音声の選び方（yt-dlp の -f）。"""
    if settings.ffmpeg_path():
        # いちばん音質の良い音声（YouTube ならたいてい Opus・約160kbps）。ffmpeg で WAV にする
        return "bestaudio/best"
    # ffmpeg がないときは、macOS の afconvert で WAV にできる m4a（AAC）を優先する
    return "bestaudio[ext=m4a]/bestaudio[acodec^=mp4a]/bestaudio/best"


def download_command(url: str, output_template: str) -> list[str]:
    """yt-dlp で音声だけをダウンロードするコマンド。"""
    command = [
        settings.PYTHON, "-m", "yt_dlp",
        # プレイリストの URL でも、最初の1本だけを取り込む
        "--no-playlist", "--playlist-items", "1",
        # 長すぎる動画とライブ配信中の動画は除外する
        "--match-filter", f"!is_live & duration <=? {MAX_VIDEO_SECONDS}",
        "-f", audio_format(),
        "--newline", "--no-mtime", "--write-info-json",
        "-o", output_template,
    ]
    node = shutil.which("node")
    if node:
        # YouTube の読み込みには JavaScript の実行環境が必要（初期設定の deno の代わりに Node.js を使う）
        command += ["--js-runtimes", f"node:{node}"]
    return [*command, url]


# 失敗したときに、yt-dlp を更新すれば直る可能性があるもの（YouTube の仕様変更など）
_UPDATE_MAY_HELP = (
    "HTTP Error 403",
    "Sign in to confirm",
    "Requested format is not available",
    "nsig",
    "Unable to extract",
    "Precondition check failed",
    "Some formats may be missing",
    "please report this issue",
)


def update_may_help(logs: list[str]) -> bool:
    text = "\n".join(logs[-40:])
    return any(marker in text for marker in _UPDATE_MAY_HELP)


_STAMP = "yt-dlp-updated"
_update_lock = threading.Lock()
UPDATE_INTERVAL = 24 * 60 * 60


def ytdlp_version() -> str | None:
    try:
        result = subprocess.run(
            [settings.PYTHON, "-m", "yt_dlp", "--version"],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def update_ytdlp(force: bool = False, log=None) -> bool:
    """yt-dlp を最新版にする（前回から1日たっていなければ何もしない）。更新を試みたら True。

    uv が見つからない・ネットにつながらないなどで失敗しても、例外は出さない。
    """
    say = log or (lambda _line: None)
    stamp = settings.STATE_DIR / _STAMP
    with _update_lock:
        try:
            last = stamp.stat().st_mtime
        except OSError:
            last = 0
        if not force and time.time() - last < UPDATE_INTERVAL:
            return False
        uv = settings.uv_path()
        if not uv:
            say("[RVC Studio] uv が見つからないため、yt-dlp を自動で更新できません（npm run rvc:setup で更新できます）。")
            return False
        before = ytdlp_version()
        say("[RVC Studio] yt-dlp（動画のダウンロードに使う部品）を最新版にしています…")
        try:
            result = subprocess.run(
                [
                    uv, "pip", "install", "--python", settings.PYTHON,
                    "yt-dlp[default]", "yt-dlp-ejs",
                    "--upgrade-package", "yt-dlp", "--upgrade-package", "yt-dlp-ejs",
                ],
                capture_output=True, text=True, timeout=300,
                env={**settings.applio_env(), "UV_HTTP_TIMEOUT": "120"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            say(f"[RVC Studio] yt-dlp を更新できませんでした（{error}）。")
            return True
        settings.STATE_DIR.mkdir(parents=True, exist_ok=True)
        stamp.touch()
        if result.returncode != 0:
            say("[RVC Studio] yt-dlp を更新できませんでした（インターネット接続を確認してください）。")
            return True
        after = ytdlp_version()
        if before != after:
            say(f"[RVC Studio] yt-dlp を更新しました: {before} → {after}")
        else:
            say(f"[RVC Studio] yt-dlp は最新版です（{after}）。")
        return True


def explain_failure(logs: list[str]) -> str:
    text = "\n".join(logs[-40:])
    if "does not pass filter" in text:
        return "3時間を超える動画や、ライブ配信中の動画は使えません。"
    for marker, message in _ERROR_HINTS:
        if marker in text:
            return message
    return "動画の音声をダウンロードできませんでした。URL を確認してください（詳しくはログ）。"


def find_download(directory: Path, stem: str) -> Path | None:
    for path in sorted(directory.glob(f"{stem}.*")):
        if path.name.endswith((".info.json", ".part", ".ytdl", ".temp")):
            continue
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def read_info(directory: Path, stem: str) -> dict:
    try:
        info = json.loads((directory / f"{stem}.info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {"title": info.get("title"), "duration": info.get("duration")}


def to_dataset_wav(downloaded: Path, target: Path, start: float | None, end: float | None) -> float:
    """ダウンロードした音声を WAV にし、指定の範囲だけを切り出して保存する。長さ（秒）を返す。"""
    wav = audio.make_readable(downloaded, mono=False)
    info = sf.info(str(wav))
    rate = info.samplerate
    first = int((start or 0) * rate)
    last = min(int(end * rate), info.frames) if end else info.frames
    if first >= info.frames:
        raise SourceError("開始時刻が、動画の長さを超えています。")
    if last - first < rate:
        raise SourceError("切り出した範囲が短すぎます（1秒以上にしてください）。")
    if first == 0 and last == info.frames:
        shutil.move(str(wav), target)
    else:
        data, _ = sf.read(str(wav), start=first, stop=last, dtype="int16")
        sf.write(str(target), data, rate, subtype="PCM_16")
        wav.unlink(missing_ok=True)
    return (last - first) / rate


def work_dir_for(job_id: str) -> Path:
    path = settings.UPLOADS_DIR / f"download-{job_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_work_dir(job_id: str) -> None:
    shutil.rmtree(settings.UPLOADS_DIR / f"download-{job_id}", ignore_errors=True)
