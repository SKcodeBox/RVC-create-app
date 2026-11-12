"""アップロードされた音声ファイルの確認と変換、変換結果の書き出し。

読み込みと形式の変換には ffmpeg（セットアップで入れる imageio-ffmpeg）を使う。
ffmpeg が使えない Mac では、macOS 付属の afconvert で代わりに変換する。
"""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import settings

# Applio がそのまま読める形式（学習の前処理が対応している拡張子）
NATIVE_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
# WAV に変換してから受け付ける形式（音声ファイル）
CONVERTIBLE_AUDIO_EXTS = {".m4a", ".aac", ".aif", ".aiff", ".caf", ".opus", ".wma", ".alac", ".webm"}
# 音声を取り出して受け付ける動画ファイル（ffmpeg が必要）
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
CONVERTIBLE_EXTS = CONVERTIBLE_AUDIO_EXTS | VIDEO_EXTS
ACCEPTED_EXTS = NATIVE_EXTS | CONVERTIBLE_EXTS
# afconvert（macOS 付属）で変換できる形式
AFCONVERT_EXTS = {".m4a", ".aac", ".aif", ".aiff", ".caf", ".mp4", ".mov", ".m4v"}

# 変換結果として書き出せる形式
EXPORT_FORMATS = {
    "wav": {"label": "WAV", "media_type": "audio/wav"},
    "flac": {"label": "FLAC", "media_type": "audio/flac"},
    "mp3": {"label": "MP3", "media_type": "audio/mpeg"},
}


class AudioError(Exception):
    """利用者に見せるメッセージを持つ、音声ファイルのエラー。"""


def safe_filename(name: str, fallback: str = "audio") -> str:
    """アップロードされたファイル名から、保存に使える安全な名前を作る。"""
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", base).strip().lstrip(".")
    stem, ext = Path(base).stem[:80].rstrip(" ."), Path(base).suffix.lower()
    # Windows で使えない名前（CON・NUL など）を避ける
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        stem = f"_{stem}"
    if not stem:
        stem = fallback
    return f"{stem}{ext}"


def unique_path(directory: Path, filename: str) -> Path:
    """同じ名前のファイルがあれば「名前_2」のように番号を付ける。"""
    candidate = directory / filename
    stem, ext = candidate.stem, candidate.suffix
    number = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{number}{ext}"
        number += 1
    return candidate


def accepted_extensions() -> set[str]:
    """いま受け付けられる拡張子（ffmpeg がなければ、変換できるものだけ）。"""
    if settings.ffmpeg_path():
        return set(ACCEPTED_EXTS)
    if settings.IS_MAC and shutil.which("afconvert"):
        return NATIVE_EXTS | AFCONVERT_EXTS
    return set(NATIVE_EXTS)


def check_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    allowed_exts = accepted_extensions()
    if ext not in allowed_exts:
        allowed = "・".join(sorted(e.lstrip(".").upper() for e in allowed_exts))
        raise AudioError(f"「{filename}」は対応していない形式です。対応形式: {allowed}")


def duration_of(path: Path) -> float | None:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def is_readable(path: Path) -> bool:
    """音声ファイルとして読めて、中身（1サンプル以上）があるか。"""
    try:
        info = sf.info(str(path))
        return info.frames > 0
    except Exception:
        return False


def is_silent(path: Path, threshold_db: float = -60.0) -> bool:
    """音声全体がほぼ無音か（いちばん大きな音でも threshold_db より小さい）。読めなければ False。"""
    limit = 10 ** (threshold_db / 20)
    try:
        for block in sf.blocks(str(path), blocksize=1 << 18, dtype="float32", always_2d=True):
            if block.size and float(np.abs(block).max()) >= limit:
                return False
    except Exception:
        return False
    return True


def ffmpeg_to_wav(
    source: Path,
    target: Path,
    mono: bool = False,
    sample_rate: int | None = None,
    channels: int | None = None,
    timeout: float = 1800,
    display_name: str | None = None,
) -> None:
    """ffmpeg で音声（動画なら音声だけ）を 16bit の WAV にする。display_name はエラーに出すファイル名。"""
    label = display_name or source.name
    ffmpeg = settings.ffmpeg_path()
    if not ffmpeg:
        raise AudioError("ffmpeg が見つかりません。npm run rvc:setup を実行してください。")
    options = ["-ac", "1"] if mono else (["-ac", str(channels)] if channels else [])
    if sample_rate:
        options += ["-ar", str(sample_rate)]
    command = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
        "-vn", "-sn", "-dn", *options, "-acodec", "pcm_s16le", "-f", "wav", str(target),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        raise AudioError(f"「{label}」の変換に時間がかかりすぎたため中止しました。")
    if result.returncode != 0 or not is_readable(target):
        target.unlink(missing_ok=True)
        raise AudioError(
            f"「{label}」を音声として読み込めませんでした。ファイルが壊れていないか、"
            "音声が入っているか確認してください。"
        )


def _afconvert_to_wav(source: Path, target: Path, mono: bool, display_name: str | None = None) -> None:
    label = display_name or source.name
    afconvert = shutil.which("afconvert")
    if not afconvert or source.suffix.lower() not in AFCONVERT_EXTS:
        raise AudioError(
            f"「{label}」を読み込めませんでした。WAV か MP3 に変換してからアップロードしてください。"
        )
    channels = ["-c", "1"] if mono else []
    command = [afconvert, "-f", "WAVE", "-d", "LEI16", *channels, str(source), str(target)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise AudioError(f"「{label}」の変換に時間がかかりすぎたため中止しました。")
    if result.returncode != 0 or not is_readable(target):
        target.unlink(missing_ok=True)
        raise AudioError(
            f"「{label}」を音声として読み込めませんでした。ファイルが壊れていないか確認してください。"
        )


def _convert_to_wav(source: Path, target: Path, mono: bool, display_name: str | None = None) -> None:
    if settings.ffmpeg_path():
        ffmpeg_to_wav(source, target, mono=mono, display_name=display_name)
    else:
        _afconvert_to_wav(source, target, mono, display_name)


def make_readable(path: Path, mono: bool = True, display_name: str | None = None) -> Path:
    """Applio が読める音声ファイルにして、そのパスを返す。

    そのまま読めない場合は同じフォルダに WAV を作り、元のファイルは削除する。
    RVC はモノラルで処理するので、ふつうはモノラルにしてファイルを小さくする。
    学習用・曲の変換用の音声はステレオのまま残す（BGM と声を分けるときに、左右の違いも手がかりになるため）。
    """
    if path.suffix.lower() in NATIVE_EXTS and is_readable(path):
        return path
    target = unique_path(path.parent, f"{path.stem}.wav")
    _convert_to_wav(path, target, mono, display_name)
    path.unlink(missing_ok=True)
    return target


def write_audio(path: Path, data: np.ndarray, rate: int, fmt: str) -> Path:
    """音声を指定の形式（wav / flac / mp3）で書き出す。data は (サンプル数,) か (サンプル数, チャンネル)。"""
    data = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    if fmt == "wav":
        sf.write(str(path), data, rate, subtype="PCM_16")
    elif fmt == "flac":
        sf.write(str(path), data, rate, format="FLAC", subtype="PCM_16")
    elif fmt == "mp3":
        _write_mp3(path, data, rate)
    else:
        raise AudioError(f"書き出せない形式です: {fmt}")
    return path


def _write_mp3(path: Path, data: np.ndarray, rate: int) -> None:
    ffmpeg = settings.ffmpeg_path()
    if ffmpeg:
        temp = path.with_suffix(".tmp.wav")
        sf.write(str(temp), data, rate, subtype="PCM_16")
        try:
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(temp),
                 "-codec:a", "libmp3lame", "-b:a", "320k", str(path)],
                capture_output=True, text=True, timeout=900,
            )
        finally:
            temp.unlink(missing_ok=True)
        if result.returncode == 0 and path.is_file():
            return
    # ffmpeg がないときは soundfile（libsndfile）で書き出す
    try:
        sf.write(str(path), data, rate, format="MP3", subtype="MPEG_LAYER_III")
    except Exception as error:
        raise AudioError(f"MP3 で書き出せませんでした（{error}）。WAV か FLAC を選んでください。")
