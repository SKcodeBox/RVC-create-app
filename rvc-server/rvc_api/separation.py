"""音声の下ごしらえ（学習の前）と、曲の声と伴奏を分ける処理（変換の前）。

- vocals   … BGM・伴奏を取り除いて声だけにする（ボーカル分離）
- karaoke  … ハモリ（コーラス）を取り除いて、メインの声だけにする
- dereverb … 残響（リバーブ）とエコーを取り除く
- denoise  … 雑音を減らす（Applio の DeepFilterNet。話し声向け。学習の前だけ）

vocals・karaoke・dereverb は AI モデルを使う処理で、Applio とは別の仮想環境の Python で
separation/separate.py を動かして行う（使うモデルは separation/models.json で決める）。
BGM・伴奏の除去には「高精度（2つのモデルの組み合わせ）」と「速さ優先」がある。
"""

import json
from functools import cache

from . import settings

DENOISE = "denoise"
QUALITIES = ("high", "fast")
DEFAULT_QUALITY = "high"


class PrepError(Exception):
    """利用者に見せるメッセージを持つエラー。"""


@cache
def _config() -> dict:
    return json.loads((settings.SEPARATION_DIR / "models.json").read_text(encoding="utf-8"))


def model_steps() -> list[str]:
    """AI モデルで行う手順（行う順）。"""
    return list(_config()["order"])


def label(step: str) -> str:
    """処理中に表示する説明（例: 「BGM・伴奏と声を分けています」）。"""
    return _config()["steps"][step]["label"]


def name(step: str) -> str:
    """手順の短い名前（例: 「BGM・伴奏の除去」）。"""
    if step == DENOISE:
        return "雑音の除去"
    return _config()["steps"][step]["name"]


def is_installed() -> bool:
    """ボーカル分離用の仮想環境が用意されているか（npm run rvc:setup で作られる）。"""
    return (
        settings.SEPARATION_PYTHON.is_file()
        and (settings.SEPARATION_DIR / "separate.py").is_file()
    )


def _setting(step: str, quality: str) -> dict:
    qualities = _config()["steps"][step]["quality"]
    return qualities.get(quality) or qualities["high"]


def models_of(step: str, quality: str = DEFAULT_QUALITY) -> list[dict]:
    """その手順で使うモデル（複数なら結果を組み合わせる）。separation/separate.py と同じ決め方。"""
    return list(_setting(step, quality)["models"])


def is_downloaded(step: str, quality: str = DEFAULT_QUALITY) -> bool:
    return all(
        (settings.SEPARATION_MODELS_DIR / model["file"]).is_file()
        for model in models_of(step, quality)
    )


def download_mb(steps: list[str], quality: str = DEFAULT_QUALITY) -> int:
    """まだダウンロードしていないモデルの合計（MB）。"""
    files: dict[str, int] = {}
    for step in steps:
        for model in models_of(step, quality):
            if not (settings.SEPARATION_MODELS_DIR / model["file"]).is_file():
                files[model["file"]] = model.get("download_mb", 0)
    return sum(files.values())


def speed(steps: list[str], quality: str = DEFAULT_QUALITY) -> float:
    """処理にかかる時間の目安（音声の長さに対する倍率。Apple M5 で測った値）。"""
    return round(sum(_setting(step, quality).get("speed", 1.0) for step in steps), 2)


def status() -> dict:
    """画面に知らせる状態（準備ができているか、モデルをダウンロード済みか、時間の目安）。"""
    return {
        "installed": is_installed(),
        "steps": {
            step: {
                quality: {
                    "downloaded": is_downloaded(step, quality),
                    "download_mb": sum(model.get("download_mb", 0) for model in models_of(step, quality)),
                    "pending_mb": download_mb([step], quality),
                    "speed": speed([step], quality),
                    "models": len(models_of(step, quality)),
                }
                for quality in QUALITIES
            }
            for step in model_steps()
        },
    }


def parse_quality(raw: str | None) -> str:
    quality = (raw or DEFAULT_QUALITY).strip()
    if quality not in QUALITIES:
        raise PrepError("BGM・伴奏の除去の精度の指定が正しくありません。")
    return quality


def _require_installed() -> None:
    if not is_installed():
        raise PrepError(
            "BGM・伴奏を取り除く機能の準備ができていません。npm run rvc:setup を実行してから、"
            "サーバーを起動し直してください。"
        )


def parse_steps(raw: str | None) -> list[str]:
    """画面から届いた手順（"vocals,dereverb" のような文字列）を確かめ、行う順に並べる。"""
    requested = {part.strip() for part in (raw or "").split(",") if part.strip()}
    known = [*model_steps(), DENOISE]
    unknown = requested - set(known)
    if unknown:
        raise PrepError("音声の下ごしらえの指定が正しくありません。")
    steps = [step for step in known if step in requested]
    if any(step in model_steps() for step in steps):
        _require_installed()
    return steps


def parse_song_steps(dereverb: bool, harmony: bool) -> list[str]:
    """曲の変換で行う手順（BGM・伴奏の除去は必ず行う）。"""
    _require_installed()
    requested = {"vocals"}
    if dereverb:
        requested.add("dereverb")
    if harmony:
        requested.add("karaoke")
    return [step for step in model_steps() if step in requested]


def _base_command(steps: list[str], quality: str) -> list[str]:
    return [
        str(settings.SEPARATION_PYTHON),
        str(settings.SEPARATION_DIR / "separate.py"),
        "--models-dir", str(settings.SEPARATION_MODELS_DIR),
        "--steps", ",".join(steps),
        "--quality", quality,
    ]


def download_command(steps: list[str], quality: str = DEFAULT_QUALITY) -> list[str]:
    return [*_base_command(steps, quality), "--download-only"]


def separate_command(steps: list[str], files, work_dir, quality: str = DEFAULT_QUALITY, keep_stages: bool = False) -> list[str]:
    command = [*_base_command(steps, quality), "--work-dir", str(work_dir)]
    if keep_stages:
        command.append("--keep-stages")
    return [*command, *(str(path) for path in files)]


def song_command(steps: list[str], source, work_dir, quality: str = DEFAULT_QUALITY) -> list[str]:
    return [*_base_command(steps, quality), "--mode", "song", "--work-dir", str(work_dir), str(source)]
