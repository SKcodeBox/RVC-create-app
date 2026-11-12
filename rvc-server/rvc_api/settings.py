"""サーバー全体で使うパスと設定。Windows と macOS（と Linux）の違いもここで吸収する。

保存先はすべて rvc-server/data/ の中にまとめる（環境変数 RVC_DATA_DIR で変更可能）。
"""

import json
import os
import shutil
import sys
from functools import cache
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# 使っているライブラリが、開発元へ利用状況を送る機能（テレメトリ）を止める（アプリの動きには関係ない）。
# 送信の機能を含むだけで使わない onnxruntime は、入れていない（rvc-server/patches・separation/separate.py）。
# サーバーと、サーバーが起動するすべての処理に効く。
NO_TELEMETRY_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",  # Hugging Face のライブラリ（Applio が声の特徴を取り出すのに使う）
    "DO_NOT_TRACK": "1",  # この取り決めに従うライブラリ全般
}
os.environ.update(NO_TELEMETRY_ENV)

SERVER_DIR = Path(__file__).resolve().parent.parent  # rvc-server/
APPLIO_DIR = Path(os.environ.get("RVC_APPLIO_DIR", SERVER_DIR / "applio")).resolve()
DATA_DIR = Path(os.environ.get("RVC_DATA_DIR", SERVER_DIR / "data")).resolve()
# このアプリ独自の処理（学習の評価・声の高さの解析など）。Applio の Python で動かす
ENGINE_DIR = SERVER_DIR / "engine"

# Applio の「logs」フォルダとして使う。学習の作業ファイルと完成したモデルが入る
MODELS_DIR = DATA_DIR / "models"
# 学習用にアップロードされた音声（モデル名ごとのフォルダ）
DATASETS_DIR = DATA_DIR / "datasets"
# 変換用にアップロードされた音声（ジョブごとのフォルダ）
UPLOADS_DIR = DATA_DIR / "uploads"
# 変換した音声
OUTPUTS_DIR = DATA_DIR / "outputs"
# ジョブ（学習・変換）の記録
JOBS_DIR = DATA_DIR / "jobs"
# Applio の設定ファイル（ユーザーのホームフォルダに書かれないよう、ここに置く）
APPLIO_CONFIG_DIR = DATA_DIR / "applio-config"
# ボーカル分離（BGM・伴奏と声を分ける）に使う AI モデル（初めて使うときにダウンロードする）
# （テストでは、ダウンロード済みのものを使い回せるよう、環境変数で場所を変えられる）
SEPARATION_MODELS_DIR = Path(os.environ.get("RVC_SEPARATION_MODELS_DIR", DATA_DIR / "separation-models")).resolve()
# コミュニティの事前学習モデル（初めて使うときにダウンロードする）
PRETRAINED_DIR = Path(os.environ.get("RVC_PRETRAINED_DIR", DATA_DIR / "pretrained")).resolve()
# サーバーが覚えておく小さな状態（yt-dlp を最後に更新した日時など）
STATE_DIR = DATA_DIR / "state"


def venv_python(venv: Path) -> Path:
    """仮想環境の Python のパス（Windows は Scripts\\python.exe、それ以外は bin/python）。"""
    if IS_WINDOWS:
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


# ボーカル分離は、Applio と PyTorch のバージョンが合わないライブラリ（audio-separator）を使うため、
# 別の仮想環境（rvc-server/.venv-separation）の Python で separation/separate.py を動かす
SEPARATION_DIR = SERVER_DIR / "separation"
SEPARATION_PYTHON = venv_python(SERVER_DIR / ".venv-separation")

# サーバーを動かしている Python（.venv の Python）で Applio のスクリプトも動かす
PYTHON = sys.executable

HOST = os.environ.get("RVC_HOST", "127.0.0.1")
PORT = int(os.environ.get("RVC_PORT", "8765"))

# Applio が models フォルダの中に作る、モデルではないフォルダ
RESERVED_NAMES = {"reference", "mute", "mute_spin", "mute_spin-v2", "zips"}

# 学習で選べるサンプルレート（HiFi-GAN の事前学習モデルがあるもの）
SAMPLE_RATES = (32000, 40000, 48000)


def ensure_dirs() -> None:
    for directory in (
        MODELS_DIR,
        DATASETS_DIR,
        UPLOADS_DIR,
        OUTPUTS_DIR,
        JOBS_DIR,
        APPLIO_CONFIG_DIR,
        SEPARATION_MODELS_DIR,
        PRETRAINED_DIR,
        STATE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


@cache
def ffmpeg_path() -> str | None:
    """ffmpeg のパス。セットアップで入れたもの（imageio-ffmpeg）を優先し、なければ PATH から探す。"""
    configured = os.environ.get("RVC_FFMPEG")
    if configured and Path(configured).is_file():
        return configured
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and Path(path).is_file():
            return path
    except Exception:  # ライブラリがない・バイナリが壊れているなど
        pass
    return shutil.which("ffmpeg")


def uv_path() -> str | None:
    """uv（Python のパッケージ管理ツール）のパス。yt-dlp の自動更新に使う。"""
    configured = os.environ.get("RVC_UV")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("uv")
    if found:
        return found
    home = Path.home() / ".local" / "bin" / ("uv.exe" if IS_WINDOWS else "uv")
    return str(home) if home.is_file() else None


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            # ログをすぐに受け取れるようにする
            "PYTHONUNBUFFERED": "1",
            # Windows でも日本語のファイル名やログを正しく扱えるようにする
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            # Mac の GPU（MPS）で未対応の処理は CPU で代わりに行う
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            # 進捗バーの更新を控えめにしてログを読みやすくする
            "TQDM_MININTERVAL": "2",
            **NO_TELEMETRY_ENV,
        }
    )
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        # ffmpeg を呼ぶライブラリ（pydub など）が見つけられるようにする
        env["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + env.get("PATH", "")
        env["RVC_FFMPEG"] = ffmpeg
    return env


def applio_env() -> dict[str, str]:
    """Applio のスクリプト（と engine/ のスクリプト）を動かすときの環境変数。"""
    env = _base_env()
    # これが設定されていると、Applio は出力を別ファイルに書いてしまう
    env.pop("APPLIO_TERMINAL_LOG", None)
    env.update(
        {
            "APPLIO_LOGS_DIR": str(MODELS_DIR),
            "APPLIO_CONFIG_DIR": str(APPLIO_CONFIG_DIR),
            # スクリプトをフルパスで起動しても「rvc.～」を import できるようにする
            "PYTHONPATH": str(APPLIO_DIR),
        }
    )
    # Applio の core.py が macOS で設定している値に合わせる
    if IS_MAC:
        env.setdefault("OMP_NUM_THREADS", "1")
    return env


def separation_env() -> dict[str, str]:
    """ボーカル分離を動かすときの環境変数（Applio 用の設定は渡さない）。"""
    env = _base_env()
    # 別の仮想環境で動かすので、Applio のフォルダを import の対象にしない
    for key in ("PYTHONPATH", "PYTHONHOME", "APPLIO_TERMINAL_LOG"):
        env.pop(key, None)
    return env


def applio_version() -> str | None:
    try:
        template = APPLIO_DIR / "assets" / "config_template.json"
        return json.loads(template.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        return None


def worker_cores() -> int:
    """前処理・特徴抽出で使う CPU コア数（他の作業のために少し残す）。"""
    return max(1, (os.cpu_count() or 2) - 2)


# Applio が使う AI モデルの置き場所
PREDICTOR_RMVPE = APPLIO_DIR / "rvc" / "models" / "predictors" / "rmvpe.pt"
EMBEDDER_CONTENTVEC = (
    APPLIO_DIR / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin"
)


def standard_pretrained_paths(sample_rate: int) -> tuple[Path, Path]:
    """RVC 標準の HiFi-GAN 事前学習モデル（生成器 G と識別器 D）のパス。"""
    folder = APPLIO_DIR / "rvc" / "models" / "pretraineds" / "hifi-gan"
    khz = sample_rate // 1000
    return folder / f"f0G{khz}k.pth", folder / f"f0D{khz}k.pth"
