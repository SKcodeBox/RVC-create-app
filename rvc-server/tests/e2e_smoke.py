"""通しの動作確認（npm run test:e2e）。本物の Applio と AI モデルを使うので数分〜十数分かかる。

一時フォルダを保存先にしたサーバーを別のポートで起動し、次を順に確かめる:
  1. 短い音声で学習する（検証用の取り分け・保存ごとの評価と試し聴き・おすすめのエポック・インデックス）
  2. 声の高さを自動で合わせて変換し、MP3 で書き出す（変換後の雑音除去も）
  3. 曲（声＋伴奏）を、声と伴奏に分けて変換し、合わせ直す（ハモリも分けて変換）
  4. いらないエポックの削除と、保存容量の整理
ダウンロード済みの AI モデル（rvc-server/data の中）は、そのまま使い回す。
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))
HEADERS = {"X-RVC-Client": "1"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Api:
    def __init__(self, base: str) -> None:
        self.base = base

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=60) as response:
            return json.loads(response.read())

    def send(self, method: str, path: str, data: dict | None = None, files: dict | None = None):
        import httpx

        response = httpx.request(method, self.base + path, headers=HEADERS, data=data, files=files, timeout=600)
        if response.status_code >= 400:
            raise AssertionError(f"{method} {path}: {response.status_code} {response.text}")
        return response.json()

    def send_json(self, method: str, path: str, body: dict):
        import httpx

        response = httpx.request(method, self.base + path, headers=HEADERS, json=body, timeout=600)
        if response.status_code >= 400:
            raise AssertionError(f"{method} {path}: {response.status_code} {response.text}")
        return response.json()

    def wait(self, job_id: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            job = self.get(f"/api/jobs/{job_id}?logs=400")["job"]
            label = (job["status"], job.get("stage_label"))
            if label != last:
                print(f"    {job['status']}: {job.get('stage_label') or ''}", flush=True)
                last = label
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(3)
        raise AssertionError(f"ジョブ {job_id} が {timeout} 秒で終わりませんでした")


def speech(path: Path, text: str, voice: str) -> Path:
    """macOS の読み上げで話し声を作る（ほかの OS では、声に似せた合成音）。"""
    if shutil.which("say"):
        aiff = path.with_suffix(".aiff")
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        data, rate = sf.read(str(aiff))
        aiff.unlink()
        sf.write(str(path), data, rate, subtype="PCM_16")
        return path
    rate = 44100
    seconds = max(20.0, len(text) * 0.12)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voice_like = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = (np.sin(2 * np.pi * 2.3 * t) > -0.3).astype(float)
    sf.write(str(path), (0.2 * voice_like * envelope).astype(np.float32), rate, subtype="PCM_16")
    return path


def song(path: Path, vocal: Path) -> Path:
    """話し声に、和音の伴奏を重ねた「曲」を作る（ステレオ・44.1kHz）。"""
    data, rate = sf.read(str(vocal), always_2d=True)
    voice = data.mean(axis=1)
    t = np.arange(len(voice)) / rate
    chords = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (110, 138.6, 164.8, 220))
    beat = 0.15 * np.sin(2 * np.pi * 55 * t) * (np.mod(t, 0.5) < 0.08)
    left = voice * 0.9 + chords + beat
    right = voice * 0.9 + chords * 0.8 + beat
    sf.write(str(path), np.stack([left, right], axis=1).astype(np.float32), rate, subtype="PCM_16")
    return path


TEXT = (
    "こんにちは。今日は声の学習を試しています。"
    "天気がいいので、少し散歩に出かけようと思います。"
    "新しい歌を練習して、みんなの前で歌ってみたいです。"
    "音声の変換は、思っていたよりもずっと自然に聞こえます。"
) * 2


def main() -> int:
    temp = Path(tempfile.mkdtemp(prefix="rvc-studio-e2e-"))
    port = free_port()
    python = sys.executable
    env = {
        **os.environ,
        "RVC_DATA_DIR": str(temp / "data"),
        "RVC_SEPARATION_MODELS_DIR": str(SERVER / "data" / "separation-models"),
        "RVC_PRETRAINED_DIR": str(SERVER / "data" / "pretrained"),
    }
    print(f"一時フォルダ: {temp}")
    server = subprocess.Popen(
        [python, "-m", "uvicorn", "rvc_api.main:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
        cwd=SERVER, env=env,
    )
    api = Api(f"http://127.0.0.1:{port}")
    try:
        for _ in range(60):
            try:
                system = api.get("/api/system")
                break
            except OSError:
                time.sleep(1)
        else:
            raise AssertionError("サーバーが起動しませんでした")
        assert system["ready"], system["missing"]
        print(f"サーバー: {system['platform']} / ffmpeg {system['ffmpeg']}")

        print("1. 学習（短い音声・12エポック・4エポックごとに保存）")
        voice = speech(temp / "voice.wav", TEXT * 2, "Kyoko" if shutil.which("say") else "")
        print(f"   学習用の音声: {sf.info(str(voice)).duration:.0f}秒")
        with voice.open("rb") as handle:
            job = api.send("POST", "/api/train", data={
                "name": "e2e-voice", "sample_rate": "40000", "total_epochs": "12", "batch_size": "4",
                "save_every": "4", "prep": "", "early_stop": "true",
            }, files={"files": ("voice.wav", handle, "audio/wav")})["job"]
        job = api.wait(job["id"], 1800)
        assert job["status"] == "done", (job["message"], "\n".join(job["logs"][-30:]))
        model = api.get("/api/models/e2e-voice")["model"]
        quality = api.get("/api/models/e2e-voice/quality")
        epochs = [item["epoch"] for item in quality["evaluations"]]
        print(f"   評価したエポック: {epochs}・検証用 {quality['validation_clips']}個・おすすめ {quality['recommended_epoch']}")
        assert epochs == [4, 8, 12], epochs
        assert quality["validation_clips"] and quality["validation_clips"] >= 3
        assert all(item["val_loss"] is not None and item["has_sample"] for item in quality["evaluations"])
        assert quality["recommended_epoch"] in epochs
        assert model["index_file"], "インデックスがありません"
        assert model["target_pitch"], "声の高さが記録されていません"
        assert quality["curve"].get("mel"), "学習の記録（TensorBoard）を読めませんでした"
        with urllib.request.urlopen(f"{api.base}/api/models/e2e-voice/samples/{epochs[-1]}") as response:
            assert response.read(4) == b"fLaC"

        print("2. 変換（声の高さを自動で合わせる・MP3・変換後の雑音除去）")
        source = speech(temp / "source.wav", "これは変換のテストです。少し低い声で話しています。", "Eddy (日本語（日本）)" if shutil.which("say") else "")
        with source.open("rb") as handle:
            job = api.send("POST", "/api/convert", data={
                "model": "e2e-voice", "pitch_mode": "auto", "export_format": "mp3", "clean_output": "true",
            }, files={"file": ("source.wav", handle, "audio/wav")})["job"]
        job = api.wait(job["id"], 900)
        assert job["status"] == "done", (job["message"], "\n".join(job["logs"][-30:]))
        result = job["result"]
        print(f"   ピッチ {result['pitch_used']:+d}（元 {result['source_pitch']}Hz → モデル {result['target_pitch']}Hz）")
        assert result["output"].endswith(".mp3") and result["duration"] > 1
        assert job["params"]["weight"] == model["recommended_weight"], "おすすめのエポックが使われていません"

        print("3. 曲の変換（声と伴奏を分ける・ハモリも分けて変換・速さ優先）")
        mix = song(temp / "song.wav", source)
        with mix.open("rb") as handle:
            job = api.send("POST", "/api/convert", data={
                "model": "e2e-voice", "input_kind": "song", "pitch_mode": "auto", "harmony": "convert",
                "separation_quality": "fast", "export_format": "flac", "reverb": "0.3",
            }, files={"file": ("song.wav", handle, "audio/wav")})["job"]
        job = api.wait(job["id"], 1800)
        assert job["status"] == "done", (job["message"], "\n".join(job["logs"][-30:]))
        result = job["result"]
        assert result["pitch_used"] % 12 == 0, "曲なのにオクターブ単位になっていません"
        assert set(result["stems"]) == {"vocals", "instrumental"}
        output = Path(env["RVC_DATA_DIR"]) / "outputs" / result["output"]
        info = sf.info(str(output))
        print(f"   {info.channels}ch・{info.samplerate}Hz・{info.duration:.1f}秒・ミックス {result.get('mix')}")
        assert info.channels == 2 and abs(info.duration - sf.info(str(mix)).duration) < 0.5

        print("4. エポックの削除と、保存容量の整理")
        api.send("DELETE", "/api/models/e2e-voice/weights/" + next(
            w["file"] for w in model["weights"] if w["epoch"] == 4
        ))
        cleaned = api.send_json("POST", "/api/models/e2e-voice/cleanup", {"weights": True, "checkpoints": True})
        print(f"   {cleaned['freed'] / 1e6:.0f}MB 空けました")
        assert not cleaned["model"]["can_resume"]
        print("すべて成功しました")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        if os.environ.get("RVC_E2E_KEEP") != "1":
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
