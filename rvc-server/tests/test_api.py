import pytest
from fastapi.testclient import TestClient

from rvc_api import evaluation, library, main

HEADERS = {"X-RVC-Client": "1"}


@pytest.fixture
def client():
    # 本物の画面と同じく、この PC を指すアドレスで呼ぶ（それ以外は断る仕組みのため）
    with TestClient(main.app, base_url="http://127.0.0.1:8765") as test_client:
        yield test_client


def make_model(name="voice"):
    folder = library.model_dir(name)
    folder.mkdir(parents=True)
    for epoch in (10, 20):
        (folder / f"{name}_{epoch}e_{epoch * 5}s.pth").write_bytes(b"w" * 10)
    library.write_meta(name, source="trained", created_at=1.0)
    evaluation.write_evaluations(folder, [{"epoch": 10, "val_loss": 2.0}, {"epoch": 20, "val_loss": 3.0}])
    return folder


def test_changes_require_client_header(client):
    response = client.post("/api/models/import")
    assert response.status_code == 403


def test_system_reports_platform_and_features(client):
    data = client.get("/api/system").json()
    assert data["platform"] in ("mac", "windows", "linux")
    assert {"pretrains", "embedders", "separation", "accepted_extensions", "ffmpeg"} <= set(data)
    assert any(item["id"] == "standard" for item in data["pretrains"])


def test_quality_endpoint(client):
    make_model()
    data = client.get("/api/models/voice/quality").json()
    assert data["recommended_epoch"] == 10
    assert [item["epoch"] for item in data["evaluations"]] == [10, 20]


def test_models_list_has_recommendation(client):
    make_model()
    models = client.get("/api/models").json()["models"]
    assert models[0]["recommended_weight"] == "voice_10e_50s.pth"


def test_delete_weight_and_cleanup(client):
    make_model()
    response = client.delete("/api/models/voice/weights/voice_20e_100s.pth", headers=HEADERS)
    assert response.status_code == 200
    assert [w["epoch"] for w in response.json()["model"]["weights"]] == [10]
    response = client.post("/api/models/voice/cleanup", json={"stages": True}, headers=HEADERS)
    assert response.status_code == 200 and "storage" in response.json()


def test_set_voice_type(client):
    make_model()
    response = client.patch("/api/models/voice", json={"voice_type": "low"}, headers=HEADERS)
    assert response.json()["model"]["target_pitch"] == 155.0
    response = client.patch("/api/models/voice", json={"voice_type": "weird"}, headers=HEADERS)
    assert response.status_code == 400


def test_unknown_model_is_404ish(client):
    assert client.get("/api/models/nothing").status_code == 400
    assert client.get("/api/models/..%2F..%2Fetc").status_code in (400, 404)


def test_rejects_foreign_host_dns_rebinding(client):
    """悪意のあるサイトが行き先をこの PC にすり替えても（DNS リバインディング）、操作できない。"""
    make_model()
    evil = {"Host": "evil.example:8765", "Origin": "http://evil.example:8765", **HEADERS}
    assert client.delete("/api/models/voice", headers=evil).status_code == 403
    assert client.get("/api/models", headers={"Host": "evil.example:8765"}).status_code == 403
    # Host が正しくても、別のサイトの画面からの操作は断る
    assert client.delete("/api/models/voice", headers={"Origin": "https://evil.example", **HEADERS}).status_code == 403
    assert library.model_dir("voice").is_dir()
    for host in ("localhost:8765", "127.0.0.1", "[::1]:8765"):
        assert client.get("/api/models", headers={"Host": host}).status_code == 200
    assert client.get("/api/models", headers={"Origin": "http://localhost:3000"}).status_code == 200


def test_host_name_parsing():
    assert main._host_name("127.0.0.1:8765") == "127.0.0.1"
    assert main._host_name("[::1]:8765") == "[::1]"
    assert main._host_name("LOCALHOST") == "localhost"
    assert main._host_name("::1") == "::1"


def test_rejects_windows_reserved_model_name(client, tone, data_dir):
    voice = tone(data_dir / "v.wav", seconds=12)
    with voice.open("rb") as handle:
        response = client.post("/api/train", headers=HEADERS, data={"name": "CON"}, files={"files": ("v.wav", handle)})
    assert response.status_code in (400, 503)
    if response.status_code == 400:
        assert "Windows" in response.json()["detail"]


def test_rejects_silent_training_audio(client, data_dir, monkeypatch):
    import numpy as np
    import soundfile as sf

    monkeypatch.setattr(main, "_require_ready", lambda: None)
    silent = data_dir / "silence.wav"
    sf.write(str(silent), np.zeros(44100 * 15, dtype=np.float32), 44100)
    with silent.open("rb") as handle:
        response = client.post("/api/train", headers=HEADERS, data={"name": "quiet"}, files={"files": ("silence.wav", handle)})
    assert response.status_code == 400
    assert "無音" in response.json()["detail"]
    assert not library.dataset_dir("quiet").exists()


def test_rejects_hybrid_pitch_methods(client, tone, data_dir, monkeypatch):
    monkeypatch.setattr(main, "_require_ready", lambda: None)
    make_model()
    voice = tone(data_dir / "a.wav", seconds=2)
    with voice.open("rb") as handle:
        response = client.post(
            "/api/convert", headers=HEADERS, data={"model": "voice", "f0_method": "hybrid[rmvpe+fcpe]"}, files={"file": ("a.wav", handle)}
        )
    assert response.status_code == 400


def test_convert_error_uses_original_file_name(client, data_dir, monkeypatch):
    monkeypatch.setattr(main, "_require_ready", lambda: None)
    make_model()
    broken = data_dir / "broken.wav"
    broken.write_bytes(b"not audio" * 100)
    with broken.open("rb") as handle:
        response = client.post("/api/convert", headers=HEADERS, data={"model": "voice"}, files={"file": ("わたしの録音.wav", handle)})
    assert response.status_code == 400
    assert "わたしの録音.wav" in response.json()["detail"] and "input.wav" not in response.json()["detail"]


def test_resume_refuses_broken_checkpoint(client, monkeypatch):
    from tests.conftest import write_checkpoint

    monkeypatch.setattr(main, "_require_ready", lambda: None)
    folder = make_model()
    write_checkpoint(folder / "G_2333333.pth", 20)
    (folder / "D_2333333.pth").write_bytes(b"PK\x03\x04 truncated")
    (folder / "filelist.txt").write_text("x", encoding="utf-8")
    response = client.post("/api/train", headers=HEADERS, data={"name": "voice", "resume": "true", "total_epochs": "40"})
    assert response.status_code == 400
    assert "壊れて" in response.json()["detail"]
