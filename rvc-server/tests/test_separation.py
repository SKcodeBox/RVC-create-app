import json

import pytest

from rvc_api import separation, settings


def test_models_json_is_complete():
    config = json.loads((settings.SEPARATION_DIR / "models.json").read_text(encoding="utf-8"))
    assert config["order"] == ["vocals", "dereverb", "karaoke"]
    for step in config["order"]:
        qualities = config["steps"][step]["quality"]
        assert "high" in qualities
        for setting in qualities.values():
            assert setting["models"], step
            for model in setting["models"]:
                assert model["file"] and model["stem"] and model["download_mb"] > 0
            if len(setting["models"]) > 1:
                assert setting["ensemble"]


def test_high_quality_vocals_is_an_ensemble():
    assert len(separation.models_of("vocals", "high")) == 2
    assert len(separation.models_of("vocals", "fast")) == 1
    # 速さ優先の設定がない手順は、高精度の設定を使う
    assert separation.models_of("dereverb", "fast") == separation.models_of("dereverb", "high")


def test_parse_steps_orders_and_validates(monkeypatch):
    monkeypatch.setattr(separation, "is_installed", lambda: True)
    assert separation.parse_steps("karaoke,vocals,denoise") == ["vocals", "karaoke", "denoise"]
    assert separation.parse_steps("") == []
    with pytest.raises(separation.PrepError):
        separation.parse_steps("vocals,unknown")


def test_parse_steps_requires_installation(monkeypatch):
    monkeypatch.setattr(separation, "is_installed", lambda: False)
    assert separation.parse_steps("denoise") == ["denoise"]
    with pytest.raises(separation.PrepError):
        separation.parse_steps("vocals")


def test_song_steps(monkeypatch):
    monkeypatch.setattr(separation, "is_installed", lambda: True)
    assert separation.parse_song_steps(True, True) == ["vocals", "dereverb", "karaoke"]
    assert separation.parse_song_steps(False, False) == ["vocals"]


def test_quality_and_status():
    assert separation.parse_quality(None) == "high"
    with pytest.raises(separation.PrepError):
        separation.parse_quality("best")
    status = separation.status()
    assert set(status["steps"]) == {"vocals", "dereverb", "karaoke"}
    assert status["steps"]["vocals"]["high"]["models"] == 2
    assert separation.speed(["vocals"], "high") > separation.speed(["vocals"], "fast")


def test_commands():
    command = separation.separate_command(["vocals"], ["a.wav"], "work", "fast", keep_stages=True)
    assert "--keep-stages" in command and command[-1] == "a.wav"
    assert command[command.index("--quality") + 1] == "fast"
    song = separation.song_command(["vocals", "karaoke"], "mix.wav", "work")
    assert song[song.index("--mode") + 1] == "song"
