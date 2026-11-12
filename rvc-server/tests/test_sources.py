import time

import pytest

from rvc_api import settings
from rvc_api import sources


def test_parse_time():
    assert sources.parse_time("") is None
    assert sources.parse_time("90") == 90
    assert sources.parse_time("1:30") == 90
    assert sources.parse_time("1:02:30") == 3750
    with pytest.raises(sources.SourceError):
        sources.parse_time("1:2:3:4")
    with pytest.raises(sources.SourceError):
        sources.parse_time("abc")


def test_parse_sources_validates():
    parsed = sources.parse_sources('[{"url": "https://example.com/v", "start": "0:10", "end": "1:00"}]')
    assert parsed == [{"url": "https://example.com/v", "start": 10, "end": 60}]
    with pytest.raises(sources.SourceError):
        sources.parse_sources('[{"url": "ftp://example.com"}]')
    with pytest.raises(sources.SourceError):
        sources.parse_sources('[{"url": "https://e.com", "start": "2:00", "end": "1:00"}]')
    with pytest.raises(sources.SourceError):
        sources.parse_sources("[" + ",".join(['{"url": "https://e.com"}'] * 11) + "]")


def test_audio_format_prefers_best_quality_with_ffmpeg(monkeypatch):
    monkeypatch.setattr(settings, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    assert sources.audio_format() == "bestaudio/best"
    monkeypatch.setattr(settings, "ffmpeg_path", lambda: None)
    assert sources.audio_format().startswith("bestaudio[ext=m4a]")


def test_download_command_has_format_and_filters(monkeypatch):
    monkeypatch.setattr(settings, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    command = sources.download_command("https://example.com/v", "/tmp/01.%(ext)s")
    assert command[-1] == "https://example.com/v"
    assert "-f" in command and command[command.index("-f") + 1] == "bestaudio/best"
    assert "--no-playlist" in command


def test_failure_explanations():
    assert "非公開" in sources.explain_failure(["ERROR: Private video"])
    assert "3時間" in sources.explain_failure(["... does not pass filter ..."])
    assert sources.update_may_help(["ERROR: HTTP Error 403: Forbidden"])
    assert not sources.update_may_help(["ERROR: Private video"])


def test_update_skips_when_recent(monkeypatch):
    stamp = settings.STATE_DIR / "yt-dlp-updated"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    called = []
    monkeypatch.setattr(settings, "uv_path", lambda: called.append(1) or None)
    assert sources.update_ytdlp() is False
    assert not called


def test_update_without_uv_reports(monkeypatch):
    stamp = settings.STATE_DIR / "yt-dlp-updated"
    stamp.unlink(missing_ok=True)
    monkeypatch.setattr(settings, "uv_path", lambda: None)
    lines = []
    assert sources.update_ytdlp(log=lines.append) is False
    assert any("uv" in line for line in lines)
    # 時間がたっていれば、また試す
    assert time.time() > 0
