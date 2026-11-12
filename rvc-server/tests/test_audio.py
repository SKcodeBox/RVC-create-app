import subprocess

import numpy as np
import pytest
import soundfile as sf

from rvc_api import audio, settings


def test_safe_filename_removes_paths_and_reserved_names():
    assert audio.safe_filename("../../etc/passwd.wav") == "passwd.wav"
    assert audio.safe_filename("C:\\Users\\me\\voice.MP3") == "voice.mp3"
    assert audio.safe_filename("CON.wav") == "_CON.wav"
    assert audio.safe_filename("a:b*c?.wav") == "a_b_c_.wav"
    assert audio.safe_filename("") == "audio"
    assert audio.safe_filename("名前 .wav") == "名前.wav"


def test_unique_path(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "a_2.wav").write_bytes(b"x")
    assert audio.unique_path(tmp_path, "a.wav").name == "a_3.wav"


def test_check_extension_rejects_unknown():
    with pytest.raises(audio.AudioError):
        audio.check_extension("notes.txt")
    audio.check_extension("voice.wav")


needs_ffmpeg = pytest.mark.skipif(settings.ffmpeg_path() is None, reason="ffmpeg がありません")


@needs_ffmpeg
@pytest.mark.parametrize("ext,codec", [(".m4a", "aac"), (".webm", "libopus"), (".mp4", "aac"), (".opus", "libopus")])
def test_converts_compressed_and_video_files(tmp_path, tone, ext, codec):
    source = tone(tmp_path / "tone.wav", seconds=1.5, channels=2)
    encoded = tmp_path / f"input{ext}"
    result = subprocess.run(
        [settings.ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(source), "-c:a", codec, str(encoded)],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    audio.check_extension(encoded.name)
    readable = audio.make_readable(encoded, mono=True)
    assert readable.suffix == ".wav"
    info = sf.info(str(readable))
    assert info.channels == 1
    assert 1.3 < info.duration < 1.7
    assert not encoded.exists()


@needs_ffmpeg
def test_ffmpeg_to_wav_resamples_to_stereo(tmp_path, tone):
    source = tone(tmp_path / "mono.wav", seconds=1.0, rate=22050, channels=1)
    target = tmp_path / "out.wav"
    audio.ffmpeg_to_wav(source, target, sample_rate=44100, channels=2)
    info = sf.info(str(target))
    assert info.samplerate == 44100 and info.channels == 2


@pytest.mark.parametrize("fmt", ["wav", "flac"] + (["mp3"] if settings.ffmpeg_path() else []))
def test_write_audio_formats(tmp_path, fmt):
    data = (0.2 * np.sin(np.linspace(0, 200, 44100))).astype(np.float32)
    path = audio.write_audio(tmp_path / f"out.{fmt}", np.stack([data, data], axis=1), 44100, fmt)
    info = sf.info(str(path))
    assert info.channels == 2
    assert 0.9 < info.duration < 1.2


def test_write_audio_clips_instead_of_wrapping(tmp_path):
    data = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    path = audio.write_audio(tmp_path / "loud.wav", data, 8000, "wav")
    read, _ = sf.read(str(path))
    assert read.max() <= 1.0 and read.min() >= -1.0


def test_is_silent(tmp_path, tone):
    import numpy as np
    import soundfile as sf

    quiet = tmp_path / "quiet.wav"
    sf.write(str(quiet), np.full(44100, 1e-5, dtype=np.float32), 44100)
    assert audio.is_silent(quiet)
    assert not audio.is_silent(tone(tmp_path / "tone.wav", seconds=1))
    assert not audio.is_silent(tmp_path / "missing.wav")  # 読めないものは「無音」とはしない
