import numpy as np
import soundfile as sf

from rvc_api import mixing


def write(path, data, rate):
    sf.write(str(path), data, rate, subtype="FLOAT")
    return path


def sine(seconds, rate, freq, level):
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_matching_gain_restores_original_level():
    rate = 44100
    original = sine(2, rate, 220, 0.4)[:, None]
    converted = sine(2, rate, 220, 0.1)[:, None]
    gain = mixing.matching_gain(converted, original, rate)
    assert abs(20 * np.log10(gain) - 12.04) < 0.2


def test_matching_gain_is_limited():
    rate = 16000
    gain = mixing.matching_gain(sine(1, rate, 200, 0.001)[:, None], sine(1, rate, 200, 0.9)[:, None], rate)
    assert gain <= 10 ** (12 / 20) + 1e-6


def test_mix_song_shapes_and_peak(tmp_path):
    rate = 44100
    lead = write(tmp_path / "lead.wav", np.stack([sine(3, rate, 300, 0.5)] * 2, axis=1), rate)
    # 変換した声は 40kHz・モノラル・少し短い
    converted = write(tmp_path / "conv.wav", sine(2.9, 40000, 300, 0.2), 40000)
    inst = write(tmp_path / "inst.wav", np.stack([sine(3, rate, 110, 0.9)] * 2, axis=1), rate)
    song, vocals, music, out_rate, info = mixing.mix_song(
        converted, lead, inst, None, reverb=0.3, vocal_gain_db=0, instrumental_gain_db=0
    )
    assert out_rate == rate
    assert song.shape == (3 * rate, 2) and vocals.shape == song.shape and music.shape == song.shape
    assert np.max(np.abs(song)) <= mixing.PEAK_LIMIT + 1e-6
    assert info["vocal_match_db"] > 5  # 小さく変換された声を元の大きさに戻している
    assert info["peak_reduced_db"] < 0  # 大きすぎたので下げている


def test_reverb_adds_tail_and_stereo():
    rate = 22050
    dry = np.zeros((rate, 1), dtype=np.float32)
    dry[100] = 1.0
    wet = mixing.add_reverb(dry, rate, 0.5)
    assert wet.shape == (rate, 2)
    assert np.abs(wet[rate // 2 :]).sum() > 0  # 響きが残っている
    assert not np.allclose(wet[:, 0], wet[:, 1])  # 左右で少し違う
    assert mixing.add_reverb(dry, rate, 0) is dry


def test_resample_length():
    data = np.zeros((40000, 2), dtype=np.float32)
    assert abs(len(mixing.resample(data, 40000, 44100)) - 44100) <= 2
