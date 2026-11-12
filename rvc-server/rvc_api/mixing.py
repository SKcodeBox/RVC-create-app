"""曲の変換の仕上げ: 変換した声と伴奏（とハモリ）を合わせて、1つの曲にする。

- 変換した声の大きさを、元の曲の声の大きさに合わせる（伴奏とのバランスを元の曲のままにする）
- 残響を取り除いた場合は、軽い残響を付け直せる（声だけが浮かないように）
- 音割れしないよう、大きすぎるときは全体を下げる
"""

from pathlib import Path

import numpy as np
import soundfile as sf

# 「声がある」とみなす大きさ（dB）。これより小さい部分は、声の大きさを比べるときに使わない
VOICED_DB = -45.0
# 書き出すときの最大の大きさ（-0.3dBFS）
PEAK_LIMIT = 10 ** (-0.3 / 20)


def read(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data, rate


def resample(data: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    if rate == target_rate:
        return data
    import soxr

    return soxr.resample(data, rate, target_rate, quality="VHQ").astype(np.float32)


def to_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.shape[1] == channels:
        return data
    mono = data.mean(axis=1, keepdims=True)
    return np.repeat(mono, channels, axis=1)


def fit_length(data: np.ndarray, frames: int) -> np.ndarray:
    if len(data) >= frames:
        return data[:frames]
    pad = np.zeros((frames - len(data), data.shape[1]), dtype=data.dtype)
    return np.concatenate([data, pad])


def voiced_rms(data: np.ndarray, rate: int) -> float:
    """声が入っている部分だけの大きさ（RMS）。"""
    mono = data.mean(axis=1) if data.ndim > 1 else data
    frame = max(int(rate * 0.05), 1)
    count = len(mono) // frame
    if count == 0:
        return 0.0
    frames = mono[: count * frame].reshape(count, frame).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    loud = rms[20 * np.log10(rms + 1e-9) > VOICED_DB]
    return float(np.sqrt(np.mean(loud**2))) if len(loud) else 0.0


def matching_gain(converted: np.ndarray, original: np.ndarray, rate: int) -> float:
    """変換した声を、元の声と同じくらいの大きさにする倍率（極端な値にはしない）。"""
    target = voiced_rms(original, rate)
    current = voiced_rms(converted, rate)
    if target <= 0 or current <= 0:
        return 1.0
    return float(np.clip(target / current, 10 ** (-12 / 20), 10 ** (12 / 20)))


def impulse_response(rate: int, seconds: float = 1.8, decay_seconds: float = 1.2, seed: int = 7) -> np.ndarray:
    """部屋の響きを作るための、左右で少し違うインパルス応答（ノイズを指数的に減衰させたもの）。"""
    rng = np.random.default_rng(seed)
    length = int(rate * seconds)
    t = np.arange(length) / rate
    # RT60（響きが 60dB 小さくなるまでの時間）が decay_seconds になるように減衰させる
    envelope = np.exp(-6.9078 * t / decay_seconds)
    from scipy.signal import lfilter

    noise = rng.standard_normal((length, 2))
    # 高い音ほど早く消えるように、明るい響きから暗い響きへ、時間とともに移り変わらせる
    bright = lfilter([1 - 0.35], [1, -0.35], noise, axis=0)
    dark = lfilter([1 - 0.95], [1, -0.95], noise, axis=0) * 3.0
    weight = np.minimum(t / decay_seconds, 1.0)[:, None]
    ir = (bright * (1 - weight) + dark * weight) * envelope[:, None]
    predelay = int(rate * 0.02)
    ir = np.concatenate([np.zeros((predelay, 2)), ir])
    ir /= np.sqrt(np.sum(ir**2, axis=0, keepdims=True)) + 1e-9
    return ir.astype(np.float32)


def add_reverb(dry: np.ndarray, rate: int, amount: float) -> np.ndarray:
    """声に残響を付ける。amount は 0（なし）〜1（多め）。"""
    if amount <= 0:
        return dry
    from scipy.signal import fftconvolve

    mono = dry.mean(axis=1)
    ir = impulse_response(rate)
    wet = np.stack([fftconvolve(mono, ir[:, channel])[: len(mono)] for channel in range(2)], axis=1)
    # amount 0.3 で約 -16dB、1.0 で約 -8dB の響き
    wet_gain = 10 ** ((-20 + 12 * amount) / 20)
    return (to_channels(dry, 2) + wet.astype(np.float32) * wet_gain).astype(np.float32)


def limit_peak(data: np.ndarray) -> tuple[np.ndarray, float]:
    """音割れしないよう、大きすぎるときは全体を下げる。下げた量（dB、0 なら下げていない）も返す。"""
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= PEAK_LIMIT:
        return data, 0.0
    gain = PEAK_LIMIT / peak
    return (data * gain).astype(np.float32), round(20 * np.log10(gain), 2)


def mix_song(
    lead_converted: Path,
    lead_original: Path,
    instrumental: Path,
    backing: Path | None,
    *,
    reverb: float,
    vocal_gain_db: float,
    instrumental_gain_db: float,
    backing_gain_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """(曲, 変換した声だけ, 伴奏だけ, サンプルレート, 情報) を返す。"""
    inst, rate = read(instrumental)
    inst = to_channels(inst, 2)
    frames = len(inst)

    original, original_rate = read(lead_original)
    original = resample(original, original_rate, rate)
    converted, converted_rate = read(lead_converted)
    converted = resample(converted, converted_rate, rate)
    converted = fit_length(to_channels(converted, 2), frames)
    gain = matching_gain(converted, fit_length(to_channels(original, 2), frames), rate)
    vocals = converted * gain * (10 ** (vocal_gain_db / 20))
    vocals = fit_length(add_reverb(vocals, rate, reverb), frames)

    music = inst * (10 ** (instrumental_gain_db / 20))
    if backing is not None:
        back, back_rate = read(backing)
        back = fit_length(to_channels(resample(back, back_rate, rate), 2), frames)
        if reverb > 0:
            back = fit_length(add_reverb(back, rate, reverb * 0.8), frames)
        music = music + back * (10 ** (backing_gain_db / 20))

    song, reduced = limit_peak(music + vocals)
    info = {"vocal_match_db": round(20 * np.log10(gain), 2), "peak_reduced_db": reduced}
    return song, limit_peak(vocals)[0], limit_peak(music)[0], rate, info
