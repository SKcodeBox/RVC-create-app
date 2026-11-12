"""学習の出発点にする事前学習モデル（RVC 標準と、コミュニティの改良版）。

改良版は Applio が一覧にしているもの（https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json）
から、HiFi-GAN 用で、よく使われているものを選んでいる。初めて使うときにダウンロードする（1つ約1.3GB）。
"""

import threading
from pathlib import Path

from . import settings

HF_BASE = "https://huggingface.co/"
STANDARD = "standard"

# id: (表示名, 説明, {サンプルレート: (G のパス, D のパス)})
CATALOG: dict[str, dict] = {
    STANDARD: {
        "name": "RVC 標準",
        "description": "RVC に付属する事前学習モデル。どの声にも安定して使えます",
        "rates": {},
    },
    "titan": {
        "name": "TITAN",
        "description": "Applio の開発者（blaise-tk）が公開している改良版",
        "rates": {
            32000: ("blaise-tk/TITAN/resolve/main/models/medium/32k/pretrained/G-f032k-TITAN-Medium.pth",
                    "blaise-tk/TITAN/resolve/main/models/medium/32k/pretrained/D-f032k-TITAN-Medium.pth"),
            40000: ("blaise-tk/TITAN/resolve/main/models/medium/40k/pretrained/G-f040k-TITAN-Medium.pth",
                    "blaise-tk/TITAN/resolve/main/models/medium/40k/pretrained/D-f040k-TITAN-Medium.pth"),
            48000: ("blaise-tk/TITAN/resolve/main/models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth",
                    "blaise-tk/TITAN/resolve/main/models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth"),
        },
    },
    "snowie": {
        "name": "Snowie V3.1",
        "description": "MUSTAR が公開している改良版",
        "rates": {
            32000: ("MUSTAR/SnowieV3.1-32k/resolve/main/G_SnowieV3.1_32k.pth",
                    "MUSTAR/SnowieV3.1-32k/resolve/main/D_SnowieV3.1_32k.pth"),
            40000: ("MUSTAR/SnowieV3.1-40k/resolve/main/G_SnowieV3.1_40k.pth",
                    "MUSTAR/SnowieV3.1-40k/resolve/main/D_SnowieV3.1_40k.pth"),
            48000: ("MUSTAR/SnowieV3.1-48k/resolve/main/G_SnowieV3.1_48k.pth",
                    "MUSTAR/SnowieV3.1-48k/resolve/main/D_SnowieV3.1_48k.pth"),
        },
    },
    "klm4": {
        "name": "KLM 4",
        "description": "SeoulStreamingStation が公開している改良版",
        "rates": {
            32000: ("SeoulStreamingStation/KLM49_HFG/resolve/main/G_KLM_HFG_32k.pth",
                    "SeoulStreamingStation/KLM49_HFG/resolve/main/D_KLM_HFG_32k.pth"),
            40000: ("SeoulStreamingStation/KLM49_HFG/resolve/main/G_KLM_HFG_40k.pth",
                    "SeoulStreamingStation/KLM49_HFG/resolve/main/D_KLM_HFG_40k.pth"),
            48000: ("SeoulStreamingStation/KLM49_HFG/resolve/main/G_KLM_HFG_48k.pth",
                    "SeoulStreamingStation/KLM49_HFG/resolve/main/D_KLM_HFG_48k.pth"),
        },
    },
    "ov2super": {
        "name": "Ov2 Super",
        "description": "ORVC が公開している改良版（32kHz・40kHz のみ）",
        "rates": {
            32000: ("ORVC/Ov2Super/resolve/main/f0Ov2Super32kG.pth",
                    "ORVC/Ov2Super/resolve/main/f0Ov2Super32kD.pth"),
            40000: ("ORVC/Ov2Super/resolve/main/f0Ov2Super40kG.pth",
                    "ORVC/Ov2Super/resolve/main/f0Ov2Super40kD.pth"),
        },
    },
}

_lock = threading.Lock()


class PretrainedError(Exception):
    """利用者に見せるメッセージを持つエラー。"""


def supports(pretrained: str, sample_rate: int) -> bool:
    if pretrained == STANDARD:
        return sample_rate in settings.SAMPLE_RATES
    entry = CATALOG.get(pretrained)
    return bool(entry) and sample_rate in entry["rates"]


def validate(pretrained: str | None, sample_rate: int) -> str:
    pretrained = (pretrained or STANDARD).strip()
    if pretrained not in CATALOG:
        raise PretrainedError("事前学習モデルの指定が正しくありません。")
    if not supports(pretrained, sample_rate):
        raise PretrainedError(
            f"「{CATALOG[pretrained]['name']}」は {sample_rate // 1000}kHz に対応していません。"
            "サンプルレートを変えるか、別の事前学習モデルを選んでください。"
        )
    return pretrained


def paths(pretrained: str, sample_rate: int) -> tuple[Path, Path]:
    """事前学習モデル（生成器 G と識別器 D）を置く場所。"""
    if pretrained == STANDARD:
        return settings.standard_pretrained_paths(sample_rate)
    g_url, d_url = CATALOG[pretrained]["rates"][sample_rate]
    folder = settings.PRETRAINED_DIR / pretrained / str(sample_rate)
    return folder / Path(g_url).name, folder / Path(d_url).name


def is_downloaded(pretrained: str, sample_rate: int) -> bool:
    return all(path.is_file() for path in paths(pretrained, sample_rate))


def catalog() -> list[dict]:
    """画面に出す一覧（ダウンロード済みかどうかつき）。"""
    items = []
    for key, entry in CATALOG.items():
        rates = list(settings.SAMPLE_RATES) if key == STANDARD else sorted(entry["rates"])
        items.append(
            {
                "id": key,
                "name": entry["name"],
                "description": entry["description"],
                "sample_rates": rates,
                "downloaded": {str(rate): is_downloaded(key, rate) for rate in rates},
            }
        )
    return items


def download(pretrained: str, sample_rate: int, on_progress, cancelled) -> None:
    """コミュニティの事前学習モデルをダウンロードする。on_progress(0〜1)、cancelled() で中止。"""
    import requests  # サーバーの起動を軽くするため、使うときに読み込む

    g_url, d_url = CATALOG[pretrained]["rates"][sample_rate]
    targets = list(zip((g_url, d_url), paths(pretrained, sample_rate)))
    with _lock:
        sizes: list[int] = []
        for url, _ in targets:
            try:
                head = requests.head(HF_BASE + url, allow_redirects=True, timeout=60)
                sizes.append(int(head.headers.get("content-length") or 0))
            except requests.RequestException:
                sizes.append(0)
        total = sum(sizes) or 1
        done = 0
        for (url, target), size in zip(targets, sizes):
            if target.is_file() and (not size or target.stat().st_size == size):
                done += size
                on_progress(done / total)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".part")
            try:
                with requests.get(HF_BASE + url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with temp.open("wb") as out:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if cancelled():
                                raise PretrainedError("キャンセルしました。")
                            out.write(chunk)
                            done += len(chunk)
                            on_progress(min(done / total, 1.0))
                if size and temp.stat().st_size != size:
                    raise PretrainedError("事前学習モデルのダウンロードが途中で切れました。もう一度お試しください。")
                temp.replace(target)
            except requests.RequestException:
                raise PretrainedError(
                    "事前学習モデルをダウンロードできませんでした。インターネット接続を確認して、もう一度お試しください。"
                )
            finally:
                temp.unlink(missing_ok=True)
