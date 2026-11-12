"""Applio に、このアプリで必要な修正を当てる（npm run rvc:setup が実行する。何度実行しても同じ結果になる）。

git がない Windows でも当てられるよう、差分ファイルではなく「置き換える文字列」で書いている。
使い方: python apply_patches.py <Applio のフォルダ>
"""

import sys
from pathlib import Path

# (ファイル, 説明, 元の文字列, 置き換え後の文字列)
PATCHES = [
    (
        "rvc/train/extract/extract.py",
        "Mac の GPU（MPS）は float64 を扱えないため、GPU に送る前に float32 にする",
        """        feats = (
            torch.from_numpy(load_audio(wav_file_path, SAMPLE_RATE_16K))
            .to(device)
            .float()
        )""",
        """        # MPS (Apple Silicon) cannot hold float64 tensors: convert before moving.
        feats = (
            torch.from_numpy(load_audio(wav_file_path, SAMPLE_RATE_16K))
            .float()
            .to(device)
        )""",
    ),
    (
        "rvc/train/utils.py",
        "学習の再開用ファイルを、別名で書き終えてから置き換える（保存中に止まっても壊れたファイルを残さない）",
        """        checkpoint_path,
    )

    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")""",
        """        f"{checkpoint_path}.tmp",
    )
    # Write to a temporary file first so that an interrupted save never leaves a
    # truncated checkpoint behind (RVC Studio patch).
    os.replace(f"{checkpoint_path}.tmp", checkpoint_path)

    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")""",
    ),
    (
        "rvc/train/process/extract_model.py",
        "変換用のモデルファイルも、別名で書き終えてから置き換える（書きかけのファイルが一覧に出ないように）",
        """            model_path,
        )

        print(f"Saved model '{model_path}' (epoch {epoch} and step {step})")""",
        """            f"{model_path}.tmp",
        )
        # Write to a temporary file first so that an interrupted save never leaves a
        # truncated model behind (RVC Studio patch).
        os.replace(f"{model_path}.tmp", model_path)

        print(f"Saved model '{model_path}' (epoch {epoch} and step {step})")""",
    ),
    (
        "rvc/train/process/extract_index.py",
        "インデックスを Python で書き出す（faiss は Windows で日本語などを含む場所に書けないため）",
        """    faiss.write_index(index_added, index_filepath_added)""",
        """    # faiss.write_index opens the path with C fopen, which fails on Windows when the
    # path has non-ASCII characters. Serialize in memory and let Python write the file
    # (RVC Studio patch). Write to a temporary file first so a crash leaves no broken index.
    with open(f"{index_filepath_added}.tmp", "wb") as index_file:
        faiss.serialize_index(index_added).tofile(index_file)
    os.replace(f"{index_filepath_added}.tmp", index_filepath_added)""",
    ),
    (
        "rvc/infer/pipeline.py",
        "インデックスを Python で読み込む（faiss は Windows で日本語などを含む場所を読めないため）",
        """                index = faiss.read_index(file_index)""",
        """                # faiss.read_index opens the path with C fopen, which fails on Windows when
                # the path has non-ASCII characters. Read it with Python instead (RVC Studio patch).
                with open(file_index, "rb") as index_file:
                    index = faiss.deserialize_index(np.fromfile(index_file, dtype=np.uint8))""",
    ),
    # onnxruntime は、このアプリで使わない機能（swift のピッチ検出・リアルタイム変換）だけが使う。
    # 読み込むたびに Microsoft へ利用状況を送る処理が動き、終了時に落ちることもあるため、入れない
    (
        "requirements-engine.txt",
        "onnxruntime を入れない（使わない機能専用で、Microsoft へ利用状況を送る処理を含むため）",
        """onnx==1.23.1
onnxruntime>=1.17.0; sys_platform != "win32\"""",
        """onnx==1.23.1
# onnxruntime is only used by SwiftF0 and realtime conversion, which RVC Studio does not use,
# and it sends usage data to Microsoft. Not installed (RVC Studio patch).""",
    ),
    (
        "requirements-engine.txt",
        "Windows 用の onnxruntime と、それだけが使う NVIDIA の部品（約1.7GB）を入れない（PyTorch は自分の分を持っている）",
        """onnxruntime-gpu==1.22.0; sys_platform == "win32"
nvidia-cublas-cu12==12.9.2.10; sys_platform == "win32"
nvidia-cuda-runtime-cu12==12.9.79; sys_platform == "win32"
nvidia-cudnn-cu12==9.26.0.51; sys_platform == "win32"
nvidia-cufft-cu12==11.4.1.4; sys_platform == "win32\"""",
        """# onnxruntime-gpu and the NVIDIA libraries below were only needed by onnxruntime-gpu.
# PyTorch for Windows bundles its own CUDA libraries. Not installed (RVC Studio patch).""",
    ),
    (
        "rvc/lib/predictors/f0.py",
        "onnxruntime を使う swift のピッチ検出は、使うときだけ読み込む（onnxruntime を入れなくても動くように）",
        """from rvc.lib.predictors.Swift import SwiftPredictor
""",
        """# SwiftPredictor needs onnxruntime, which RVC Studio does not install: it is imported
# only when SWIFT is used (RVC Studio patch).
""",
    ),
    (
        "rvc/lib/predictors/f0.py",
        "swift のピッチ検出を使うときに、onnxruntime を読み込む",
        """        model_path = os.path.join("rvc", "models", "predictors", model_name)
        self.model = SwiftPredictor(""",
        """        model_path = os.path.join("rvc", "models", "predictors", model_name)
        from rvc.lib.predictors.Swift import SwiftPredictor  # needs onnxruntime

        self.model = SwiftPredictor(""",
    ),
]


def main() -> int:
    root = Path(sys.argv[1])
    failed = False
    for relative, description, old, new in PATCHES:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if new in text:
            print(f"  当て済み: {relative}（{description}）")
            continue
        if text.count(old) != 1:
            print(f"  当てられませんでした: {relative}（Applio のバージョンが違う可能性があります）", file=sys.stderr)
            failed = True
            continue
        path.write_text(text.replace(old, new), encoding="utf-8", newline="")
        print(f"  当てました: {relative}（{description}）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
