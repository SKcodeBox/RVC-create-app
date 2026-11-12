import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER / "patches"))
import apply_patches  # noqa: E402

APPLIO = SERVER / "applio"


def pristine(relative: str) -> str | None:
    """パッチを当てる前の Applio のファイル（git があればそこから、なければ当てた後のものを元に戻して作る）。"""
    if (APPLIO / ".git").exists() and shutil.which("git"):
        result = subprocess.run(["git", "-C", str(APPLIO), "show", f"HEAD:{relative}"], capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            return result.stdout
    current = APPLIO / relative
    if not current.is_file():
        return None
    text = current.read_text(encoding="utf-8")
    for patch_relative, _description, old, new in apply_patches.PATCHES:
        if patch_relative == relative:
            text = text.replace(new, old)
    return text


def test_patches_apply_once(tmp_path):
    originals = {relative: pristine(relative) for relative, *_ in apply_patches.PATCHES}
    if any(text is None for text in originals.values()):
        pytest.skip("Applio がありません")
    for relative, text in originals.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    sys.argv = ["apply_patches.py", str(tmp_path)]
    assert apply_patches.main() == 0
    patched = {relative: (tmp_path / relative).read_text(encoding="utf-8") for relative in originals}
    for relative, _description, _old, new in apply_patches.PATCHES:
        assert patched[relative].count(new) == 1
    # 2回目は何も変えない
    assert apply_patches.main() == 0
    assert {relative: (tmp_path / relative).read_text(encoding="utf-8") for relative in originals} == patched


def test_patched_files_still_compile(tmp_path):
    """パッチを当てた後の Applio のファイルが、Python として正しいこと。"""
    for relative, *_ in apply_patches.PATCHES:
        if not relative.endswith(".py"):
            continue
        path = APPLIO / relative
        if not path.is_file():
            pytest.skip("Applio がありません")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_onnxruntime_is_not_installed():
    """修正を当てた後の Applio のインストール一覧に、onnxruntime と、それだけが使う NVIDIA の部品がないこと。"""
    relative = "requirements-engine.txt"
    text = pristine(relative)
    if text is None:
        pytest.skip("Applio がありません")
    assert "onnxruntime" in text  # 元の一覧には入っている
    for patch_relative, _description, old, new in apply_patches.PATCHES:
        if patch_relative == relative:
            text = text.replace(old, new)
    requirements = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not [line for line in requirements if line.startswith(("onnxruntime", "nvidia-"))]
    assert "onnx==1.23.1" in requirements and "torch==2.11.0" in requirements
