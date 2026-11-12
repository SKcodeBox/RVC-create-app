import pytest

from rvc_api import evaluation, library, settings
from tests.conftest import write_checkpoint


def make_model(name="voice", epochs=(10, 20, 30), checkpoint=True):
    folder = library.model_dir(name)
    folder.mkdir(parents=True)
    for epoch in epochs:
        (folder / f"{name}_{epoch}e_{epoch * 5}s.pth").write_bytes(b"w" * 100)
    if checkpoint:
        write_checkpoint(folder / "G_2333333.pth", max(epochs, default=0))
        write_checkpoint(folder / "D_2333333.pth", max(epochs, default=0))
        (folder / "filelist.txt").write_text("x", encoding="utf-8")
        (folder / "sliced_audios").mkdir()
        (folder / "sliced_audios" / "0_0_0.wav").write_bytes(b"s" * 500)
    library.write_meta(name, source="trained", created_at=1.0)
    return folder


def test_recommended_weight_uses_lowest_validation_loss():
    folder = make_model()
    evaluation.write_evaluations(folder, [
        {"epoch": 10, "val_loss": 5.0}, {"epoch": 20, "val_loss": 3.0}, {"epoch": 30, "val_loss": 4.0},
    ])
    summary = library.model_summary("voice")
    assert summary["recommended_epoch"] == 20
    assert summary["latest_weight"].startswith("voice_30e")


def test_delete_weight_rules():
    make_model()
    with pytest.raises(library.LibraryError):
        library.delete_weight("voice", "voice_30e_150s.pth")  # 最新は続きから学習に必要
    library.delete_weight("voice", "voice_10e_50s.pth")
    assert [w["epoch"] for w in library.list_weights("voice")] == [20, 30]
    with pytest.raises(library.LibraryError):
        library.delete_weight("voice", "../voice_20e_100s.pth")
    with pytest.raises(library.LibraryError):
        library.delete_weight("voice", "G_2333333.pth")


def test_cannot_delete_last_weight():
    make_model(epochs=(10,), checkpoint=False)
    with pytest.raises(library.LibraryError):
        library.delete_weight("voice", "voice_10e_50s.pth")


def test_cleanup_keeps_recommended_and_latest():
    folder = make_model(epochs=(10, 20, 30, 40))
    evaluation.write_evaluations(folder, [{"epoch": 20, "val_loss": 1.0}, {"epoch": 40, "val_loss": 2.0}])
    before = library.storage("voice")
    assert before["extra_weight_count"] == 2
    result = library.cleanup("voice", weights=True)
    assert result["freed"] == 200
    assert [w["epoch"] for w in library.list_weights("voice")] == [20, 40]
    result = library.cleanup("voice", checkpoints=True)
    assert result["freed"] >= 2500
    summary = library.model_summary("voice")
    assert not summary["can_resume"]
    assert not (folder / "sliced_audios").exists()


def test_stage_file_is_safe(data_dir):
    stages = library.dataset_dir("voice") / library.STAGES_DIR / "song"
    stages.mkdir(parents=True)
    (stages / "vocals.flac").write_bytes(b"f")
    make_model()
    assert library.stage_file("voice", "song.wav", "vocals").name == "vocals.flac"
    with pytest.raises(library.LibraryError):
        library.stage_file("voice", "song.wav", "secret")
    with pytest.raises(library.LibraryError):
        library.stage_file("voice", "../../x.wav", "vocals")


def test_target_pitch():
    make_model()
    assert library.target_pitch("voice") is None
    library.write_meta("voice", voice_type="high")
    assert library.target_pitch("voice") == 255.0
    library.write_meta("voice", pitch={"median": 180.5})
    assert library.target_pitch("voice") == 180.5


def test_validate_names():
    assert library.validate_name("my-voice_1") == "my-voice_1"
    for bad in ("", "-x", "a" * 41, "名前", "reference"):
        with pytest.raises(library.LibraryError):
            library.validate_name(bad)
    with pytest.raises(library.LibraryError):
        library.validate_embedder("unknown")


def test_windows_reserved_names_are_rejected():
    for bad in ("CON", "nul", "Aux", "com1", "LPT9", "prn"):
        with pytest.raises(library.LibraryError):
            library.validate_name(bad)
    assert library.validate_name("console") == "console"


def test_resolve_model_uses_real_folder_name():
    make_model("Voice", checkpoint=False)
    assert library.resolve_model("Voice") == "Voice"
    case_insensitive = (settings.MODELS_DIR / "VOICE").exists()
    if case_insensitive:  # Mac・Windows: 大文字小文字が違っても、本当の名前にそろえる
        assert library.resolve_model("voice") == "Voice"
    else:
        with pytest.raises(library.LibraryError):
            library.resolve_model("voice")


def test_checkpoint_state_reads_epoch_without_loading_tensors():
    make_model(epochs=(10, 20))
    state = library.checkpoint_state("voice")
    assert state == {"exists": True, "ok": True, "epoch": 20}
    summary = library.model_summary("voice")
    assert summary["can_resume"] and not summary["checkpoint_broken"]
    assert library.checkpoint_epoch("voice") == 20


def test_checkpoint_newer_than_weights():
    # 変換用のモデルを書き出す前に止まった: 再開用のファイルは 21 エポック、モデルは 20 エポックまで
    folder = make_model(epochs=(10, 20))
    write_checkpoint(folder / "G_2333333.pth", 21)
    write_checkpoint(folder / "D_2333333.pth", 21)
    summary = library.model_summary("voice")
    assert summary["trained_epochs"] == 21 and library.checkpoint_epoch("voice") == 21


def test_truncated_checkpoint_is_broken():
    folder = make_model(epochs=(10, 20))
    data = (folder / "D_2333333.pth").read_bytes()
    (folder / "D_2333333.pth").write_bytes(data[: len(data) // 2])  # 保存の途中で止まった
    state = library.checkpoint_state("voice")
    assert state["exists"] and not state["ok"]
    summary = library.model_summary("voice")
    assert not summary["can_resume"] and summary["checkpoint_broken"]
    assert library.checkpoint_epoch("voice") == 0


def test_checkpoint_reader_does_not_run_code(tmp_path):
    import pickle
    import zipfile

    class Evil:
        def __reduce__(self):
            return (exec, ("raise SystemExit('executed')",))

    path = tmp_path / "G_1.pth"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"iteration": 3, "x": Evil()}))
    assert library._checkpoint_iteration(path) == 3


def test_real_torch_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "G_2333333.pth"
    torch.save({"model": {"w": torch.zeros(4, 4)}, "iteration": 42, "optimizer": {}, "learning_rate": 1e-4}, path)
    assert library._checkpoint_iteration(path) == 42


def test_cleanup_removes_partial_saves():
    folder = make_model(epochs=(10, 20))
    (folder / "G_2333333.pth.tmp").write_bytes(b"x" * 10)
    library.cleanup("voice", checkpoints=True)
    assert not list(folder.glob("*.tmp")) and not list(folder.glob("[GD]_*.pth"))
