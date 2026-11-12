import pytest

from rvc_api import library, pipelines, settings


@pytest.mark.parametrize(
    "source,target,song,expected",
    [
        (120, 240, False, 12),  # 1オクターブ上
        (120, 240, True, 12),
        (200, 300, False, 7),  # 半音7つ上（話し声はそのまま）
        (200, 300, True, 12),  # 曲はキーを変えないようオクターブ単位（+7 は +12 に近い）
        (200, 250, True, 0),  # +4 は 0 に近い
        (300, 150, True, -12),
        (50, 1000, False, 24),  # 大きすぎる変化は ±24 まで
    ],
)
def test_auto_pitch_shift(source, target, song, expected):
    assert pipelines.auto_pitch_shift(source, target, octave_only=song) == expected


def test_index_path_avoids_trained_substring(tmp_path):
    name = "pretrained-voice"  # Applio は "trained" を "added" に書き換えてしまう
    folder = library.model_dir(name)
    folder.mkdir(parents=True)
    (folder / f"{name}.index").write_bytes(b"index-data")
    work = settings.UPLOADS_DIR / "job1"
    work.mkdir(parents=True)
    path = pipelines._index_for(name, work)
    assert "trained" not in path
    with open(path, "rb") as handle:
        assert handle.read() == b"index-data"


def test_index_path_kept_when_safe():
    name = "voice"
    folder = library.model_dir(name)
    folder.mkdir(parents=True)
    (folder / f"{name}.index").write_bytes(b"x")
    assert pipelines._index_for(name, settings.UPLOADS_DIR) == str(folder / f"{name}.index")


def test_stage_tracker_progress():
    class FakeJob:
        stage = None
        stage_label = None
        progress = None

    job = FakeJob()
    tracker = pipelines.StageTracker(job, ["prepare", "convert"])
    tracker.enter("prepare")
    tracker.update(1.0)
    first = job.progress
    tracker.enter("convert")
    tracker.update(0.5)
    assert 0 < first < job.progress < 1



def test_sweep_orphan_files():
    from rvc_api.jobs import Job

    keep = Job(id="aaaaaaaaaaaa", kind="convert", params={"upload_id": "up1"})
    (settings.OUTPUTS_DIR / "aaaaaaaaaaaa.wav").write_bytes(b"k")
    (settings.OUTPUTS_DIR / "aaaaaaaaaaaa_vocals.wav").write_bytes(b"k")
    (settings.OUTPUTS_DIR / "bbbbbbbbbbbb.mp3").write_bytes(b"x" * 10)  # 履歴から消えたもの
    (settings.UPLOADS_DIR / "up1").mkdir()
    (settings.UPLOADS_DIR / "up2").mkdir()
    (settings.UPLOADS_DIR / "up2" / "input.wav").write_bytes(b"x" * 20)
    (settings.UPLOADS_DIR / "separate-cccccccccccc").mkdir()  # 前回落ちたときの作業用フォルダ
    freed = pipelines.sweep_orphan_files([keep])
    assert freed == 30
    assert sorted(p.name for p in settings.OUTPUTS_DIR.iterdir()) == ["aaaaaaaaaaaa.wav", "aaaaaaaaaaaa_vocals.wav"]
    assert [p.name for p in settings.UPLOADS_DIR.iterdir()] == ["up1"]


def test_failed_conversion_removes_work_files(monkeypatch):
    from rvc_api.jobs import Job, JobFailed

    work = settings.UPLOADS_DIR / "up1"
    (work / "separated" / "final").mkdir(parents=True)
    (work / "input.wav").write_bytes(b"i")
    (work / "mix.wav").write_bytes(b"m")
    (work / "converted.wav").write_bytes(b"c")
    job = Job(id="dddddddddddd", kind="convert", params={"input_path": str(work / "input.wav")})
    (settings.OUTPUTS_DIR / "dddddddddddd.wav").write_bytes(b"partial")

    def fail(job, jm):
        raise JobFailed("失敗")

    monkeypatch.setattr(pipelines, "_convert", fail)
    with pytest.raises(JobFailed):
        pipelines.run_conversion(job, None)
    # 元の音声だけ残り、大きな作業用ファイルと書きかけの結果は消える
    assert [p.name for p in work.iterdir()] == ["input.wav"]
    assert not list(settings.OUTPUTS_DIR.glob("dddddddddddd*"))


def test_infer_accepts_finished_output_after_abnormal_exit(tmp_path, tone):
    """変換を書き出した後でプロセスが異常終了しても（ライブラリの後片付けで落ちるなど）、結果は使う。"""
    from rvc_api.jobs import Job

    job = Job(id="eeeeeeeeeeee", kind="convert", params={})
    target = tmp_path / "converted.wav"

    class FakeManager:
        def __init__(self, write: bool, finished: bool):
            self.write, self.finished = write, finished

        def run(self, job, command, on_line=None, **kwargs):
            if self.write:
                tone(target, seconds=0.5)
            if self.finished:
                on_line(f"Conversion completed at '{target}' in 1.0 seconds.")
            return -11  # SIGSEGV

    params = {"index_rate": 0.75, "volume_envelope": 1.0, "protect": 0.33, "f0_method": "rmvpe"}
    pipelines._infer(job, FakeManager(True, True), params, tmp_path / "m.pth", "", "contentvec", tmp_path / "in.wav", target, 0)
    assert any("異常終了" in line for line in job.logs)
    for write, finished in ((True, False), (False, True)):
        with pytest.raises(pipelines.JobFailed):
            pipelines._infer(job, FakeManager(write, finished), params, tmp_path / "m.pth", "", "contentvec", tmp_path / "in.wav", target, 0)
