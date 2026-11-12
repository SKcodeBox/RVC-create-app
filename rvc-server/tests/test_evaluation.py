import json

import numpy as np

from rvc_api import evaluation


def make_experiment(tmp_path, clips: int):
    model = tmp_path / "voice"
    sliced = model / "sliced_audios"
    sliced.mkdir(parents=True)
    lines = []
    for index in range(clips):
        name = f"0_{index // 4}_{index % 4}"
        (sliced / f"{name}.wav").write_bytes(b"")
        lines.append(f"{(sliced / name).as_posix()}.wav|feat|f0|f0nsf|0")
    lines.append("/somewhere/mute/sliced_audios/mute40000.wav|m|m|m|0")
    (model / "filelist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model


def test_split_validation_holds_out_a_few_percent(tmp_path):
    model = make_experiment(tmp_path, 200)
    held = evaluation.split_validation(model)
    assert len(held) == 8  # 200 × 4%
    kept = (model / "filelist.txt").read_text(encoding="utf-8").splitlines()
    assert len(kept) == 201 - 8
    assert any("mute" in line for line in kept)
    assert not any(f"/{name}.wav|" in line for line in kept for name in held)
    # 2回目（続きから学習）は同じものを返し、一覧は変えない
    assert evaluation.split_validation(model) == held
    assert (model / "filelist.txt").read_text(encoding="utf-8").splitlines() == kept


def test_split_validation_skips_small_datasets(tmp_path):
    model = make_experiment(tmp_path, 10)
    assert evaluation.split_validation(model) == []
    assert len((model / "filelist.txt").read_text(encoding="utf-8").splitlines()) == 11


def test_dataset_pitch(tmp_path):
    folder = tmp_path / "f0_voiced"
    folder.mkdir()
    np.save(folder / "a.wav.npy", np.array([0, 200, 210, 190] * 50, dtype=np.float32))
    pitch = evaluation.dataset_pitch(tmp_path)
    assert 195 <= pitch["median"] <= 205


def evals(values, every=10):
    return [{"epoch": every * (i + 1), "val_loss": v} for i, v in enumerate(values)]


def test_should_stop_after_patience():
    # 50エポック目が一番良く、そのあと良くならない
    values = [10, 8, 7, 6.5, 6.0, 6.0, 6.05, 6.1, 6.2, 6.1, 6.3]
    stop, best = evaluation.should_stop(evals(values), save_every=10)
    assert best == 50 and stop  # 110 - 50 >= 50


def test_should_not_stop_while_improving():
    values = [10, 9, 8, 7, 6, 5.5, 5.2, 5.0, 4.8]
    stop, best = evaluation.should_stop(evals(values), save_every=10)
    assert not stop and best == 90


def test_should_not_stop_too_early():
    stop, _ = evaluation.should_stop(evals([5, 5, 5], every=10), save_every=10, min_epochs=40)
    assert not stop


def test_small_improvements_do_not_reset_patience():
    values = [6.0] + [6.0 - 0.001 * i for i in range(1, 7)]
    stop, best = evaluation.should_stop(evals(values), save_every=10)
    assert best == 10 and stop


def test_best_evaluation_respects_available_epochs():
    items = evals([5, 3, 4])
    assert evaluation.best_evaluation(items)["epoch"] == 20
    assert evaluation.best_evaluation(items, {10, 30})["epoch"] == 30
    assert evaluation.best_evaluation([{"epoch": 10, "val_loss": None}]) is None


def test_write_read_forget(tmp_path):
    evaluation.write_evaluations(tmp_path, evals([3, 2, 1]))
    sample = evaluation.sample_path(tmp_path, 20)
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"x")
    evaluation.forget_epochs(tmp_path, {20})
    assert [item["epoch"] for item in evaluation.read_evaluations(tmp_path)] == [10, 30]
    assert not sample.exists()
    assert json.loads(evaluation.eval_path(tmp_path).read_text())["evaluations"]


def test_training_curve_skips_large_records(tmp_path):
    """学習の記録から数値だけを読む（試し聴きの音声などの大きな記録は飛ばす）。"""
    pytest = __import__("pytest")
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    log_dir = tmp_path / "eval"
    writer = EventFileWriter(str(log_dir))
    for step in (10, 20, 30):
        for tag, value in (("loss/g/mel", 30.0 - step / 10), ("loss/d/adv", 4.0), ("other", 1.0)):
            summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)])
            writer.add_event(event_pb2.Event(step=step, summary=summary))
        audio = summary_pb2.Summary.Audio(sample_rate=40000, encoded_audio_string=b"\0" * 200_000)
        writer.add_event(event_pb2.Event(step=step, summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag="gen/audio", audio=audio)])))
    writer.close()
    curve = evaluation.training_curve(tmp_path, steps_per_epoch=10)
    assert curve["mel"] == [[1, 29.0], [2, 28.0], [3, 27.0]]
    assert curve["discriminator"] == [[1, 4.0], [2, 4.0], [3, 4.0]]
    assert "generator" not in curve
