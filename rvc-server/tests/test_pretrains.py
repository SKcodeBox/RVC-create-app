import pytest

from rvc_api import pretrains, settings


def test_catalog_entries():
    for key, entry in pretrains.CATALOG.items():
        assert entry["name"] and entry["description"]
        for rate, (g, d) in entry["rates"].items():
            assert rate in settings.SAMPLE_RATES
            assert "/resolve/" in g and "/resolve/" in d
            assert g.endswith(".pth") and d.endswith(".pth")
    listed = pretrains.catalog()
    assert listed[0]["id"] == "standard"


def test_validate():
    assert pretrains.validate(None, 40000) == "standard"
    assert pretrains.validate("titan", 48000) == "titan"
    with pytest.raises(pretrains.PretrainedError):
        pretrains.validate("ov2super", 48000)  # 48kHz 版はない
    with pytest.raises(pretrains.PretrainedError):
        pretrains.validate("nothing", 40000)


def test_paths_are_inside_data_dir():
    g, d = pretrains.paths("titan", 40000)
    assert settings.PRETRAINED_DIR in g.parents and g.name.startswith("G-") and d.name.startswith("D-")
