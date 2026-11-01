"""Exercise checkpoint selection and inference-only Hub downloads offline."""

import fnmatch
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import hf_io


@pytest.fixture
def hub(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    downloads = []

    def put(name, data="weights"):
        path = remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")

    def files():
        return [p.relative_to(remote).as_posix() for p in remote.rglob("*") if p.is_file()]

    api = SimpleNamespace(list_repo_files=lambda *a, **k: files())
    monkeypatch.setattr(hf_io, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(hf_io, "hf_hub_download", lambda repo, name, **kwargs: str(remote / name))

    def download(**kwargs):
        downloads.append(kwargs["allow_patterns"])
        for name in files():
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
                dest = Path(kwargs["local_dir"]) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remote / name, dest)

    monkeypatch.setattr(hf_io, "snapshot_download", download)
    return SimpleNamespace(remote=remote, put=put, downloads=downloads, api=api)


def stage_epochs(hub):
    for step in (1099, 2198):
        prefix = f"checkpoint-{step}"
        hub.put(f"{prefix}/model.safetensors")
        hub.put(f"{prefix}/config.json", "{}")
        hub.put(f"{prefix}/tokenizer_config.json", "{}")
        hub.put(f"{prefix}/optimizer.pt")
        hub.put(f"{prefix}/trainer_state.json", json.dumps({
            "best_model_checkpoint": "/content/run/checkpoint-1099",
            "log_history": [{"step": 1099, "eval_em": .194, "eval_f1": .4606}],
        }))


def arguments(tmp_path, spec="best"):
    return SimpleNamespace(repo="offline/test", dir=str(tmp_path / "local"),
                           ckpt=spec, info_out=str(tmp_path / "info.json"))


@pytest.mark.parametrize("stale_root", [False, True])
def test_best_uses_recorded_epoch_without_needing_final_root(hub, tmp_path, stale_root):
    stage_epochs(hub)
    if stale_root:
        hub.put("model.safetensors", "older root export")
    args = arguments(tmp_path)
    hf_io.prepare(args, token=None)
    info = json.loads(Path(args.info_out).read_text(encoding="utf-8"))
    assert info["requested_ckpt"] == "best" and info["ckpt"] == "checkpoint-1099"
    assert Path(info["model_dir"]) == Path(args.dir) / "checkpoint-1099"
    assert info["pred_path_in_repo"] == "predictions/checkpoint-1099"
    assert (Path(info["model_dir"]) / "tokenizer_config.json").is_file()
    assert not list(Path(args.dir).rglob("optimizer.pt"))
    assert not (Path(args.dir) / "checkpoint-2198" / "model.safetensors").exists()


def test_root_export_remains_usable_without_epoch_checkpoints(hub, tmp_path):
    hub.put("model.safetensors")
    args = arguments(tmp_path)
    hf_io.prepare(args, token=None)
    info = json.loads(Path(args.info_out).read_text())
    assert info["ckpt"] == "best" and info["model_dir"] == args.dir


def test_latest_is_explicit_and_does_not_substitute_recorded_best(hub, tmp_path):
    stage_epochs(hub)
    args = arguments(tmp_path, "latest")
    hf_io.prepare(args, token=None)
    info = json.loads(Path(args.info_out).read_text())
    assert info["ckpt"] == "checkpoint-2198"


def test_best_without_metadata_fails_before_empty_download(hub, tmp_path):
    hub.put("checkpoint-1099/model.safetensors")
    args = arguments(tmp_path)
    with pytest.raises(SystemExit, match="trainer_state.json"):
        hf_io.prepare(args, token=None)
    assert not hub.downloads and not Path(args.info_out).exists()


def test_local_best_checkpoint_works_when_hub_is_unavailable(hub, tmp_path):
    stage_epochs(hub)
    args = arguments(tmp_path)
    shutil.copytree(hub.remote, args.dir)

    def offline(*a, **kwargs):
        raise ConnectionError("offline")

    hub.api.list_repo_files = offline
    hf_io.prepare(args, token=None)
    info = json.loads(Path(args.info_out).read_text())
    assert info["ckpt"] == "checkpoint-1099" and not hub.downloads
