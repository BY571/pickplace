import json

from pickplace import artifacts as A


def test_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOD_ROBOT_ARTIFACTS", str(tmp_path))
    assert A.artifacts_root() == tmp_path


def test_git_commit_prefers_env(monkeypatch):
    monkeypatch.setenv("FOOD_ROBOT_GIT_COMMIT", "abc123-dirty")
    assert A.git_commit() == "abc123-dirty"


def test_json_roundtrip_update_and_sha(tmp_path):
    p = tmp_path / "m.json"
    A.write_json(p, {"a": 1})
    assert A.update_json(p, b=2) == {"a": 1, "b": 2}
    assert json.loads(p.read_text()) == {"a": 1, "b": 2}
    assert not (tmp_path / "m.json.tmp").exists()
    assert len(A.sha256_file(p)) == 64


def test_new_run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOD_ROBOT_ARTIFACTS", str(tmp_path))
    d = A.new_run_dir("teachers", "t1")
    assert d == tmp_path / "teachers" / "t1" and d.is_dir()
    auto = A.new_run_dir("teachers")
    assert auto.parent == tmp_path / "teachers" and auto.name.startswith("teachers_")
