"""Local GGUF discovery.

`translate` used to fail with the model sitting in `models/`, because
MODEL_REGISTRY["docker"]["filename"] said `docker_gemma4_e2b_q4km.gguf` while the
file the README told people to download was `docker_gemma3_4b_q4km.gguf`. Exact
filename matching turned a one-character registry drift into "model not found".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlcli_wizard.model import ModelManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A ModelManager searching only inside tmp_path.

    The real search list includes a path relative to the installed package, so the
    developer's own `models/` would otherwise leak into every assertion here.
    """
    (tmp_path / "models").mkdir()
    (tmp_path / "cache").mkdir()

    m = ModelManager(cli_tool="docker")
    m.cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        m, "_search_dirs", lambda: [tmp_path / "models", m.cache_dir]
    )
    return m


def test_no_model_found_in_an_empty_tree(manager):
    assert manager._find_local_model() is None


def test_exact_registry_filename_is_found(manager, tmp_path):
    target = tmp_path / "models" / manager._get_model_filename()
    target.write_bytes(b"GGUF")
    assert manager._find_local_model() == target


def test_file_the_readme_names_is_found_even_if_the_registry_drifts(manager, tmp_path):
    """The regression this module exists for."""
    target = tmp_path / "models" / "docker_gemma3_4b_q4km.gguf"
    target.write_bytes(b"GGUF")
    assert manager._find_local_model() == target


def test_exact_filename_wins_over_the_tool_name_fallback(manager, tmp_path):
    exact = tmp_path / "models" / manager._get_model_filename()
    other = tmp_path / "models" / "docker_something_else.gguf"
    other.write_bytes(b"GGUF")
    exact.write_bytes(b"GGUF")
    assert manager._find_local_model() == exact


def test_a_gguf_for_another_tool_is_not_picked_up(manager, tmp_path):
    (tmp_path / "models" / "venvy_gemma3_q4km.gguf").write_bytes(b"GGUF")
    assert manager._find_local_model() is None


def test_non_gguf_files_are_ignored(manager, tmp_path):
    (tmp_path / "models" / "docker_notes.md").write_text("models folder")
    (tmp_path / "models" / "readme.md").write_text("models folder")
    assert manager._find_local_model() is None


def test_the_cache_directory_is_searched_too(manager):
    target = manager.cache_dir / "docker_gemma3_4b_q4km.gguf"
    target.write_bytes(b"GGUF")
    assert manager._find_local_model() == target


def test_models_dir_is_preferred_over_the_cache(manager, tmp_path):
    (manager.cache_dir / "docker_gemma3_4b_q4km.gguf").write_bytes(b"GGUF")
    preferred = tmp_path / "models" / "docker_gemma3_4b_q4km.gguf"
    preferred.write_bytes(b"GGUF")
    assert manager._find_local_model() == preferred


def test_discovery_is_deterministic_when_several_match(manager, tmp_path):
    for name in ("docker_c.gguf", "docker_a.gguf", "docker_b.gguf"):
        (tmp_path / "models" / name).write_bytes(b"GGUF")
    first = manager._find_local_model()
    assert first == tmp_path / "models" / "docker_a.gguf"
    assert manager._find_local_model() == first


def test_a_missing_search_directory_does_not_raise(tmp_path, monkeypatch):
    """`models/` need not exist. glob() on a missing directory must not blow up."""
    m = ModelManager(cli_tool="docker")
    monkeypatch.setattr(
        m, "_search_dirs", lambda: [tmp_path / "absent", tmp_path / "also-absent"]
    )
    assert m._find_local_model() is None


def test_search_dirs_includes_cwd_models_and_the_cache():
    """The real search list, pinned so the fixture above cannot drift from it."""
    m = ModelManager(cli_tool="docker")
    dirs = [str(d) for d in m._search_dirs()]
    assert dirs[0] == "models"
    assert str(m.cache_dir) in dirs
