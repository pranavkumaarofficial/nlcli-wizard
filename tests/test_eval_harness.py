"""Tests for the evaluation harness.

The harness decides what every published number means, so it gets tested before it
gets trusted. The regression these guard against is the original failure: a metric
that quietly reports something other than what it claims.
"""

import json

import pytest

from eval.contamination import Example, audit, prompt_fingerprint, self_audit, template_fingerprint
from eval.metrics import categorize, score_all, score_one
from eval.normalize import normalized_string, parse_command
from eval.run_eval import extract_command
from eval.splits import partition_test_by_novelty, split_by_command


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

def test_flag_order_does_not_change_parse():
    a = parse_command("docker run -d -p 8080:80 nginx")
    b = parse_command("docker run -p 8080:80 -d nginx")
    assert a.flags == b.flags
    assert a.path == b.path
    assert a.positionals == b.positionals


def test_value_flags_consume_their_argument():
    p = parse_command("docker run -p 8080:80 --name web nginx")
    assert ("--publish", "8080:80") in p.flags
    assert ("--name", "web") in p.flags
    # nginx is the image, not a stray value
    assert p.positionals == ("nginx",)


def test_bundled_short_flags_expand():
    p = parse_command("docker exec -it web bash")
    assert ("--interactive", None) in p.flags
    assert ("--tty", None) in p.flags
    assert p.positionals == ("web", "bash")


def test_t_is_tag_for_build_but_tty_for_run():
    build = parse_command("docker build -t myapp:1.0 .")
    assert ("--tag", "myapp:1.0") in build.flags

    run = parse_command("docker run -t nginx")
    assert ("--tty", None) in run.flags


def test_equals_form_and_space_form_agree():
    a = parse_command("docker build --build-arg VERSION=2.1 .")
    b = parse_command("docker build --build-arg=VERSION=2.1 .")
    assert a.flags == b.flags


def test_positional_order_is_significant():
    a = parse_command("docker exec web bash")
    b = parse_command("docker exec bash web")
    assert a.positionals != b.positionals


def test_nested_subcommand_path():
    assert parse_command("docker network create devnet").path == ("docker", "network", "create")
    assert parse_command("docker volume prune").path == ("docker", "volume", "prune")


def test_unparseable_command_does_not_raise():
    p = parse_command('docker run --name "unclosed nginx')
    assert p.parse_ok is False


def test_empty_command():
    p = parse_command("")
    assert p.parse_ok is False
    assert p.path == ()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_exact_match_is_exact():
    j = score_one("docker ps", "docker ps")
    assert j.exact and j.normalized and j.functional


def test_reordered_flags_are_normalized_not_exact():
    j = score_one("docker run -p 8080:80 -d nginx", "docker run -d -p 8080:80 nginx")
    assert not j.exact
    assert j.normalized
    assert j.functional


def test_wrong_flag_is_a_miss_on_every_metric():
    j = score_one("docker ps -a", "docker ps")
    assert not j.exact and not j.normalized and not j.functional


def test_detach_is_not_ignorable():
    """-d changes behaviour; it must never be treated as noise."""
    j = score_one("docker run nginx", "docker run -d nginx")
    assert not j.functional


def test_docker_ps_equals_container_ls():
    j = score_one("docker container ls", "docker ps")
    assert not j.exact
    assert j.functional


def test_compose_plugin_and_binary_are_equivalent():
    j = score_one("docker compose up -d", "docker-compose up -d")
    assert j.functional


def test_empty_prediction_is_a_miss():
    j = score_one("", "docker ps")
    assert not j.functional


def test_exact_match_forces_looser_metrics_true():
    """Guards against a normalization bug downgrading a byte-identical answer."""
    weird = "docker run --some-unknown-flag=x img"
    j = score_one(weird, weird)
    assert j.exact and j.normalized and j.functional


def test_score_all_counts_and_categories():
    preds = ["docker ps", "docker ps -a", "docker run -d nginx"]
    golds = ["docker ps", "docker ps", "docker run -d nginx"]
    r = score_all(preds, golds, ["ps_images", "ps_images", "run"])
    assert r.n == 3
    assert r.exact == 2
    assert r.per_category["ps_images"]["n"] == 2
    assert r.per_category["run"]["functional"] == 1


def test_score_all_rejects_length_mismatch():
    with pytest.raises(ValueError):
        score_all(["a"], ["a", "b"])


def test_categorize():
    assert categorize("docker-compose up -d") == "compose"
    assert categorize("docker run nginx") == "run"
    assert categorize("docker build .") == "build"
    assert categorize("docker exec -it web bash") == "exec"
    assert categorize("docker network ls") == "network"
    assert categorize("docker volume prune") == "volume"
    assert categorize("docker ps") == "ps_images"
    assert categorize("docker system prune") == "system"


# --------------------------------------------------------------------------
# contamination
# --------------------------------------------------------------------------

def _ex(prompt, command):
    return Example(f"Translate to docker command: {prompt}", command)


def test_verbatim_prompt_leak_is_detected():
    train = [_ex("list running containers", "docker ps")]
    test = [_ex("list running containers", "docker ps")]
    r = audit(train, test)
    assert len(r.prompt_verbatim) == 1
    assert not r.is_clean


def test_near_duplicate_prompt_is_detected():
    train = [_ex("list running containers", "docker ps")]
    test = [_ex("Running containers, list!", "docker ps")]
    r = audit(train, test)
    assert len(r.prompt_near_dup) == 1
    assert not r.is_clean


def test_shared_template_is_detected():
    """The leak that survives a naive random split."""
    train = [_ex("run nginx on port 8080", "docker run -p 8080:80 nginx")]
    test = [_ex("run redis on port 6379", "docker run -p 6379:6379 redis")]
    r = audit(train, test)
    assert len(r.template_shared) == 1


def test_target_overlap_is_disclosed_but_not_fatal():
    train = [_ex("list running containers", "docker ps")]
    test = [_ex("what is currently up", "docker ps")]
    r = audit(train, test)
    assert len(r.target_overlap) == 1
    assert r.is_clean          # prompt-clean
    assert not r.is_strict_clean


def test_clean_pair_is_clean():
    train = [_ex("list running containers", "docker ps")]
    test = [_ex("delete the uploads volume", "docker volume rm uploads")]
    r = audit(train, test)
    assert r.is_strict_clean


def test_fingerprints():
    assert prompt_fingerprint("List  running Containers!") == prompt_fingerprint(
        "containers running list"
    )
    assert template_fingerprint("run nginx on port 8080") == template_fingerprint(
        "run redis on port 6379"
    )


def test_self_audit_flags_unsafe_random_split():
    dup = [_ex("what venv am i in", "venvy current")] * 5
    stats = self_audit(dup)
    assert stats["random_split_is_safe"] is False
    assert stats["prompt_duplication_factor"] == 5.0


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def test_command_split_never_leaks_a_command():
    examples = []
    for i in range(40):
        cmd = f"docker run --name c{i} nginx"
        examples.append(_ex(f"start container number {i}", cmd))
        examples.append(_ex(f"launch c{i} please", cmd))  # paraphrase, same target

    split = split_by_command(examples, test_fraction=0.25, seed=0)
    train_cmds = {normalized_string(e.command) for e in split.train}
    test_cmds = {normalized_string(e.command) for e in split.test}

    assert test_cmds
    assert train_cmds & test_cmds == set(), "command leaked across the split"


def test_novelty_partition():
    train = [_ex("list containers", "docker ps")]
    test = [
        _ex("what is running", "docker ps"),              # seen command
        _ex("remove the uploads volume", "docker volume rm uploads"),  # unseen
    ]
    unseen_cmd, unseen_phrasing = partition_test_by_novelty(test, train)
    assert len(unseen_cmd) == 1
    assert len(unseen_phrasing) == 1


# --------------------------------------------------------------------------
# output parsing
# --------------------------------------------------------------------------

def test_extract_command_from_structured_output():
    raw = "COMMAND: docker ps -a\nCONFIDENCE: 0.94\nEXPLANATION: lists all"
    assert extract_command(raw) == "docker ps -a"


def test_extract_command_falls_back_to_first_line():
    assert extract_command("docker ps -a\nsome noise") == "docker ps -a"


def test_extract_command_strips_markdown_fence():
    raw = "```bash\ndocker ps -a\n```"
    assert extract_command(raw) == "docker ps -a"


def test_extract_command_on_empty():
    assert extract_command("") == ""
    assert extract_command(None) == ""


# --------------------------------------------------------------------------
# the real test set
# --------------------------------------------------------------------------

def test_handwritten_test_set_is_clean_against_training(tmp_path):
    """The headline guarantee, asserted in CI.

    If someone regenerates the training data and reintroduces a leak, this fails.
    """
    from pathlib import Path

    train_path = Path("data/docker_training.jsonl")
    test_path = Path("data/docker_test_handwritten.jsonl")
    if not train_path.exists() or not test_path.exists():
        pytest.skip("datasets not present")

    from eval.contamination import load_jsonl

    r = audit(load_jsonl(train_path), load_jsonl(test_path))
    assert r.is_clean, (
        f"contamination detected: {len(r.prompt_verbatim)} verbatim, "
        f"{len(r.prompt_near_dup)} near-duplicate prompts"
    )


def test_handwritten_test_set_commands_all_parse():
    from pathlib import Path

    test_path = Path("data/docker_test_handwritten.jsonl")
    if not test_path.exists():
        pytest.skip("dataset not present")

    with open(test_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            p = parse_command(row["command"])
            assert p.parse_ok, f"line {lineno}: {row['command']!r} failed to parse"
            assert p.path[0] in {"docker", "docker-compose"}, f"line {lineno}"
