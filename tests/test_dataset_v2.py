"""Tests for the v2 dataset generator.

v2 exists to fix measured defects in v1, so each fix gets a test. Several of these
guard bugs that were actually present in the first v2 build and would otherwise
creep back the next time the phrasing tables are edited.
"""

import json
import re
from pathlib import Path

import pytest

from eval.contamination import audit, load_jsonl
from eval.normalize import parse_command
from nlcli_wizard.dataset_v2 import (
    DockerDatasetV2,
    _dedup_key,
    audit as v2_audit,
    load_exclusions,
)

TRAIN_V2 = Path("data/docker_train_v2.jsonl")
TEST_SET = Path("data/docker_test_handwritten.jsonl")


@pytest.fixture(scope="module")
def sample():
    return DockerDatasetV2(seed=1234).generate(1500)


# --------------------------------------------------------------------------
# The point of v2: composition
# --------------------------------------------------------------------------

def test_multi_flag_examples_are_a_large_share(sample):
    """v1 had 12.6% multi-flag and scored 5% on 2-flag commands."""
    stats = v2_audit(sample)
    share = float(stats["multi_flag_share"].rstrip("%"))
    assert share > 40.0, f"multi-flag share too low: {share}%"


def test_flag_pair_coverage_far_exceeds_v1(sample):
    """v1 had 17 distinct flag pairs, 47 occurrences of which were `-i`+`-t`."""
    stats = v2_audit(sample)
    assert stats["distinct_flag_pairs"] >= 40


def test_three_plus_flag_examples_exist(sample):
    """v1 contained 4 examples with 3+ flags, and scored 0/12 on them."""
    n3 = sum(1 for e in sample if len(parse_command(e.command).flags) >= 3)
    assert n3 > len(sample) * 0.10


# --------------------------------------------------------------------------
# Correctness of generated commands
# --------------------------------------------------------------------------

def test_all_generated_commands_parse(sample):
    for e in sample:
        p = parse_command(e.command)
        assert p.parse_ok, e.command
        assert p.path[0] in {"docker", "docker-compose"}, e.command


def test_no_duplicate_prompts(sample):
    keys = [_dedup_key(e.prompt) for e in sample]
    assert len(keys) == len(set(keys))


def test_interactive_and_detach_are_never_combined(sample):
    """-it and -d are contradictory; a dataset teaching both at once is wrong."""
    for e in sample:
        flags = {f for f, _ in parse_command(e.command).flags}
        assert not ({"--interactive", "--tty"} <= flags and "--detach" in flags), e.command


def test_interactive_run_always_gets_a_shell(sample):
    for e in sample:
        p = parse_command(e.command)
        if p.path == ("docker", "run") and "--tty" in {f for f, _ in p.flags}:
            assert p.positionals[-1] in {"bash", "sh"}, e.command


# --------------------------------------------------------------------------
# Defects found in the first v2 build
# --------------------------------------------------------------------------

def test_no_doubled_conjunction(sample):
    """Phrasings that already begin with 'and' were getting a second one."""
    for e in sample:
        assert " and and " not in e.prompt, e.prompt


def test_no_port_published_on_non_serving_image(sample):
    """`docker run -p 3000:8080 busybox` was being generated."""
    for e in sample:
        assert not re.search(r"-p \d+:\d+ .*(busybox|ubuntu|alpine)\b", e.command), e.command


def test_env_var_matches_image(sample):
    """`docker run -e MONGO_INITDB_ROOT_PASSWORD=secret caddy` was being generated."""
    pairs = [
        ("POSTGRES_PASSWORD", "postgres"),
        ("MYSQL_ROOT_PASSWORD", "mysql"),
        ("MONGO_INITDB_ROOT_PASSWORD", "mongo"),
        ("REDIS_PASSWORD", "redis"),
    ]
    for e in sample:
        for var, img in pairs:
            if var in e.command:
                assert re.search(rf"\b{img}\b", e.command), e.command


def test_build_arg_values_match_their_names(sample):
    """`--build-arg VERSION=production` / `NODE_ENV=1.0` were being generated."""
    for e in sample:
        assert not re.search(r"VERSION=(production|dev|staging|prod)\b", e.command), e.command
        assert not re.search(r"NODE_ENV=[0-9]", e.command), e.command


# --------------------------------------------------------------------------
# Contamination — the guarantee that matters most
# --------------------------------------------------------------------------

def test_generator_respects_exclusions():
    """The held-out set must be blocked at generation time, not just audited after.

    The first v2 build leaked 12 test prompts verbatim because the same author
    wrote both the test set and the generator phrasings.
    """
    if not TEST_SET.exists():
        pytest.skip("test set not present")
    excl = load_exclusions([TEST_SET])
    assert excl, "exclusion set should not be empty"

    gen = DockerDatasetV2(seed=99).generate(1500, exclude=excl)
    keys = {_dedup_key(e.prompt) for e in gen}
    assert keys & excl == set(), "generator emitted a held-out prompt"


@pytest.mark.skipif(not TRAIN_V2.exists(), reason="v2 dataset not generated")
def test_shipped_v2_dataset_is_clean_against_the_test_set():
    """CI gate: the committed v2 training file must not leak the held-out set."""
    report = audit(load_jsonl(TRAIN_V2), load_jsonl(TEST_SET))
    assert report.is_clean, (
        f"{len(report.prompt_verbatim)} verbatim, "
        f"{len(report.prompt_near_dup)} near-duplicate prompt leaks"
    )


@pytest.mark.skipif(not TRAIN_V2.exists(), reason="v2 dataset not generated")
def test_shipped_v2_dataset_rows_are_well_formed():
    with open(TRAIN_V2, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = json.loads(line)
            assert row["instruction"].startswith("Translate to docker command:"), lineno
            assert row["output"].startswith("COMMAND: "), lineno
            # v2 drops the fabricated CONFIDENCE field entirely rather than
            # training the model to emit random.uniform(0.90, 0.97).
            assert "CONFIDENCE" not in row["output"], lineno
