"""ReplayBackend must re-score a run recorded by a different prompt builder.

Recording every generation is only useful if a published result can actually be
re-scored later. It could not: `run_eval` builds prompts from `CHAT_TEMPLATES`,
while the Colab notebook builds them with `tokenizer.apply_chat_template`, which
prepends `<bos>`. The strings differ, so replaying a notebook run raised
KeyError on the first example. The instruction is the stable key.
"""

from __future__ import annotations

import json

import pytest

from eval.backends import ReplayBackend, build_prompt

INSTRUCTION = "Translate to docker command: show all running containers"
RAW = "COMMAND: docker ps\nEXPLANATION: lists running containers"


def write(tmp_path, rows):
    p = tmp_path / "gen.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_replays_on_an_exact_prompt_match(tmp_path):
    prompt = build_prompt(INSTRUCTION, "gemma3")
    b = ReplayBackend(write(tmp_path, [
        {"prompt": prompt, "instruction": INSTRUCTION, "raw_output": RAW}]))
    assert b.generate(prompt).text == RAW


def test_replays_a_notebook_recording_whose_prompt_has_a_bos_token(tmp_path):
    """The regression. apply_chat_template prepends <bos>; run_eval does not."""
    notebook_prompt = "<bos>" + build_prompt(INSTRUCTION, "gemma3")
    b = ReplayBackend(write(tmp_path, [
        {"prompt": notebook_prompt, "instruction": INSTRUCTION, "raw_output": RAW}]))

    # run_eval asks with its own prompt, which is not the recorded one.
    assert b.generate(build_prompt(INSTRUCTION, "gemma3")).text == RAW


@pytest.mark.parametrize("template", ["gemma3", "gemma4", "qwen", "llama3"])
def test_instruction_fallback_works_across_templates(tmp_path, template):
    b = ReplayBackend(write(tmp_path, [
        {"prompt": "something else entirely", "instruction": INSTRUCTION,
         "raw_output": RAW}]))
    assert b.generate(build_prompt(INSTRUCTION, template)).text == RAW


def test_records_without_a_prompt_field_are_still_replayable(tmp_path):
    b = ReplayBackend(write(tmp_path, [
        {"instruction": INSTRUCTION, "raw_output": RAW}]))
    assert b.generate(build_prompt(INSTRUCTION, "gemma3")).text == RAW


def test_latency_and_logprob_survive_the_replay(tmp_path):
    b = ReplayBackend(write(tmp_path, [
        {"instruction": INSTRUCTION, "raw_output": RAW,
         "latency_s": 1.25, "mean_logprob": -0.4}]))
    g = b.generate(build_prompt(INSTRUCTION, "gemma3"))
    assert g.latency_s == 1.25
    assert g.mean_logprob == -0.4


def test_a_null_latency_does_not_become_none(tmp_path):
    """The notebook records latency_s as null; scoring sums it."""
    b = ReplayBackend(write(tmp_path, [
        {"instruction": INSTRUCTION, "raw_output": RAW, "latency_s": None}]))
    assert b.generate(build_prompt(INSTRUCTION, "gemma3")).latency_s == 0.0


def test_an_unknown_instruction_still_raises(tmp_path):
    b = ReplayBackend(write(tmp_path, [
        {"instruction": INSTRUCTION, "raw_output": RAW}]))
    with pytest.raises(KeyError, match="no recorded generation"):
        b.generate(build_prompt("Translate to docker command: something unseen",
                                "gemma3"))


def test_an_empty_recording_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="no replayable records"):
        ReplayBackend(write(tmp_path, []))
