"""Regression tests for the llama-cpp-python backend.

The backend constructed `Llama(..., logits_all=False)` and then asked for
`logprobs=1` on every call. llama-cpp-python raises

    ValueError: logprobs is not supported for models created with logits_all=False

so `--backend llama-cpp-python` raised on the first example, every time. Nothing
caught it because the published run used llama-server and no test exercised this
backend at all.

These tests run against a stub that reproduces that constraint, so they need
neither llama-cpp-python nor a GGUF.
"""

from __future__ import annotations

import sys
import types

import pytest

from eval.backends import LlamaCppPythonBackend


class FakeLlama:
    """Mimics the one llama-cpp-python behaviour this backend got wrong."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logits_all = kwargs.get("logits_all", False)
        self.calls = []

    def __call__(self, prompt, **kwargs):
        if "logprobs" in kwargs and not self.logits_all:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )
        self.calls.append(kwargs)

        choice = {"text": "COMMAND: docker ps -a\n"}
        if kwargs.get("logprobs"):
            choice["logprobs"] = {"token_logprobs": [-0.1, -0.3, None, -0.2]}
        return {"choices": [choice]}


@pytest.fixture(autouse=True)
def stub_llama_cpp(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return module


def make(**kwargs):
    return LlamaCppPythonBackend("model.gguf", template="gemma3", **kwargs)


def test_generate_does_not_raise_by_default():
    """The regression. This raised ValueError on every call."""
    backend = make()
    gen = backend.generate("<start_of_turn>user\nhi<end_of_turn>\n")
    assert gen.text == "COMMAND: docker ps -a"


def test_logprobs_is_not_requested_when_logits_all_is_off():
    """Asking for logprobs against logits_all=False is what raised."""
    backend = make()
    backend.generate("hi")
    assert "logprobs" not in backend._llm.calls[0]


def test_default_construction_leaves_logits_all_off():
    assert make()._llm.kwargs["logits_all"] is False


def test_mean_logprob_is_none_when_logprobs_were_not_requested():
    assert make().generate("hi").mean_logprob is None


def test_opting_in_turns_on_logits_all_and_returns_a_mean_logprob():
    backend = make(want_logprobs=True)
    assert backend._llm.kwargs["logits_all"] is True

    gen = backend.generate("hi")
    assert backend._llm.calls[0]["logprobs"] == 1
    # None entries are dropped before averaging: mean(-0.1, -0.3, -0.2)
    assert gen.mean_logprob == pytest.approx(-0.2)


def test_stop_tokens_come_from_the_template():
    backend = make()
    backend.generate("hi")
    assert backend._llm.calls[0]["stop"] == ["<end_of_turn>", "<eos>"]


def test_describe_reports_whether_logprobs_are_on():
    assert make().describe()["logprobs"] == "False"
    assert make(want_logprobs=True).describe()["logprobs"] == "True"


def test_greedy_decoding_uses_top_p_1():
    backend = make()
    backend.generate("hi")
    assert backend._llm.calls[0]["top_p"] == 1.0
    assert backend._llm.calls[0]["temperature"] == 0.0
