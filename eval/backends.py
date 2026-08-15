"""Inference backends.

The harness must not care how a model is executed. Three backends implement one
interface so the same evaluation runs against a GGUF on a laptop, a GGUF via a
prebuilt llama.cpp binary (no Python build toolchain required), or a HuggingFace
model on a Colab GPU.

This also keeps baselines honest: a base model and a fine-tune go through exactly
the same code path.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class Generation:
    """One model response."""

    text: str
    latency_s: float = 0.0
    mean_logprob: Optional[float] = None
    """Mean token logprob, when the backend can supply it.

    This is the honest replacement for the `random.uniform(0.90, 0.97)` confidence
    the model was trained to emit. See Milestone 3 in notes/PROGRESS.md.
    """


class Backend(ABC):
    """Common interface for anything that can answer a prompt."""

    name: str = "backend"

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 128) -> Generation:
        ...

    def describe(self) -> Dict[str, str]:
        return {"backend": self.name}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

CHAT_TEMPLATES = {
    # Gemma 1/2/3
    "gemma3": (
        "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        ["<end_of_turn>", "<eos>"],
    ),
    # Gemma 4 (E2B/E4B) — different turn markers, verified against the model's
    # chat_template.jinja on HuggingFace.
    "gemma4": (
        "<|turn>user\n{instruction}<turn|>\n<|turn>model\n",
        ["<turn|>", "<|turn>"],
    ),
    # Qwen2/Qwen3
    "qwen": (
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        ["<|im_end|>"],
    ),
    # Llama 3.x
    "llama3": (
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ["<|eot_id|>"],
    ),
    # No template — raw completion, for base (non-instruct) models.
    "raw": ("{instruction}\n", ["\n\n"]),
}


def build_prompt(instruction: str, template: str) -> str:
    if template not in CHAT_TEMPLATES:
        raise ValueError(
            f"unknown template {template!r}; known: {sorted(CHAT_TEMPLATES)}"
        )
    fmt, _ = CHAT_TEMPLATES[template]
    return fmt.format(instruction=instruction)


def stop_tokens(template: str) -> List[str]:
    return list(CHAT_TEMPLATES[template][1])


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class LlamaCppPythonBackend(Backend):
    """llama-cpp-python bindings. Preferred when the wheel installs."""

    name = "llama-cpp-python"

    def __init__(
        self,
        model_path: Path,
        template: str,
        n_ctx: int = 512,
        n_threads: int = 4,
        temperature: float = 0.0,
    ):
        from llama_cpp import Llama  # imported lazily; optional dependency

        self.model_path = Path(model_path)
        self.template = template
        self.temperature = temperature
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            logits_all=False,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 128) -> Generation:
        import time

        t0 = time.time()
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=1.0 if self.temperature == 0 else 0.9,
            stop=stop_tokens(self.template),
            logprobs=1,
            echo=False,
        )
        dt = time.time() - t0
        choice = out["choices"][0]

        mean_lp = None
        lp = choice.get("logprobs") or {}
        token_lps = [x for x in (lp.get("token_logprobs") or []) if x is not None]
        if token_lps:
            mean_lp = sum(token_lps) / len(token_lps)

        return Generation(text=choice["text"].strip(), latency_s=dt, mean_logprob=mean_lp)

    def describe(self) -> Dict[str, str]:
        return {
            "backend": self.name,
            "model": self.model_path.name,
            "template": self.template,
            "temperature": str(self.temperature),
        }


class LlamaCppBinaryBackend(Backend):
    """Prebuilt `llama-cli` binary, driven by subprocess.

    Exists because llama-cpp-python has no Windows wheel for every Python version
    and building it needs MSVC. The official llama.cpp release binaries need no
    toolchain, so the eval can run anywhere.
    """

    name = "llama-cli"

    def __init__(
        self,
        model_path: Path,
        template: str,
        binary: Optional[Path] = None,
        n_ctx: int = 512,
        n_threads: int = 4,
        temperature: float = 0.0,
    ):
        self.model_path = Path(model_path)
        self.template = template
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature

        self.binary = Path(binary) if binary else self._discover_binary()
        if self.binary is None or not Path(self.binary).exists():
            raise FileNotFoundError(
                "llama-cli binary not found. Download a release from "
                "https://github.com/ggml-org/llama.cpp/releases and pass --llama-bin, "
                "or set LLAMA_CLI on PATH."
            )

    @staticmethod
    def _discover_binary() -> Optional[Path]:
        for candidate in ("llama-cli", "llama-cli.exe", "main", "main.exe"):
            found = shutil.which(candidate)
            if found:
                return Path(found)
        for candidate in (
            Path("llama.cpp/build/bin/llama-cli"),
            Path("llama.cpp/build/bin/Release/llama-cli.exe"),
            Path("vendor/llama.cpp/llama-cli.exe"),
            Path("vendor/llama.cpp/llama-cli"),
        ):
            if candidate.exists():
                return candidate
        return None

    def generate(self, prompt: str, max_tokens: int = 128) -> Generation:
        import time

        cmd = [
            str(self.binary),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "--temp", str(self.temperature),
            "-no-cnv",          # completion mode, not the interactive chat REPL
            "--no-warmup",
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0

        text = proc.stdout
        # llama-cli echoes the prompt before the completion; strip it.
        if prompt in text:
            text = text.split(prompt, 1)[1]
        for stop in stop_tokens(self.template):
            if stop in text:
                text = text.split(stop, 1)[0]
        # Trailing performance summary llama-cli writes to stdout on some builds.
        text = re.split(r"\n\s*llama_(perf|print)", text)[0]

        return Generation(text=text.strip(), latency_s=dt)

    def describe(self) -> Dict[str, str]:
        return {
            "backend": self.name,
            "binary": str(self.binary),
            "model": self.model_path.name,
            "template": self.template,
            "temperature": str(self.temperature),
        }


class LlamaServerBackend(Backend):
    """Talks to a `llama-server` HTTP endpoint.

    Preferred GGUF backend. Three reasons over per-prompt subprocess invocation:

      * the model loads once, not once per example (116 examples x ~3s of load time
        is six minutes of pure waste);
      * the response is JSON, so there is no scraping of a TUI that changes between
        llama.cpp releases — `llama-cli` in b10437 ignores `-no-cnv` and drops into
        an interactive chat UI, which silently breaks subprocess scraping;
      * it returns per-token logprobs, which is the honest confidence signal that
        replaces the `random.uniform(0.90, 0.97)` the model was trained to emit.

    Starts the server itself unless pointed at an already-running one.
    """

    name = "llama-server"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        template: str = "gemma3",
        binary: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_ctx: int = 512,
        n_threads: int = 4,
        temperature: float = 0.0,
        server_url: Optional[str] = None,
        startup_timeout: float = 180.0,
    ):
        self.template = template
        self.temperature = temperature
        self.model_path = Path(model_path) if model_path else None
        self._proc: Optional[subprocess.Popen] = None

        if server_url:
            self.url = server_url.rstrip("/")
            return

        self.url = f"http://{host}:{port}"
        binary = Path(binary) if binary else self._discover_server()
        if binary is None:
            raise FileNotFoundError(
                "llama-server binary not found. Download a release from "
                "https://github.com/ggml-org/llama.cpp/releases and pass --llama-bin."
            )

        cmd = [
            str(binary),
            "-m", str(self.model_path),
            "--host", host,
            "--port", str(port),
            "-c", str(n_ctx),
            "-t", str(n_threads),
            "--no-webui",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._wait_until_ready(startup_timeout)

    @staticmethod
    def _discover_server() -> Optional[Path]:
        for candidate in ("llama-server", "llama-server.exe"):
            found = shutil.which(candidate)
            if found:
                return Path(found)
        for candidate in (
            Path("vendor/llama.cpp/llama-server.exe"),
            Path("vendor/llama.cpp/llama-server"),
            Path("llama.cpp/build/bin/llama-server"),
        ):
            if candidate.exists():
                return candidate
        return None

    def _wait_until_ready(self, timeout: float) -> None:
        import time
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._proc.returncode})"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as e:  # not up yet
                last_err = e
            time.sleep(1.0)
        raise TimeoutError(f"llama-server did not become ready in {timeout}s: {last_err}")

    def generate(self, prompt: str, max_tokens: int = 128) -> Generation:
        import json as _json
        import time
        import urllib.request

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0 if self.temperature == 0 else 0.9,
            "stop": stop_tokens(self.template),
            "n_probs": 1,
            "cache_prompt": False,
        }
        req = urllib.request.Request(
            f"{self.url}/completion",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            body = _json.loads(r.read().decode("utf-8"))
        dt = time.time() - t0

        mean_lp = None
        probs = body.get("completion_probabilities") or []
        lps = []
        for entry in probs:
            if "logprob" in entry:
                lps.append(entry["logprob"])
            elif entry.get("probs"):
                top = entry["probs"][0]
                if "prob" in top and top["prob"] > 0:
                    import math

                    lps.append(math.log(top["prob"]))
        if lps:
            mean_lp = sum(lps) / len(lps)

        return Generation(
            text=(body.get("content") or "").strip(), latency_s=dt, mean_logprob=mean_lp
        )

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def describe(self) -> Dict[str, str]:
        return {
            "backend": self.name,
            "model": self.model_path.name if self.model_path else "remote",
            "template": self.template,
            "temperature": str(self.temperature),
            "url": self.url,
        }


class TransformersBackend(Backend):
    """HuggingFace transformers — for GPU runs (Colab) and for base-model baselines."""

    name = "transformers"

    def __init__(
        self,
        model_id: str,
        template: Optional[str] = None,
        device: str = "cuda",
        temperature: float = 0.0,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.temperature = temperature
        self.template = template  # None => use the tokenizer's own chat template

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"torch_dtype": torch.float16, "device_map": device}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def build_prompt_native(self, instruction: str) -> str:
        """Prefer the tokenizer's own chat template over our hardcoded table.

        This is the fix for the bug where inference hardcoded Gemma 3 markers while
        training used Gemma 4's.
        """
        if self.template is not None:
            return build_prompt(instruction, self.template)
        messages = [{"role": "user", "content": instruction}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, max_tokens: int = 128) -> Generation:
        import time

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        dt = time.time() - t0

        seq = out.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(seq, skip_special_tokens=True)

        mean_lp = None
        if out.scores:
            lps = []
            for step, tok in zip(out.scores, seq):
                lp = torch.log_softmax(step[0].float(), dim=-1)[tok].item()
                lps.append(lp)
            if lps:
                mean_lp = sum(lps) / len(lps)

        return Generation(text=text.strip(), latency_s=dt, mean_logprob=mean_lp)

    def describe(self) -> Dict[str, str]:
        return {
            "backend": self.name,
            "model": self.model_id,
            "template": self.template or "tokenizer-native",
            "temperature": str(self.temperature),
        }


class ReplayBackend(Backend):
    """Replays recorded generations from a JSONL file.

    Lets the scoring path be developed and unit-tested without a GPU, and lets a
    published result be re-scored later under improved metrics without re-running
    inference.
    """

    name = "replay"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._by_prompt: Dict[str, Generation] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self._by_prompt[row["prompt"]] = Generation(
                    text=row.get("raw_output", ""),
                    latency_s=row.get("latency_s", 0.0),
                    mean_logprob=row.get("mean_logprob"),
                )

    def generate(self, prompt: str, max_tokens: int = 128) -> Generation:
        if prompt not in self._by_prompt:
            raise KeyError(f"no recorded generation for prompt: {prompt[:80]!r}")
        return self._by_prompt[prompt]

    def describe(self) -> Dict[str, str]:
        return {"backend": self.name, "source": str(self.path)}
