"""The runtime install must not need torch, transformers or huggingface-hub.

The inference path is GGUF-only: llama.cpp reads the quantized file directly. Those
three were declared as hard dependencies in pyproject anyway, so a install of a
CLI that never imports them on that path pulled ~2 GB of wheels. They now live in
the `training` extra, and these tests are what stops them drifting back.

Two angles, because either alone is easy to fool:
  * a static scan, which catches a module-level import even on a machine where the
    package happens to be installed;
  * an import with those names blocked, which catches an indirect import through
    some other module.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

import nlcli_wizard

HEAVY = {"torch", "transformers", "huggingface_hub"}

PACKAGE_DIR = Path(nlcli_wizard.__file__).parent
MODULES = sorted(
    f"nlcli_wizard.{m.name}" for m in pkgutil.iter_modules([str(PACKAGE_DIR)])
)


def test_there_are_modules_to_check():
    """Guard against the scan silently passing because it found nothing."""
    assert len(MODULES) >= 5, MODULES


@pytest.mark.parametrize("source_file", sorted(PACKAGE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_module_level_import_of_a_heavy_dependency(source_file):
    """A heavy import is allowed only inside a function or a guarded try block."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    offenders = []
    for node in tree.body:  # module level only; nested imports are deliberate
        if isinstance(node, ast.Import):
            offenders += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            offenders.append(node.module.split(".")[0])

    assert not (set(offenders) & HEAVY), (
        f"{source_file.name} imports {sorted(set(offenders) & HEAVY)} at module level. "
        "Import it lazily inside the function that needs it, or guard it with "
        "try/except ImportError and report what to install."
    )


def test_package_imports_with_heavy_dependencies_blocked(monkeypatch):
    """Every module must import when torch/transformers/huggingface_hub are absent."""
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name.split(".")[0] in HEAVY:
            raise ImportError(f"blocked by test: {name}")
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name.split(".")[0] in ("nlcli_wizard",) or name.split(".")[0] in HEAVY:
            monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    for module in MODULES:
        importlib.import_module(module)
