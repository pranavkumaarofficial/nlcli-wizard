"""Real pytest tests land here.

`pyproject.toml` previously pointed `testpaths` at a `tests/` directory that did not
exist, while the actual `test/` directory held three print-based demo scripts. This
directory is the fix; the demo scripts moved to `scripts/legacy/`.

First real tests arrive with Milestone 1 (see notes/PROGRESS.md):
  - eval/splits.py       — a command-level split must leak zero commands
  - eval/contamination.py — must flag a known-contaminated pair
  - eval/metrics.py      — flag-order-equivalent commands must score as matches
"""


def test_package_imports():
    """Smoke test: the package imports without optional heavy deps installed."""
    import nlcli_wizard

    assert nlcli_wizard.__version__
