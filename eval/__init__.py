"""Evaluation harness for nlcli-wizard.

See eval/README.md for the rules this package must obey. The short version:
splits are by target command, every accuracy number ships with a contamination
report, and metrics are reported as a set rather than a single number.
"""

__all__ = ["normalize", "metrics", "contamination", "splits"]
