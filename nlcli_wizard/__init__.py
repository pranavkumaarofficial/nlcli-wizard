"""
nlcli-wizard - Natural Language Interface for Python CLI Tools

A framework for adding natural language understanding to any Python CLI tool
using locally-running Small Language Models (SLMs).
"""

__version__ = "0.1.0"
__author__ = "Pranav Kumaar"

from nlcli_wizard.agent import NLCLIAgent
from nlcli_wizard.model import ModelManager

__all__ = ["NLCLIAgent", "ModelManager"]
