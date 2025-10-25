"""
Bharat CLI Module
================

Command-line interface toolkit for job scheduling, config management
with Typer and ExoCLI support.
"""

from .main import main, app
from .config import CLIConfig
from .commands import TrainCommand, EvalCommand, DeployCommand

__version__ = "0.1.0"
__all__ = [
    "main",
    "app",
    "CLIConfig",
    "TrainCommand",
    "EvalCommand",
    "DeployCommand"
]