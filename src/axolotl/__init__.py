"""Axolotl - Fine-tuning library for large language models.

This package provides tools and utilities for fine-tuning LLMs using
various techniques including LoRA, QLoRA, full fine-tuning, and more.

Personal fork: using this for experimenting with custom datasets and
LoRA configs on consumer hardware.

Note: logging level set to WARNING by default to reduce noise during
long training runs.
"""

import importlib.metadata
import logging

# Reduce default log verbosity - too noisy during training
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("axolotl")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
