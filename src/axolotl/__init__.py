"""
Axolotl - Fine-tuning library for large language models.

This package provides tools and utilities for fine-tuning LLMs using
various techniques including LoRA, QLoRA, full fine-tuning, and more.
"""

import importlib.metadata
import logging

logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("axolotl")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
