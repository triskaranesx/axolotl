"""Axolotl - Fine-tuning library for large language models.

This package provides tools and utilities for fine-tuning LLMs using
various techniques including LoRA, QLoRA, full fine-tuning, and more.

Personal fork: using this for experimenting with custom datasets and
LoRA configs on consumer hardware.

Note: logging level set to INFO by default so I can actually see
what's happening during training runs.

TODO: try out the new multipack sampler with my dataset, see if it
actually speeds things up on the 3090.

TODO: investigate gradient checkpointing interaction with custom
dataset collator - seeing occasional NaN losses on long sequences.
"""

import importlib.metadata
import logging

# Changed WARNING -> INFO so training progress is visible in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("axolotl")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

logger.debug("axolotl version: %s", __version__)

__all__ = ["__version__"]
