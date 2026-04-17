"""CLI entry points for axolotl."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import fire

from axolotl import __version__, configure_logging

LOG = logging.getLogger("axolotl.cli")


def train(
    config: str,
    *,
    accelerate: bool = True,
    debug: bool = False,
    gradient_checkpointing: Optional[bool] = None,
    learning_rate: Optional[float] = None,
    micro_batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Train a model using the provided configuration file.

    Args:
        config: Path to the YAML configuration file.
        accelerate: Whether to use HuggingFace Accelerate for distributed training.
        debug: Enable debug logging.
        gradient_checkpointing: Override gradient checkpointing setting.
        learning_rate: Override learning rate from config.
        micro_batch_size: Override micro batch size from config.
        num_epochs: Override number of training epochs.
        output_dir: Override output directory for checkpoints.
        seed: Override random seed.
    """
    configure_logging(log_level="DEBUG" if debug else "INFO")
    LOG.info("axolotl %s", __version__)

    config_path = Path(config)
    if not config_path.exists():
        LOG.error("Config file not found: %s", config)
        sys.exit(1)

    # Build CLI overrides dict, only including explicitly set values
    cli_args = {}
    if gradient_checkpointing is not None:
        cli_args["gradient_checkpointing"] = gradient_checkpointing
    if learning_rate is not None:
        cli_args["learning_rate"] = learning_rate
    if micro_batch_size is not None:
        cli_args["micro_batch_size"] = micro_batch_size
    if num_epochs is not None:
        cli_args["num_epochs"] = num_epochs
    if output_dir is not None:
        cli_args["output_dir"] = output_dir
    if seed is not None:
        cli_args["seed"] = seed

    from axolotl.train import do_train  # pylint: disable=import-outside-toplevel

    do_train(config_path, cli_args=cli_args, accelerate=accelerate)


def evaluate(
    config: str,
    *,
    debug: bool = False,
):
    """Evaluate a model using the provided configuration file.

    Args:
        config: Path to the YAML configuration file.
        debug: Enable debug logging.
    """
    configure_logging(log_level="DEBUG" if debug else "INFO")
    LOG.info("axolotl %s", __version__)

    config_path = Path(config)
    if not config_path.exists():
        LOG.error("Config file not found: %s", config)
        sys.exit(1)

    from axolotl.evaluate import do_evaluate  # pylint: disable=import-outside-toplevel

    do_evaluate(config_path)


def inference(
    config: str,
    *,
    debug: bool = False,
    gradio: bool = False,
):
    """Run inference using the provided configuration file.

    Args:
        config: Path to the YAML configuration file.
        debug: Enable debug logging.
        gradio: Launch a Gradio web UI for interactive inference.
    """
    configure_logging(log_level="DEBUG" if debug else "INFO")
    LOG.info("axolotl %s", __version__)

    config_path = Path(config)
    if not config_path.exists():
        LOG.error("Config file not found: %s", config)
        sys.exit(1)

    from axolotl.inference import do_inference  # pylint: disable=import-outside-toplevel

    do_inference(config_path, gradio=gradio)


def main():
    """Main CLI entry point dispatching subcommands via python-fire."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fire.Fire(
        {
            "train": train,
            "evaluate": evaluate,
            "inference": inference,
        }
    )


if __name__ == "__main__":
    main()
