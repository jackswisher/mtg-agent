"""Logging utilities for MTG-Causal-RL.

This module provides consistent logging configuration across the project.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Rich for pretty console output (optional)
try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Default format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    use_rich: bool = True,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level.
        log_file: Optional file to log to.
        use_rich: Whether to use rich for console output.
    """
    handlers: list[logging.Handler] = []

    # Console handler
    if use_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
    handlers.append(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Logger for experiment tracking.

    Logs to both console and file with experiment-specific formatting.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "results/logs",
    ):
        """Initialize the experiment logger.

        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory for log files.
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # Set up logging
        setup_logging(log_file=str(log_file))

        self.logger = get_logger(f"experiment.{experiment_name}")
        self.logger.info(f"Starting experiment: {experiment_name}")

    def log_config(self, config: dict) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary.
        """
        self.logger.info("Experiment configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics.

        Args:
            metrics: Metrics dictionary.
            step: Optional step number.
        """
        step_str = f"Step {step} | " if step is not None else ""
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"{step_str}{metrics_str}")

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        win: bool,
    ) -> None:
        """Log episode results.

        Args:
            episode: Episode number.
            reward: Episode reward.
            length: Episode length.
            win: Whether agent won.
        """
        result = "Win" if win else "Loss"
        self.logger.info(f"Episode {episode} | Reward: {reward:.3f} | Length: {length} | {result}")

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
