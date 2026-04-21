"""
Football AI Platform — Model Training Script

Usage
-----
    # Train all models from all available match data
    python scripts/train_models.py

    # Train specific models only
    python scripts/train_models.py --models xg xt

    # Force retrain even if models already saved
    python scripts/train_models.py --force

    # Train only on specific matches
    python scripts/train_models.py --match-ids 3813041

Prerequisites
-------------
    Run scripts/run_pipeline.py first to produce the feature Parquet files.
    Without feature files this script will exit with an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from football_ai.config import settings
from football_ai.logger import get_logger, setup_logging
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.ml.training.pipeline import MLTrainingPipeline
from football_ai.utils import timer

setup_logging()
logger = get_logger(__name__)

_VALID_MODELS = ["xg", "xt", "p_scores", "p_concedes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football AI Platform -- model trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=_VALID_MODELS,
        default=_VALID_MODELS,
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--match-ids",
        nargs="+",
        default=None,
        help="Specific match IDs to include (default: all available)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if saved models already exist",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override the models directory (default: models/)",
    )
    return parser.parse_args()


def print_report(report: dict) -> None:
    logger.info("=" * 60)
    logger.info("Training report")
    logger.info("=" * 60)
    for model_name, metrics in report.items():
        logger.info("  %s:", model_name)
        for k, v in metrics.items():
            if k == "feature_cols":
                continue   # skip long lists
            if isinstance(v, float):
                logger.info("    %-20s = %.4f", k, v)
            else:
                logger.info("    %-20s = %s", k, v)
    logger.info("=" * 60)


def run(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("%s v%s -- Model Trainer", settings.project_name, settings.version)
    logger.info("=" * 60)
    logger.info("Models    : %s", args.models)
    logger.info("Match IDs : %s", args.match_ids or "ALL")
    logger.info("Force     : %s", args.force)

    registry = ModelRegistry(models_dir=args.models_dir)
    pipeline = MLTrainingPipeline(registry=registry)

    with timer("Total training time"):
        report = pipeline.run(
            models=args.models,
            match_ids=args.match_ids,
            force=args.force,
        )

    print_report(report)

    available = registry.list_available()
    logger.info("Models now in registry: %s", available)
    logger.info("Models saved to: %s", registry.models_dir)


if __name__ == "__main__":
    run(parse_args())
