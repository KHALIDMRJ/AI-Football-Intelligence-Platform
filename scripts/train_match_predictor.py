"""
Train the match-outcome model from finished fixtures in the DB.

Usage
-----
    python scripts/train_match_predictor.py
    python scripts/train_match_predictor.py --min-matches 100

Reads from the configured DATABASE_URL (.env / PlatformSettings).
Writes the model to ``models/match_outcome_model.joblib`` and updates
``models/metadata.yaml``. The HTTP API picks up the new artefact on its
next prediction call (no restart required).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from football_ai.db.session import async_session_factory, dispose_engine
from football_ai.logger import get_logger, setup_logging
from football_ai.ml.training.match_outcome_trainer import train_match_outcome_model

logger = get_logger(__name__)


async def _main(min_matches: int) -> int:
    async with async_session_factory() as db:
        try:
            result = await train_match_outcome_model(db, min_matches=min_matches)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 2

    logger.info(
        "Trained match-outcome model: n=%d logloss=%.4f acc=%.4f",
        result.n_samples, result.train_logloss, result.train_accuracy,
    )
    return 0


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-matches",
        type=int,
        default=50,
        help="Minimum finished matches required to train (default: 50).",
    )
    args = parser.parse_args()

    try:
        rc = asyncio.run(_main(args.min_matches))
    finally:
        asyncio.run(dispose_engine())
    return rc


if __name__ == "__main__":
    sys.exit(main())
