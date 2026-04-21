"""
Football AI Platform — End-to-End Pipeline Runner

Phases:
    2  Ingestion    (CSV → RawEvent → Parquet)
    3  Preprocessing (SPADL → possession → game state labels)
    4  Features     (spatial + temporal + contextual, 105-dim)
    5  Models       (xG, xT, P_scores, P_concedes)
    6  VAEP         (V(Si) → V(ai) → player/team aggregation)
    7  Tactical     (rankings, weaknesses, formations, pressure maps, report)

Usage
-----
    python scripts/run_pipeline.py --input data/raw/fus_FAR.csv
    python scripts/run_pipeline.py --input data/raw/fus_FAR.csv --skip-training
    python scripts/run_pipeline.py --input data/raw/fus_FAR.csv --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from football_ai.config import settings
from football_ai.features.assembler import FeatureAssembler
from football_ai.ingestion.pipeline import IngestionPipeline
from football_ai.logger import get_logger, setup_logging
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.ml.training.pipeline import MLTrainingPipeline
from football_ai.preprocessing.pipeline import PreprocessingPipeline
from football_ai.tactical.pipeline import TacticalPipeline
from football_ai.utils import timer
from football_ai.vaep.pipeline import VAEPPipeline

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Football AI Platform -- pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",        type=Path, default=None)
    p.add_argument("--match-id",     type=str,  default=None)
    p.add_argument("--home-team-id", type=str,  default=None)
    p.add_argument("--force",        action="store_true")
    p.add_argument("--skip-training",action="store_true")
    p.add_argument("--skip-vaep",    action="store_true")
    p.add_argument("--skip-tactical",action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("%s v%s", settings.project_name, settings.version)
    logger.info("=" * 60)

    input_path = args.input or settings.abs(settings.paths.data_raw) / "events.csv"
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    registry = ModelRegistry()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    with timer("Phase 2 -- Ingestion"):
        events, report = IngestionPipeline(strict=False).run_csv(
            input_path, match_id=args.match_id, force=args.force,
        )
    logger.info(report.summary())
    if not events:
        logger.error("No valid events -- aborting.")
        sys.exit(1)

    match_id = events[0].match_id

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    with timer("Phase 3 -- Preprocessing"):
        spadl_df = PreprocessingPipeline().run(
            events, match_id=match_id,
            home_team_id=args.home_team_id, force=args.force,
        )
    logger.info("SPADL: %d actions", len(spadl_df))

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    with timer("Phase 4 -- Feature engineering"):
        feature_df = FeatureAssembler().assemble(
            spadl_df, match_id=match_id, force=args.force,
        )
    n_feat = len([c for c in feature_df.columns if c.startswith("f_")])
    logger.info("Features: %d rows x %d feature cols", len(feature_df), n_feat)

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    if not args.skip_training:
        with timer("Phase 5 -- Model training"):
            report_train = MLTrainingPipeline(registry=registry).run(force=args.force)
        trained = [k for k, v in report_train.items() if "error" not in v]
        logger.info("Phase 5 complete: %s", trained)
    else:
        logger.info("Phase 5 -- skipped (--skip-training)")

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    vaep_result = None
    if not args.skip_vaep:
        required = {"p_scores", "p_concedes"}
        if not required.issubset(set(registry.list_available())):
            logger.error(
                "Phase 6 skipped — models not ready: %s",
                required - set(registry.list_available()),
            )
        else:
            with timer("Phase 6 -- VAEP"):
                vaep_result = VAEPPipeline(registry=registry).run(
                    match_id=match_id, force=args.force,
                )
            logger.info(
                "Phase 6 complete: %d actions, %d players",
                vaep_result.total_actions, vaep_result.n_players,
            )

            if not vaep_result.player_summary.empty:
                logger.info("Top 5 players by VAEP:")
                for _, row in vaep_result.player_summary.head(5).iterrows():
                    logger.info(
                        "  %-28s  VAEP=%+.4f  Off=%.4f  Def=%.4f",
                        row.get("player_name", "—"),
                        row.get("vaep_total",    0.0),
                        row.get("vaep_offensive", 0.0),
                        row.get("vaep_defensive", 0.0),
                    )
    else:
        logger.info("Phase 6 -- skipped (--skip-vaep)")

    # ── Phase 7 ───────────────────────────────────────────────────────────────
    if not args.skip_tactical and vaep_result is not None:
        with timer("Phase 7 -- Tactical intelligence"):
            tactical_result = TacticalPipeline().run(
                match_id=match_id, force=args.force,
            )

        if tactical_result.match_report:
            mr = tactical_result.match_report
            logger.info("-" * 50)
            logger.info("Tactical report: %s vs %s", mr.home_team_name, mr.away_team_name)
            logger.info("  xG     : %.3f - %.3f", mr.xg_home, mr.xg_away)
            logger.info("  VAEP   : %.3f - %.3f", mr.vaep_home, mr.vaep_away)
            logger.info("  Form.  : %s vs %s", mr.home_formation, mr.away_formation)

        if tactical_result.weaknesses:
            for team_id, w_df in tactical_result.weaknesses.items():
                has_risk = not w_df.empty and "risk_level" in w_df.columns
                critical = (
                    (w_df["risk_level"] == "critical").sum() if has_risk else 0
                )
                logger.info("  Weaknesses [%s]: %d critical zones", team_id, critical)

        for team_id, form in tactical_result.formations.items():
            logger.info(
                "  Formation [%s]: %s (conf=%.2f)",
                form.team_name, form.formation_str, form.confidence,
            )

        logger.info("-" * 50)
    elif not args.skip_tactical:
        logger.info("Phase 7 -- skipped (Phase 6 VAEP required first)")
    else:
        logger.info("Phase 7 -- skipped (--skip-tactical)")

    logger.info("=" * 60)
    logger.info("Pipeline complete  match_id=%s", match_id)
    logger.info("=" * 60)


if __name__ == "__main__":
    run(parse_args())
