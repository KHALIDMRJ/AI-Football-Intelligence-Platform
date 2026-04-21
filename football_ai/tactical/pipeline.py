"""
Tactical intelligence pipeline.

Orchestrates all tactical modules for a single match in one call:

    1. Load VAEP Parquet + player/team summaries (Phase 6 output)
    2. Run PlayerRanker → all_rankings dict
    3. Run WeaknessDetector → per-team weakness DataFrames
    4. Run FormationAnalyser → per-team FormationResult
    5. Run PressureMap → per-team PressureMapResult
    6. Run ReportBuilder → MatchReport
    7. Save all outputs to Parquet / JSON

Saved artefacts
---------------
    data/processed/match_{id}_tactical_report.json
    data/processed/match_{id}_player_rankings.parquet
    data/processed/match_{id}_weaknesses_{team_id}.parquet
    data/processed/match_{id}_pressure_{team_id}.parquet

Usage
-----
>>> pipeline = TacticalPipeline()
>>> result = pipeline.run(match_id="3813041")
>>> print(result.match_report.to_dict())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger
from football_ai.tactical.formation_analyser import FormationAnalyser, FormationResult
from football_ai.tactical.player_ranker import PlayerRanker
from football_ai.tactical.pressure_map import PressureMap, PressureMapResult
from football_ai.tactical.report_builder import MatchReport, ReportBuilder
from football_ai.tactical.weakness_detector import WeaknessDetector
from football_ai.utils import ensure_dir, save_parquet, timer

logger = get_logger(__name__)


@dataclass
class TacticalResult:
    """All tactical outputs for a single match."""

    match_id:        str
    match_report:    MatchReport | None                  = None
    rankings:        dict[str, pd.DataFrame]               = field(default_factory=dict)
    weaknesses:      dict[str, pd.DataFrame]               = field(default_factory=dict)
    formations:      dict[str, FormationResult]            = field(default_factory=dict)
    pressure_maps:   dict[str, PressureMapResult]          = field(default_factory=dict)


class TacticalPipeline:
    """
    Runs the full tactical intelligence pipeline for one match.

    Parameters
    ----------
    store : ParquetStore, optional
    ranker : PlayerRanker, optional
    weakness_detector : WeaknessDetector, optional
    formation_analyser : FormationAnalyser, optional
    pressure_map : PressureMap, optional
    report_builder : ReportBuilder, optional

    Usage
    -----
    >>> pipeline = TacticalPipeline()
    >>> result = pipeline.run(match_id="3813041")
    """

    def __init__(
        self,
        store: ParquetStore | None = None,
        ranker: PlayerRanker | None = None,
        weakness_detector: WeaknessDetector | None = None,
        formation_analyser: FormationAnalyser | None = None,
        pressure_map: PressureMap | None = None,
        report_builder: ReportBuilder | None = None,
    ) -> None:
        self.store             = store              or ParquetStore()
        self.ranker            = ranker             or PlayerRanker()
        self.weakness_detector = weakness_detector  or WeaknessDetector()
        self.formation_analyser = formation_analyser or FormationAnalyser()
        self.pressure_map      = pressure_map       or PressureMap()
        self.report_builder    = report_builder     or ReportBuilder()

    def run(
        self,
        match_id: str,
        force: bool = False,
    ) -> TacticalResult:
        """
        Run the tactical pipeline for a single match.

        Parameters
        ----------
        match_id : str
            Must have VAEP Parquet already computed (Phase 6).
        force : bool
            Recompute even if report JSON already exists.

        Returns
        -------
        TacticalResult
        """
        report_path = self._report_path(match_id)
        if not force and report_path.exists():
            logger.info(
                "Tactical report for match %s already exists -- skipping.", match_id
            )
            return self._load_cached(match_id)

        # ── Load VAEP data ────────────────────────────────────────────────────
        with timer(f"Load VAEP [{match_id}]"):
            vaep_df = self._load_vaep(match_id)

        if vaep_df.empty:
            logger.error(
                "No VAEP data for match %s. Run Phase 6 first.", match_id
            )
            return TacticalResult(match_id=match_id)

        player_summary = self._load_summary(match_id, "player")
        team_summary   = self._load_summary(match_id, "team")

        # ── Identify teams ────────────────────────────────────────────────────
        from football_ai.constants import Cols
        team_ids = vaep_df[Cols.TEAM_ID].unique().tolist()
        home_team_id = str(team_ids[0]) if team_ids else "unknown"
        away_team_id = str(team_ids[1]) if len(team_ids) > 1 else "unknown"

        result = TacticalResult(match_id=match_id)

        # ── Player rankings ───────────────────────────────────────────────────
        with timer("Player rankings"):
            result.rankings = self.ranker.all_rankings(vaep_df)

        # ── Weakness detection ────────────────────────────────────────────────
        with timer("Weakness detection"):
            result.weaknesses = self.weakness_detector.detect_all_teams(vaep_df)

        # ── Formation analysis ────────────────────────────────────────────────
        with timer("Formation analysis"):
            result.formations = self.formation_analyser.analyse_all_teams(vaep_df)

        # ── Pressure maps ─────────────────────────────────────────────────────
        with timer("Pressure maps"):
            result.pressure_maps = self.pressure_map.build_all_teams(vaep_df)

        # ── Match report ──────────────────────────────────────────────────────
        with timer("Build match report"):
            result.match_report = self.report_builder.build_match_report(
                vaep_df=vaep_df,
                player_summary=player_summary,
                team_summary=team_summary,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                formations=result.formations,
                weaknesses=result.weaknesses,
                pressure_maps=result.pressure_maps,
            )

        # ── Save artefacts ────────────────────────────────────────────────────
        self._save(match_id, result)

        logger.info(
            "Tactical pipeline complete for match %s.", match_id
        )
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, match_id: str, result: TacticalResult) -> None:
        out_dir = self.store.processed_dir
        ensure_dir(out_dir)

        # ── Match report JSON ─────────────────────────────────────────────────
        if result.match_report:
            path = self._report_path(match_id)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result.match_report.to_dict(), fh, indent=2, default=str)
            logger.info("Saved tactical report -> %s", path)

        # ── Rankings Parquet (overall only to keep it simple) ─────────────────
        if "overall" in result.rankings and not result.rankings["overall"].empty:
            save_parquet(
                result.rankings["overall"],
                out_dir / f"match_{match_id}_player_rankings.parquet",
            )

        # ── Weaknesses Parquet ────────────────────────────────────────────────
        for team_id, weak_df in result.weaknesses.items():
            if not weak_df.empty:
                save_parquet(
                    weak_df,
                    out_dir / f"match_{match_id}_weaknesses_{team_id}.parquet",
                )

        # ── Pressure maps Parquet ─────────────────────────────────────────────
        for team_id, pm_result in result.pressure_maps.items():
            if not pm_result.opponent_threat.empty:
                save_parquet(
                    pm_result.opponent_threat,
                    out_dir / f"match_{match_id}_pressure_{team_id}.parquet",
                )

    def _load_cached(self, match_id: str) -> TacticalResult:
        """Load previously saved tactical result."""
        result = TacticalResult(match_id=match_id)
        report_path = self._report_path(match_id)
        if report_path.exists():
            with open(report_path, encoding="utf-8") as fh:
                json.load(fh)
            logger.info("Loaded cached tactical report <- %s", report_path)
            # MatchReport is not re-hydrated from JSON here (used by API layer)
        return result

    def _load_vaep(self, match_id: str) -> pd.DataFrame:
        path = self.store.processed_dir / f"match_{match_id}_vaep.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _load_summary(self, match_id: str, kind: str) -> pd.DataFrame:
        path = self.store.processed_dir / f"match_{match_id}_{kind}_summary.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _report_path(self, match_id: str) -> Path:
        return self.store.processed_dir / f"match_{match_id}_tactical_report.json"