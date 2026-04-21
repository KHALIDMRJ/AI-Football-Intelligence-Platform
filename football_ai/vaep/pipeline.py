"""
VAEP pipeline.

Orchestrates the full VAEP computation flow for one or more matches:

    1. Load feature Parquet  (Phase 4 output)
    2. Load SPADL Parquet    (Phase 3 output, needed for xT)
    3. Run StateValueComputer   → p_scores, p_concedes, state_value
    4. Run ActionValueComputer  → vaep_value, vaep_offensive, vaep_defensive
    5. Optionally add xT deltas from the xT model
    6. Save per-action VAEP Parquet
    7. Compute and return player + team summaries

Saved artefacts
---------------
    data/processed/match_{match_id}_vaep.parquet
        Per-action DataFrame with all VAEP columns.

    data/processed/match_{match_id}_player_summary.parquet
        Per-player aggregated VAEP.

    data/processed/match_{match_id}_team_summary.parquet
        Per-team aggregated VAEP.

Usage
-----
>>> pipeline = VAEPPipeline(registry=registry)
>>> result = pipeline.run(match_id="3813041")
>>> print(result.player_summary.head(10))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger
from football_ai.ml.models.p_concedes_model import PConcedesModel
from football_ai.ml.models.p_scores_model import PScoresModel
from football_ai.ml.models.xt_model import XTModel
from football_ai.ml.serving.model_registry import ModelRegistry
from football_ai.utils import ensure_dir, save_parquet, timer
from football_ai.vaep.action_value import ActionValueComputer
from football_ai.vaep.aggregator import VAEPAggregator
from football_ai.vaep.state_value import StateValueComputer

logger = get_logger(__name__)


@dataclass
class VAEPResult:
    """Container for all VAEP outputs from a single match run."""

    match_id: str
    actions: pd.DataFrame          = field(default_factory=pd.DataFrame)
    player_summary: pd.DataFrame   = field(default_factory=pd.DataFrame)
    team_summary: pd.DataFrame     = field(default_factory=pd.DataFrame)

    @property
    def total_actions(self) -> int:
        return len(self.actions)

    @property
    def n_players(self) -> int:
        return len(self.player_summary)

    def __repr__(self) -> str:
        return (
            f"VAEPResult(match_id={self.match_id!r}, "
            f"actions={self.total_actions}, "
            f"players={self.n_players})"
        )


class VAEPPipeline:
    """
    End-to-end VAEP computation pipeline.

    Loads trained models from the ModelRegistry and runs the full
    state-value → action-value → aggregation chain.

    Parameters
    ----------
    registry : ModelRegistry
        Source of trained P_scores, P_concedes, and xT models.
    store : ParquetStore, optional
        Data store for reading feature files and writing VAEP results.
    aggregator : VAEPAggregator, optional
        Custom aggregator. Defaults to VAEPAggregator(min_actions=5).
    include_xt : bool
        Whether to append xT columns from the XTModel. Default True.
        Falls back silently if the xT model is not in the registry.

    Usage
    -----
    >>> registry = ModelRegistry()
    >>> pipeline = VAEPPipeline(registry=registry)
    >>> result = pipeline.run(match_id="3813041")
    """

    def __init__(
        self,
        registry: ModelRegistry,
        store: ParquetStore | None = None,
        aggregator: VAEPAggregator | None = None,
        include_xt: bool = True,
    ) -> None:
        self.registry   = registry
        self.store      = store or ParquetStore()
        self.aggregator = aggregator or VAEPAggregator()
        self.include_xt = include_xt

        # Lazy-loaded models (loaded on first call to run())
        self._p_scores_model:   PScoresModel | None   = None
        self._p_concedes_model: PConcedesModel | None = None
        self._xt_model:         XTModel | None        = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        match_id: str,
        force: bool = False,
    ) -> VAEPResult:
        """
        Compute VAEP for a single match end-to-end.

        Parameters
        ----------
        match_id : str
            Match ID to process. Feature Parquet must already exist.
        force : bool
            Recompute even if VAEP Parquet already exists.

        Returns
        -------
        VAEPResult
            Container with per-action, per-player, and per-team DataFrames.
        """
        vaep_path = self._vaep_path(match_id)

        if not force and vaep_path.exists():
            logger.info(
                "VAEP for match %s already computed -- loading from store.", match_id
            )
            actions  = pd.read_parquet(vaep_path)
            players  = self._load_summary(match_id, "player")
            teams    = self._load_summary(match_id, "team")
            return VAEPResult(match_id=match_id, actions=actions,
                              player_summary=players, team_summary=teams)

        # ── Load models ────────────────────────────────────────────────────────
        self._load_models()

        # ── Load feature data ──────────────────────────────────────────────────
        with timer(f"Load features [{match_id}]"):
            feature_df = self.store.load_features(match_id)

        logger.info(
            "Loaded features for match %s: %d actions x %d columns",
            match_id, len(feature_df), len(feature_df.columns),
        )

        # ── State values ───────────────────────────────────────────────────────
        with timer("State value computation"):
            sv_computer = StateValueComputer(
                self._p_scores_model,   # type: ignore[arg-type]
                self._p_concedes_model,  # type: ignore[arg-type]
            )
            feature_df = sv_computer.compute(feature_df)

        # ── Action values ──────────────────────────────────────────────────────
        with timer("Action value computation"):
            av_computer = ActionValueComputer()
            feature_df  = av_computer.compute(feature_df)

        # ── xT deltas (optional) ──────────────────────────────────────────────
        if self.include_xt and self._xt_model is not None:
            with timer("xT scoring"):
                feature_df = self._xt_model.score_actions(feature_df)
        elif self.include_xt:
            logger.info("xT model not available -- skipping xT columns.")

        # ── Save per-action VAEP Parquet ──────────────────────────────────────
        save_parquet(feature_df, vaep_path)

        # ── Aggregate ─────────────────────────────────────────────────────────
        with timer("Player aggregation"):
            player_summary = self.aggregator.player_summary(feature_df)

        with timer("Team aggregation"):
            team_summary = self.aggregator.team_summary(feature_df)

        # ── Save summaries ────────────────────────────────────────────────────
        save_parquet(player_summary, self._summary_path(match_id, "player"))
        save_parquet(team_summary,   self._summary_path(match_id, "team"))

        result = VAEPResult(
            match_id=match_id,
            actions=feature_df,
            player_summary=player_summary,
            team_summary=team_summary,
        )

        logger.info(
            "VAEP complete for match %s: %d actions, %d players ranked.",
            match_id, result.total_actions, result.n_players,
        )

        return result

    def run_all(
        self,
        match_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, VAEPResult]:
        """
        Run VAEP computation for multiple matches.

        Parameters
        ----------
        match_ids : list[str], optional
            Specific match IDs. Defaults to all feature-ready matches.
        force : bool
            Recompute all matches.

        Returns
        -------
        dict[str, VAEPResult]
            Keyed by match_id.
        """
        ids = match_ids or self.store.list_feature_matches()
        if not ids:
            logger.warning("No feature matches found. Run run_pipeline.py first.")
            return {}

        results: dict[str, VAEPResult] = {}
        for mid in ids:
            logger.info("Processing match %s ...", mid)
            try:
                results[mid] = self.run(mid, force=force)
            except Exception as exc:
                logger.error("VAEP failed for match %s: %s", mid, exc, exc_info=True)

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load P_scores, P_concedes, and optionally xT from the registry."""
        if self._p_scores_model is None:
            self._p_scores_model = PScoresModel.load(self.registry)

        if self._p_concedes_model is None:
            self._p_concedes_model = PConcedesModel.load(self.registry)

        if self.include_xt and self._xt_model is None:
            if self.registry.exists("xt"):
                self._xt_model = XTModel.load(self.registry)
            else:
                logger.warning("xT model not found in registry.")

    def _vaep_path(self, match_id: str) -> Path:
        path = self.store.processed_dir / f"match_{match_id}_vaep.parquet"
        ensure_dir(path.parent)
        return path

    def _summary_path(self, match_id: str, kind: str) -> Path:
        return self.store.processed_dir / f"match_{match_id}_{kind}_summary.parquet"

    def _load_summary(self, match_id: str, kind: str) -> pd.DataFrame:
        path = self._summary_path(match_id, kind)
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()