"""
Preprocessing pipeline orchestrator.

Chains: SPADL normaliser → pitch mapper → possession detector → game-state builder
and saves the result to the Parquet store.
"""

from __future__ import annotations

import pandas as pd

from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.logger import get_logger
from football_ai.preprocessing.game_state.builder import GameStateBuilder
from football_ai.preprocessing.pitch.zone_mapper import ZoneMapper
from football_ai.preprocessing.possession.detector import PossessionDetector
from football_ai.preprocessing.spadl.normaliser import SPADLNormaliser
from football_ai.schemas import RawEvent, SPADLAction

logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing flow for a match.

    Flow
    ----
    RawEvent list
        → SPADLNormaliser      (type/result/body-part mapping)
        → ZoneMapper           (zone_id + direction normalisation)
        → PossessionDetector   (possession_id, chain_id)
        → GameStateBuilder     (label_scores, label_concedes, score context)
        → ParquetStore.save()

    Usage
    -----
    >>> pipeline = PreprocessingPipeline()
    >>> df = pipeline.run(raw_events, match_id="3813041", home_team_id="2761")
    """

    def __init__(
        self,
        store: ParquetStore | None = None,
        normaliser: SPADLNormaliser | None = None,
        zone_mapper: ZoneMapper | None = None,
        possession_detector: PossessionDetector | None = None,
        game_state_builder: GameStateBuilder | None = None,
        normalise_direction: bool = True,
    ) -> None:
        self.store = store or ParquetStore()
        self.normaliser = normaliser or SPADLNormaliser()
        self.zone_mapper = zone_mapper or ZoneMapper()
        self.possession_detector = possession_detector or PossessionDetector()
        self.game_state_builder = game_state_builder or GameStateBuilder()
        self.normalise_direction = normalise_direction

    def run(
        self,
        raw_events: list[RawEvent],
        match_id: str,
        home_team_id: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline for one match.

        Parameters
        ----------
        raw_events : list[RawEvent]
        match_id : str
        home_team_id : str, optional
            Required for accurate score tracking in GameStateBuilder.
            If None, first team encountered is treated as home.
        force : bool
            Re-process even if SPADL Parquet already exists.

        Returns
        -------
        pd.DataFrame
            Fully preprocessed SPADL actions ready for feature engineering.
        """
        if not force and self.store.has_processed(match_id):
            logger.info(
                "Preprocessed data for match %s already exists -- loading from store.",
                match_id,
            )
            return self.store.load_spadl_actions(match_id)

        logger.info("Preprocessing match %s (%d raw events)", match_id, len(raw_events))

        # Step 1: SPADL normalisation
        actions: list[SPADLAction] = self.normaliser.normalise(raw_events)
        df = pd.DataFrame([a.model_dump() for a in actions])

        if df.empty:
            logger.warning("No actions produced for match %s", match_id)
            return df

        # Step 2: Zone mapping (adds zone_id)
        df = self.zone_mapper.add_zones(df)

        # Step 3: Direction normalisation (optional)
        if self.normalise_direction:
            df = self.zone_mapper.normalise_direction(df)

        # Step 4: Possession and chain detection
        df = self.possession_detector.detect(df)

        # Step 5: Game-state labels
        if home_team_id is None:
            home_team_id = str(df["team_id"].iloc[0]) if not df.empty else "unknown"
        df = self.game_state_builder.build(df, home_team_id=home_team_id)

        # Step 6: Persist
        self.store.save_spadl_actions(df, match_id)
        logger.info(
            "Preprocessing complete for match %s: %d actions saved", match_id, len(df)
        )

        return df
