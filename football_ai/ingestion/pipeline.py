"""
Ingestion pipeline orchestrator.

Ties together: adapter → validator → storage.
Single entry point for all data loading operations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ai.ingestion.adapters.base_adapter import BaseAdapter
from football_ai.ingestion.adapters.csv_adapter import CSVAdapter
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.ingestion.validators.schema_validator import (
    SchemaValidator,
    ValidationReport,
)
from football_ai.logger import get_logger
from football_ai.schemas import RawEvent

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow for a single data source.

    Flow
    ----
    1. Choose adapter (CSV, Wyscout, Opta, …)
    2. Load and convert to RawEvent objects
    3. Validate with SchemaValidator
    4. Persist valid events to Parquet
    5. Return the clean event list for downstream processing

    Usage
    -----
    >>> pipeline = IngestionPipeline()
    >>> events, report = pipeline.run_csv("data/raw/fus_FAR.csv", match_id="3813041")
    >>> print(report.summary())
    """

    def __init__(
        self,
        store: ParquetStore | None = None,
        validator: SchemaValidator | None = None,
        strict: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        store : ParquetStore, optional
            Storage backend.  Defaults to the standard directory layout.
        validator : SchemaValidator, optional
            Validator instance.  Defaults to SchemaValidator().
        strict : bool
            If True, raise ValueError when validation errors are found.
        """
        self.store = store or ParquetStore()
        self.validator = validator or SchemaValidator()
        self.strict = strict

    # ── Public entry points ───────────────────────────────────────────────────

    def run_csv(
        self,
        csv_path: str | Path,
        match_id: str | None = None,
        force: bool = False,
    ) -> tuple[list[RawEvent], ValidationReport]:
        """
        Ingest a StatsBomb-format CSV file.

        Parameters
        ----------
        csv_path : str | Path
        match_id : str, optional
            Override the match_id present in the CSV (handy for legacy files).
        force : bool
            Re-ingest even if the Parquet file already exists.

        Returns
        -------
        (list[RawEvent], ValidationReport)
        """
        adapter = CSVAdapter(csv_path, match_id_override=match_id)
        return self._run(adapter, force=force)

    def run_adapter(
        self,
        adapter: BaseAdapter,
        force: bool = False,
    ) -> tuple[list[RawEvent], ValidationReport]:
        """
        Ingest using any adapter (future-proofs adding Wyscout, Opta, etc.).

        Parameters
        ----------
        adapter : BaseAdapter
            A fully-constructed adapter instance.
        force : bool
            Re-ingest even if the Parquet file already exists.
        """
        return self._run(adapter, force=force)

    # ── Core orchestration ────────────────────────────────────────────────────

    def _run(
        self,
        adapter: BaseAdapter,
        force: bool = False,
    ) -> tuple[list[RawEvent], ValidationReport]:
        """Run the ingestion flow end-to-end."""
        # 1. Load raw events from the adapter
        all_events = adapter.ingest()

        if not all_events:
            logger.warning("Adapter returned 0 events -- nothing to ingest.")
            from football_ai.ingestion.validators.schema_validator import (
                ValidationReport,
            )
            return [], ValidationReport(total_events=0, valid_events=0)

        # Determine match_id from first event
        match_id = all_events[0].match_id
        logger.info("Ingesting match_id=%s (%d events)", match_id, len(all_events))

        # 2. Check if already stored (skip if not forced)
        if not force and self.store.has_raw(match_id):
            logger.info(
                "Raw data for match %s already exists. Use force=True to re-ingest.",
                match_id,
            )
            # Still validate, but load from store
            df = self.store.load_raw_events(match_id)
            all_events = self._df_to_raw_events(df)

        # 3. Validate
        report = self.validator.validate(all_events)

        if self.strict and not report.is_valid:
            raise ValueError(
                f"Validation failed for match {match_id}: {report.summary()}"
            )

        # 4. Filter to valid events only
        valid_events = self.validator.filter_valid(all_events, report)
        logger.info(
            "Valid events: %d / %d", len(valid_events), len(all_events)
        )

        # 5. Persist
        if force or not self.store.has_raw(match_id):
            self.store.save_raw_events(valid_events, match_id)

        return valid_events, report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _df_to_raw_events(df: pd.DataFrame) -> list[RawEvent]:
        """Reconstruct RawEvent objects from a stored Parquet DataFrame."""
        events = []
        for _, row in df.iterrows():
            try:
                events.append(RawEvent(**row.to_dict()))
            except Exception:
                pass
        return events
