"""
Abstract base adapter that all vendor-specific adapters must implement.

Adding a new vendor (e.g. Wyscout) means:
    1. Subclass BaseAdapter
    2. Implement load() and to_raw_events()
    3. Register in AdapterRegistry

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from football_ai.logger import get_logger
from football_ai.schemas import RawEvent

logger = get_logger(__name__)


class BaseAdapter(ABC):
    """
    Contract that every data-source adapter must satisfy.

    Attributes
    ----------
    source_name : str
        Human-readable name for this data source (e.g. "StatsBomb CSV").
    """

    source_name: str = "unknown"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"[{self.source_name}] Source not found: {self.source_path}"
            )

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the raw source file and return a DataFrame.

        Returns
        -------
        pd.DataFrame
            Raw, unmodified data from the vendor file.
        """

    @abstractmethod
    def to_raw_events(self, df: pd.DataFrame) -> list[RawEvent]:
        """
        Convert a vendor-specific DataFrame into a list of RawEvent objects.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``load()``.

        Returns
        -------
        list[RawEvent]
            Validated raw events ready for the normalisation pipeline.
        """

    def ingest(self) -> list[RawEvent]:
        """
        Full ingestion: load → convert → return validated events.
        This is the single public method callers should use.
        """
        logger.info("[%s] Loading from %s", self.source_name, self.source_path)
        df = self.load()
        logger.info("[%s] Loaded %d rows", self.source_name, len(df))
        events = self.to_raw_events(df)
        logger.info("[%s] Converted to %d RawEvent objects", self.source_name, len(events))
        return events
