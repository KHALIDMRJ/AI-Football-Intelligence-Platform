"""
Pitch zone mapper and direction normaliser.

Two responsibilities:
1. Map (x, y) coordinates to a flat zone index in an 18×12 grid.
2. Normalise all teams to attack left-to-right (mirror second-half coordinates).
"""

from __future__ import annotations

import pandas as pd

from football_ai.config import settings
from football_ai.constants import Cols
from football_ai.logger import get_logger
from football_ai.utils import zone_id

logger = get_logger(__name__)


class ZoneMapper:
    """
    Maps pitch coordinates to zone IDs and normalises attacking direction.

    Zone grid: 18 columns × 12 rows = 216 zones.
    Zone 0 = bottom-left corner (x=0, y=0).
    Zone 215 = top-right corner (x≈120, y≈80).

    Usage
    -----
    >>> mapper = ZoneMapper()
    >>> df = mapper.add_zones(actions_df)
    >>> df = mapper.normalise_direction(df)
    """

    def __init__(
        self,
        zones_x: int | None = None,
        zones_y: int | None = None,
    ) -> None:
        self.zones_x = zones_x or settings.pitch.zones_x
        self.zones_y = zones_y or settings.pitch.zones_y
        self.pitch_length = settings.pitch.length
        self.pitch_width = settings.pitch.width

    def add_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a ``zone_id`` column to a DataFrame of SPADL actions.
        Zone is computed from the action's start position.
        """
        df = df.copy()
        df[Cols.ZONE_ID] = df.apply(
            lambda r: zone_id(
                r.get(Cols.START_X, 0.0),
                r.get(Cols.START_Y, 0.0),
                self.zones_x,
                self.zones_y,
                self.pitch_length,
                self.pitch_width,
            ),
            axis=1,
        )
        return df

    def normalise_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all teams attack from left (x=0) to right (x=120).

        StatsBomb data has teams attacking in different directions by
        period and team. We detect the dominant direction from a team's
        shot locations in each period and mirror if necessary.

        This modifies start_x, start_y, end_x, end_y in place.
        """
        df = df.copy()

        if Cols.MATCH_ID not in df.columns or Cols.TEAM_ID not in df.columns:
            logger.warning("Cannot normalise direction: missing match_id or team_id")
            return df

        for (match_id, period, team_id), group in df.groupby(
            [Cols.MATCH_ID, Cols.PERIOD, Cols.TEAM_ID]
        ):
            # Check if this team is attacking right (towards x=120)
            # by looking at where their passes end on average
            if Cols.END_X in group.columns and len(group) > 2:
                mean_end_x = group[Cols.END_X].mean()
                if mean_end_x < self.pitch_length / 2:
                    # Team is attacking left → mirror all coordinates
                    mask = (
                        (df[Cols.MATCH_ID] == match_id)
                        & (df[Cols.PERIOD] == period)
                        & (df[Cols.TEAM_ID] == team_id)
                    )
                    df.loc[mask, Cols.START_X] = self.pitch_length - df.loc[mask, Cols.START_X]
                    df.loc[mask, Cols.START_Y] = self.pitch_width - df.loc[mask, Cols.START_Y]
                    df.loc[mask, Cols.END_X] = self.pitch_length - df.loc[mask, Cols.END_X]
                    df.loc[mask, Cols.END_Y] = self.pitch_width - df.loc[mask, Cols.END_Y]

        return df

    def zone_centre(self, zone: int) -> tuple[float, float]:
        """Return the (x, y) centre of a given zone ID."""
        row = zone // self.zones_x
        col = zone % self.zones_x
        x = (col + 0.5) / self.zones_x * self.pitch_length
        y = (row + 0.5) / self.zones_y * self.pitch_width
        return x, y

    def zone_grid_shape(self) -> tuple[int, int]:
        """Return (zones_y, zones_x) — matches numpy row/col convention."""
        return self.zones_y, self.zones_x

    def total_zones(self) -> int:
        return self.zones_x * self.zones_y
