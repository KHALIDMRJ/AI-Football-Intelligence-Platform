"""
Schema validator — validates a list of RawEvent objects before they
enter the preprocessing pipeline.

Responsibilities:
- Check pitch coordinate bounds
- Detect implausible period / minute values
- Flag missing player / team data
- Produce a ValidationReport with per-event details

ValidationIssue.event_index is the 0-based *list position* of the event,
matching the index used by filter_valid().
"""

from __future__ import annotations

from dataclasses import dataclass, field

from football_ai.logger import get_logger
from football_ai.schemas import RawEvent

logger = get_logger(__name__)


# ── Validation report ─────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    event_index: int   # 0-based position in the events list
    field: str
    message: str
    severity: str      # "warning" | "error"


@dataclass
class ValidationReport:
    total_events: int
    valid_events: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """True if there are no error-severity issues (warnings are acceptable)."""
        return self.error_count == 0

    def summary(self) -> str:
        return (
            f"Validated {self.total_events} events: "
            f"{self.valid_events} valid, "
            f"{self.error_count} errors, "
            f"{self.warning_count} warnings"
        )


# ── Validator ─────────────────────────────────────────────────────────────────

class SchemaValidator:
    """
    Stateless validator for lists of RawEvent objects.

    event_index in all issues is the 0-based list position of the offending
    event, so filter_valid() can match by position without ambiguity.

    Usage
    -----
    >>> validator = SchemaValidator()
    >>> report = validator.validate(events)
    >>> clean = validator.filter_valid(events, report)
    """

    _COORD_TOL: float = 0.5  # allow tiny out-of-bounds due to tracking noise

    def validate(self, events: list[RawEvent]) -> ValidationReport:
        """
        Validate all events and return a ValidationReport.
        Never raises — callers decide what to do based on the report.
        """
        issues: list[ValidationIssue] = []
        valid_count = 0

        for list_pos, evt in enumerate(events):
            evt_issues = self._validate_event(evt, list_pos)
            issues.extend(evt_issues)
            if not any(i.severity == "error" for i in evt_issues):
                valid_count += 1

        report = ValidationReport(
            total_events=len(events),
            valid_events=valid_count,
            issues=issues,
        )
        logger.info(report.summary())
        if report.error_count > 0:
            logger.warning(
                "%d validation errors found — check events before processing",
                report.error_count,
            )
        return report

    def filter_valid(
        self, events: list[RawEvent], report: ValidationReport
    ) -> list[RawEvent]:
        """
        Return only the events whose list positions have no error-level issues.

        Parameters
        ----------
        events : list[RawEvent]
            The same list that was passed to validate().
        report : ValidationReport
            Produced by validate() for this exact list.
        """
        error_positions = {
            i.event_index for i in report.issues if i.severity == "error"
        }
        return [evt for pos, evt in enumerate(events) if pos not in error_positions]

    # ── Per-event checks ───────────────────────────────────────────────────────

    def _validate_event(
        self, evt: RawEvent, list_pos: int
    ) -> list[ValidationIssue]:
        """Return all issues found for a single event.  list_pos is stored in each issue."""
        issues: list[ValidationIssue] = []

        def warn(field: str, message: str) -> None:
            issues.append(ValidationIssue(list_pos, field, message, "warning"))

        def error(field: str, message: str) -> None:
            issues.append(ValidationIssue(list_pos, field, message, "error"))

        # ── Period ────────────────────────────────────────────────────────────
        period = evt.period
        if period is None or not (1 <= period <= 5):
            error("period", f"period={period} not in valid range [1, 5]")

        # ── Minute ────────────────────────────────────────────────────────────
        if evt.minute is not None and (evt.minute < 0 or evt.minute > 130):
            warn("minute", f"minute={evt.minute} outside plausible range [0, 130]")

        # ── Coordinates ───────────────────────────────────────────────────────
        for coord, val, lo, hi in [
            ("start_x", evt.start_x, 0.0, 120.0),
            ("start_y", evt.start_y, 0.0,  80.0),
            ("end_x",   evt.end_x,   0.0, 120.0),
            ("end_y",   evt.end_y,   0.0,  80.0),
        ]:
            if val is None:
                continue
            if val < lo - self._COORD_TOL or val > hi + self._COORD_TOL:
                warn(coord, f"{coord}={val:.2f} outside pitch bounds [{lo}, {hi}]")

        # ── Player (warning, not error — some events have no player) ──────────
        if evt.player_id in (None, "None", "nan", ""):
            warn("player_id", "player_id is missing")

        # ── Team ──────────────────────────────────────────────────────────────
        if evt.team_id in (None, "None", "nan", ""):
            warn("team_id", "team_id is missing")

        # ── xG on non-shot (warning) ──────────────────────────────────────────
        if evt.xg and evt.xg > 0:
            if evt.event_type and evt.event_type.lower() not in ("shot", "goal", "penalty"):
                warn(
                    "xg",
                    f"xg={evt.xg:.4f} on non-shot event type '{evt.event_type}'",
                )

        return issues
