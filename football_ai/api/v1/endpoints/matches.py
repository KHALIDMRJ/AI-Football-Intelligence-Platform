"""
Matches endpoints (v1).

Two surfaces under ``/matches``:

* Pipeline / analytics (``/``, ``/{match_id}``, ``/{match_id}/report``) —
  parquet + tactical pipeline output via ``MatchService``. Pre-existing.
* DB fixtures (``/fixtures``, ``/fixtures/{id}``, ``/fixtures/{id}/events``,
  ``/live``) — Postgres/SQLite-backed core schedule. Added in Phase 4.

Route order matters: ``/live`` and ``/fixtures`` are registered before the
``/{match_id}`` catch-all to avoid FastAPI matching them as match IDs.

RBAC
----
Public: ``/fixtures`` list + ``/fixtures/{id}`` detail.
pro_analyst+: ``/live`` (live feed is premium) and ``/fixtures/{id}/events``
(event-level granularity drives the prediction engine — paywalled to
protect the feature moat).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
import re
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from football_ai.api.dependencies import get_db, require_role
from football_ai.api.schemas.catalog import (
    MatchEventOut,
    MatchFixtureOut,
    PagedResponse,
)
from football_ai.api.schemas.responses import (
    MatchDetailResponse,
    MatchListResponse,
    MatchReportResponse,
)
from football_ai.api.services.match_service import MatchService
from football_ai.cache import cached
from football_ai.config import settings
from football_ai.core.exceptions import DataNotFoundError
from football_ai.crud import match as match_crud
from football_ai.ingestion.pipeline import IngestionPipeline
from football_ai.ingestion.storage.parquet_store import ParquetStore
from football_ai.models.match import MatchStatus
from football_ai.models.user import UserRole

router = APIRouter(prefix="/matches", tags=["matches"])

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
_ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".parquet"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _match_service() -> MatchService:
    return MatchService()


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "").name
    clean = _SAFE_NAME_RE.sub("_", name).strip("._")
    if not clean:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")
    return clean[:120]


async def _save_upload(file: UploadFile, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(target, "wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Upload exceeds {_MAX_UPLOAD_BYTES} bytes.",
                )
            fh.write(chunk)
    return total


def _artifact_status(match_id: str) -> dict[str, object]:
    store = ParquetStore()
    raw_ready = store.has_raw(match_id)
    spadl_ready = store.has_processed(match_id)
    feature_ready = store.has_features(match_id)
    vaep_ready = (store.processed_dir / f"match_{match_id}_vaep.parquet").exists()
    report_ready = (
        store.processed_dir / f"match_{match_id}_tactical_report.json"
    ).exists()
    steps = [
        {"key": "validated", "label": "File validated", "status": "done" if raw_ready else "pending"},
        {"key": "spadl", "label": "Converting to SPADL format", "status": "done" if spadl_ready else "pending"},
        {"key": "xg", "label": "Computing xG model", "status": "done" if feature_ready else "pending"},
        {"key": "xt", "label": "Running xT mapping", "status": "done" if feature_ready else "pending"},
        {"key": "vaep", "label": "VAEP pipeline", "status": "done" if vaep_ready else "pending"},
        {"key": "report", "label": "Generating AI tactical report", "status": "done" if report_ready else "pending"},
    ]
    done = sum(1 for step in steps if step["status"] == "done")
    status_value = "ready" if report_ready or vaep_ready else ("processing" if raw_ready else "unknown")
    active_seen = False
    for step in steps:
        if step["status"] == "pending" and not active_seen and raw_ready:
            step["status"] = "active"
            active_seen = True
    return {
        "match_id": match_id,
        "status": status_value,
        "stage": next((str(s["key"]) for s in steps if s["status"] == "active"), status_value),
        "progress": round((done / len(steps)) * 100),
        "steps": steps,
    }


@router.get(
    "",
    response_model=MatchListResponse,
    summary="List all processed matches",
    description=(
        "Returns a compact summary for every match that has been "
        "processed through the full pipeline (Phases 2–7)."
    ),
)
async def list_matches(
    service: MatchService = Depends(_match_service),
) -> MatchListResponse:
    matches = service.get_match_list()
    return MatchListResponse(count=len(matches), matches=matches)


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Securely upload a match data file",
    description=(
        "Accepts small CSV or Parquet event-data files. CSV files are validated "
        "through the ingestion pipeline and stored as raw Parquet. This endpoint "
        "does not claim full VAEP/tactical processing; use /matches/{id}/status "
        "to inspect which downstream artifacts exist."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def upload_match(file: UploadFile = File(...)) -> dict[str, object]:
    filename = _safe_upload_name(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .csv and .parquet uploads are accepted.",
        )
    if file.content_type not in {
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
        "application/vnd.apache.parquet",
        "binary/octet-stream",
    }:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported upload content type.",
        )

    upload_dir = settings.abs(settings.paths.data_raw) / "uploads"
    target = upload_dir / f"{uuid.uuid4()}_{filename}"
    size = await _save_upload(file, target)

    if suffix == ".csv":
        try:
            events, report = await run_in_threadpool(
                IngestionPipeline(strict=False).run_csv,
                target,
                None,
                True,
            )
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"CSV ingestion failed: {exc.__class__.__name__}",
            ) from exc
        if not events:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="CSV did not contain valid match events.",
            )
        match_id = str(events[0].match_id)
        response = _artifact_status(match_id)
        response.update(
            {
                "id": match_id,
                "bytes": size,
                "validation": report.summary(),
            }
        )
        return response

    match_id = target.stem
    return {
        "id": match_id,
        "match_id": match_id,
        "status": "accepted",
        "stage": "validated",
        "progress": 10,
        "bytes": size,
        "detail": "Parquet upload stored; downstream processing must be run separately.",
    }


# ── Phase 4: DB-backed fixtures ──────────────────────────────────────────────
#
# IMPORTANT: these routes MUST be declared before ``/{match_id}`` so FastAPI
# doesn't match ``/live`` or ``/fixtures`` as a pipeline match-id string.

@router.get(
    "/fixtures",
    response_model=PagedResponse[MatchFixtureOut],
    summary="List fixtures (DB)",
    description=(
        "Paged list of matches in the core DB, with filters for league, "
        "season, status, involved team, and date window."
    ),
)
@cached(ttl=30, namespace="fixtures", response_model=PagedResponse[MatchFixtureOut])
async def list_fixtures(
    db: Annotated[AsyncSession, Depends(get_db)],
    league: str | None = Query(None),
    season: str | None = Query(None),
    match_status: MatchStatus | None = Query(
        None, description="scheduled | live | finished | postponed | cancelled"
    ),
    team_id: uuid.UUID | None = Query(None, description="Home or away team."),
    from_date: datetime | None = Query(None, description="ISO timestamp, inclusive."),
    to_date: datetime | None = Query(None, description="ISO timestamp, inclusive."),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PagedResponse[MatchFixtureOut]:
    rows, total = await match_crud.list_matches(
        db,
        league=league,
        season=season,
        status=match_status,
        team_id=team_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
    )
    return PagedResponse[MatchFixtureOut](
        total=total,
        limit=limit,
        offset=offset,
        items=[MatchFixtureOut.model_validate(m) for m in rows],
    )


@router.get(
    "/live",
    response_model=list[MatchFixtureOut],
    summary="Currently live matches (pro_analyst+)",
    description=(
        "All matches with status=live. Pagination is intentionally absent "
        "— the live slate is bounded (dozens of matches worldwide at peak)."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def list_live_fixtures(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MatchFixtureOut]:
    rows = await match_crud.list_live_matches(db)
    return [MatchFixtureOut.model_validate(m) for m in rows]


@router.get(
    "/fixtures/{match_id}",
    response_model=MatchFixtureOut,
    summary="Fixture detail (DB)",
    responses={404: {"description": "Fixture not found in the catalogue."}},
)
async def get_fixture(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MatchFixtureOut:
    row = await match_crud.get_match(db, match_id)
    if row is None:
        raise DataNotFoundError(f"Match {match_id} not found.")
    return MatchFixtureOut.model_validate(row)


@router.get(
    "/fixtures/{match_id}/events",
    response_model=list[MatchEventOut],
    summary="Match event timeline (pro_analyst+)",
    description=(
        "Full chronological ``MatchEvent`` list for a fixture: goals, cards, "
        "subs, VAR. Event-level detail feeds the Phase-5 prediction engine "
        "and is therefore gated to pro_analyst+."
    ),
    dependencies=[Depends(require_role(UserRole.pro_analyst))],
)
async def get_fixture_events(
    match_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MatchEventOut]:
    if (await match_crud.get_match(db, match_id)) is None:
        raise DataNotFoundError(f"Match {match_id} not found.")
    rows = await match_crud.list_match_events(db, match_id)
    return [MatchEventOut.model_validate(e) for e in rows]


# ── Pipeline / VAEP analytics (existing) ─────────────────────────────────────

@router.get(
    "/{match_id}/status",
    summary="Processing artifact status for an uploaded or processed match",
)
async def get_match_status(match_id: str) -> dict[str, object]:
    return await run_in_threadpool(_artifact_status, match_id)


@router.get(
    "/{match_id}/overview",
    summary="Frontend-friendly match overview",
    responses={404: {"description": "Match not found — run the pipeline first."}},
)
async def get_match_overview(
    match_id: str,
    service: MatchService = Depends(_match_service),
) -> dict[str, object]:
    try:
        detail = await run_in_threadpool(service.get_match_detail, match_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    timeline: list[dict[str, float | int]] = []
    try:
        vaep_df = await run_in_threadpool(service._load_vaep, match_id)
        if {"minute", "team_id", "xg"}.issubset(set(vaep_df.columns)):
            home = str(detail.home_team_id)
            away = str(detail.away_team_id)
            shot_df = vaep_df[vaep_df["xg"].fillna(0) > 0].copy()
            if not shot_df.empty:
                shot_df["team_id"] = shot_df["team_id"].astype(str)
                cumulative_home = 0.0
                cumulative_away = 0.0
                for _, row in shot_df.sort_values("minute").iterrows():
                    xg = float(row.get("xg", 0.0) or 0.0)
                    if str(row.get("team_id")) == home:
                        cumulative_home += xg
                    elif str(row.get("team_id")) == away:
                        cumulative_away += xg
                    timeline.append(
                        {
                            "minute": int(row.get("minute", 0) or 0),
                            "xg_home": round(cumulative_home, 4),
                            "xg_away": round(cumulative_away, 4),
                        }
                    )
    except Exception:
        timeline = []

    return {
        "match_id": detail.match_id,
        "home_team": detail.home_team_name,
        "away_team": detail.away_team_name,
        "xg_home": detail.xg_home,
        "xg_away": detail.xg_away,
        "possession_home": 0,
        "possession_away": 0,
        "total_actions": detail.total_actions,
        "summary": (
            f"{detail.home_team_name} generated {detail.xg_home:.2f} xG "
            f"against {detail.away_team_name}'s {detail.xg_away:.2f} xG."
        ),
        "timeline": timeline,
    }


@router.get(
    "/{match_id}",
    response_model=MatchDetailResponse,
    summary="Full match analysis",
    description=(
        "Returns xG, VAEP totals, team stats, formations, and the "
        "most valuable actions for a single match."
    ),
    responses={
        404: {"description": "Match not found — run the pipeline first."},
    },
)
async def get_match(
    match_id: str,
    service: MatchService = Depends(_match_service),
) -> MatchDetailResponse:
    try:
        return service.get_match_detail(match_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{match_id}/report",
    response_model=MatchReportResponse,
    summary="Tactical report JSON",
    description=(
        "Returns the full tactical intelligence JSON report generated "
        "by Phase 7. Includes formations, weaknesses, and player rankings."
    ),
    responses={
        404: {"description": "Tactical report not found — run Phase 7 first."},
    },
)
async def get_match_report(
    match_id: str,
    service: MatchService = Depends(_match_service),
) -> MatchReportResponse:
    try:
        report = service.get_match_report(match_id)
        return MatchReportResponse(match_id=match_id, report=report)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
