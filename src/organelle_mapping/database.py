"""Database utilities for storing evaluation results.

Uses SQLAlchemy Core for backend-agnostic database access.
Supports SQLite (default) and PostgreSQL.
"""

import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

from sqlalchemy import (
    Column,
    Engine,
    Float,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    case,
    create_engine,
    desc,
    event,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# SQLite robustness defaults. busy_timeout is how long sqlite itself will
# spin-wait on a locked DB before raising; the retry layer covers cases
# where contention exceeds that window (e.g. a writer that holds the lock
# for tens of seconds).
DEFAULT_SQLITE_TIMEOUT = 30.0
WRITE_RETRY_ATTEMPTS = 8
WRITE_RETRY_BASE_DELAY = 0.1
WRITE_RETRY_MAX_DELAY = 5.0

T = TypeVar("T")


def _is_locked_error(exc: BaseException) -> bool:
    msg = str(getattr(exc, "orig", None) or exc).lower()
    return "database is locked" in msg or "database table is locked" in msg


def _retry_on_lock(func: Callable[[], T]) -> T:
    """Run a write callable, retrying with backoff on SQLite lock errors."""
    delay = WRITE_RETRY_BASE_DELAY
    for attempt in range(1, WRITE_RETRY_ATTEMPTS + 1):
        try:
            return func()
        except OperationalError as exc:
            if not _is_locked_error(exc) or attempt == WRITE_RETRY_ATTEMPTS:
                raise
            sleep_for = min(delay + random.uniform(0, delay), WRITE_RETRY_MAX_DELAY)
            logger.warning(
                "Database locked; retrying in %.2fs (attempt %d/%d)",
                sleep_for,
                attempt,
                WRITE_RETRY_ATTEMPTS,
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, WRITE_RETRY_MAX_DELAY)
    # Unreachable: loop either returns or raises.
    raise RuntimeError("unreachable")

metadata = MetaData()

results_table = Table(
    "results",
    metadata,
    Column("run", String, nullable=False),
    Column("checkpoint", String, nullable=False),
    Column("dataset", String, nullable=False),
    Column("crop", String, nullable=False),
    Column("channel", String, nullable=False),
    Column("label", String, nullable=False),
    Column("postprocessing_type", String, nullable=False),
    Column("threshold", Float, nullable=True),
    Column("metric", String, nullable=False),
    Column("score", Float, nullable=False),
    UniqueConstraint(
        "run",
        "checkpoint",
        "dataset",
        "crop",
        "channel",
        "postprocessing_type",
        "threshold",
        "metric",
        name="uq_result",
    ),
)

UNIQUE_COLUMNS = ["run", "checkpoint", "dataset", "crop", "channel", "postprocessing_type", "threshold", "metric"]

crops_table = Table(
    "crops",
    metadata,
    Column("dataset", String, nullable=False),
    Column("crop", String, nullable=False),
    Column("label", String, nullable=False),
    Column("scale_level", String, nullable=False),
    Column("resolution_z", Float, nullable=False),
    Column("resolution_y", Float, nullable=False),
    Column("resolution_x", Float, nullable=False),
    Column("total_voxels", Float, nullable=False),
    Column("present", Float, nullable=False),
    Column("absent", Float, nullable=False),
    Column("unknown", Float, nullable=False),
    Column("crop_group", String, nullable=True),
    UniqueConstraint("dataset", "crop", "label", "scale_level", name="uq_crop"),
)

CROPS_UNIQUE_COLUMNS = ["dataset", "crop", "label", "scale_level"]


def _where_clause(values: dict):
    """Build WHERE clause handling NULLs correctly (IS NULL instead of = NULL)."""
    conditions = []
    for col in UNIQUE_COLUMNS:
        if values[col] is None:
            conditions.append(results_table.c[col].is_(None))
        else:
            conditions.append(results_table.c[col] == values[col])
    return conditions


def init_database(db_url: str, timeout: Optional[float] = None, *, read_only: bool = False) -> Engine:
    """Initialize database with schema and return an engine.

    Args:
        db_url: SQLAlchemy database URL.
            SQLite: "sqlite:///path/to/db.sqlite"
            PostgreSQL: "postgresql://user:pass@host/dbname"
        timeout: SQLite-only. Seconds to wait for a busy lock before raising
            "database is locked". Defaults to ``DEFAULT_SQLITE_TIMEOUT`` so
            that concurrent writers wait rather than fail outright.
        read_only: SQLite-only. Open the file read-only at the OS level
            (``mode=ro`` URI), and skip schema creation and the WAL pragma.
            Use when querying a snapshot — especially on NFS — where any
            accidental write risks corruption.
    """
    is_sqlite = db_url.startswith("sqlite:")

    if is_sqlite and read_only:
        if not db_url.startswith("sqlite:///"):
            raise ValueError(f"read_only=True requires a sqlite:/// URL, got {db_url}")
        path = db_url[len("sqlite:///") :]
        # SQLite URI mode with mode=ro opens the file O_RDONLY at the OS
        # level — any write attempt errors instead of corrupting.
        db_url = f"sqlite:///file:{path}?mode=ro&uri=true"
    elif db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict = {}
    if is_sqlite:
        connect_args["timeout"] = timeout if timeout is not None else DEFAULT_SQLITE_TIMEOUT

    engine = create_engine(db_url, connect_args=connect_args)

    if is_sqlite and not read_only:
        # WAL gives concurrent reads alongside a single writer and shorter
        # write-lock holds. It requires a real local filesystem — never put
        # this DB on NFS. journal_mode=WAL persists on the file; the PRAGMA
        # is set on every connect so it is self-healing if someone toggles
        # it off elsewhere.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
            finally:
                cursor.close()

    if not read_only:
        metadata.create_all(engine)
    logger.info(f"Database initialized at {db_url}")
    return engine


def vacuum_into(engine: Engine, dest_path: str) -> None:
    """Copy the SQLite DB to ``dest_path`` via ``VACUUM INTO``.

    Produces a self-contained, defragmented copy in default rollback-journal
    mode regardless of the source's journal mode. Safe to run while writers
    are active on the source. The destination must not already exist.
    """
    if engine.dialect.name != "sqlite":
        raise ValueError(f"vacuum_into only supports SQLite (got {engine.dialect.name})")
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        conn.execute(text("VACUUM INTO :dest"), {"dest": dest_path})


def insert_result(
    engine: Engine,
    run: str,
    checkpoint: str,
    dataset: str,
    crop: str,
    channel: str,
    label: str,
    metric: str,
    score: float,
    postprocessing_type: str = "threshold",
    threshold: Optional[float] = None,
) -> None:
    """Insert or update a single evaluation result."""
    values = {
        "run": run,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "crop": crop,
        "channel": channel,
        "label": label,
        "postprocessing_type": postprocessing_type,
        "threshold": threshold,
        "metric": metric,
        "score": score,
    }

    def _write() -> None:
        with engine.begin() as conn:
            existing = conn.execute(select(results_table).where(*_where_clause(values))).first()

            if existing:
                conn.execute(results_table.update().where(*_where_clause(values)).values(score=score))
            else:
                conn.execute(insert(results_table).values(**values))

    _retry_on_lock(_write)


def insert_crop(
    engine: Engine,
    dataset: str,
    crop: str,
    label: str,
    scale_level: str,
    resolution_z: float,
    resolution_y: float,
    resolution_x: float,
    total_voxels: float,
    present: float,
    absent: float,
    unknown: float,
    crop_group: Optional[str] = None,
) -> None:
    """Insert or update a crop ground truth voxel count."""
    values = {
        "dataset": dataset,
        "crop": crop,
        "label": label,
        "scale_level": scale_level,
        "resolution_z": resolution_z,
        "resolution_y": resolution_y,
        "resolution_x": resolution_x,
        "total_voxels": total_voxels,
        "present": present,
        "absent": absent,
        "unknown": unknown,
        "crop_group": crop_group,
    }

    unique_conditions = [crops_table.c[col] == values[col] for col in CROPS_UNIQUE_COLUMNS]

    def _write() -> None:
        with engine.begin() as conn:
            existing = conn.execute(select(crops_table).where(*unique_conditions)).first()

            if existing:
                conn.execute(
                    crops_table.update()
                    .where(*unique_conditions)
                    .values(
                        resolution_z=resolution_z,
                        resolution_y=resolution_y,
                        resolution_x=resolution_x,
                        total_voxels=total_voxels,
                        present=present,
                        absent=absent,
                        unknown=unknown,
                        crop_group=crop_group,
                    )
                )
            else:
                conn.execute(insert(crops_table).values(**values))

    _retry_on_lock(_write)



def _filter_conditions(filters: dict) -> list:
    """Build WHERE conditions from a filter dict, skipping None values."""
    conditions = []
    for col_name, value in filters.items():
        if value is not None:
            conditions.append(results_table.c[col_name] == value)
    return conditions


def _nonempty_crops_cte(min_present: int = 100):
    """CTE projecting (dataset, crop, label, crop_group, evaluable) for valid crops.

    - `present > min_present` filter (defaults to 100 — drops all-negative crops).
    - `crop_group` falls back to the crop name itself when NULL (singleton group).
    - `evaluable = total_voxels - unknown` at s0; the canonical voxel-weighting weight.
    """
    s0_evaluable = case(
        (crops_table.c.scale_level == "s0", crops_table.c.total_voxels - crops_table.c.unknown)
    )
    return (
        select(
            crops_table.c.dataset,
            crops_table.c.crop,
            crops_table.c.label,
            func.coalesce(func.max(crops_table.c.crop_group), crops_table.c.crop).label("crop_group"),
            func.max(s0_evaluable).label("evaluable"),
        )
        .group_by(crops_table.c.dataset, crops_table.c.crop, crops_table.c.label)
        .having(func.max(crops_table.c.present) > min_present)
        .cte("nonempty_crops")
    )


def query_results(
    engine: Engine,
    filters: Optional[dict] = None,
    limit: Optional[int] = None,
    order_by: Optional[str] = "score",
    sort_direction: str = "desc",
) -> list[dict]:
    """Query results with optional filters.

    Args:
        engine: SQLAlchemy engine.
        filters: Dict of column_name -> value to filter on. None values are ignored.
        limit: Maximum number of rows to return. None for no limit.
        order_by: Column name to sort by. None for no ordering.
        sort_direction: "asc" for ascending, "desc" for descending.
    """
    stmt = select(results_table)
    if filters:
        conditions = _filter_conditions(filters)
        if conditions:
            stmt = stmt.where(*conditions)

    if order_by is not None:
        col = results_table.c[order_by]
        stmt = stmt.order_by(col if sort_direction == "asc" else desc(col))
    if limit is not None:
        stmt = stmt.limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(row._mapping) for row in rows]


def query_best_per_label(
    engine: Engine,
    metric: str = "dice",
    filters: Optional[dict] = None,
    min_present: int = 100,
) -> list[dict]:
    """Find the best score per label for a given metric.

    Filters out all-negative (dataset, crop, label) triples via the standard
    `present > min_present` join. Returns the single highest-scoring row per
    label — no aggregation, so voxel-weighting and crop_group don't apply here.
    """
    nonempty_crops = _nonempty_crops_cte(min_present)
    conditions = [results_table.c.metric == metric]
    if filters:
        conditions.extend(_filter_conditions(filters))

    join_on = (
        (results_table.c.dataset == nonempty_crops.c.dataset)
        & (results_table.c.crop == nonempty_crops.c.crop)
        & (results_table.c.label == nonempty_crops.c.label)
    )

    # Subquery: max score per label, restricted to valid crops.
    max_subq = (
        select(results_table.c.label, func.max(results_table.c.score).label("best_score"))
        .select_from(results_table.join(nonempty_crops, join_on))
        .where(*conditions)
        .group_by(results_table.c.label)
        .subquery()
    )

    # Join back to get full row details for the best score.
    stmt = (
        select(results_table)
        .select_from(
            results_table.join(nonempty_crops, join_on).join(
                max_subq,
                (results_table.c.label == max_subq.c.label) & (results_table.c.score == max_subq.c.best_score),
            )
        )
        .where(*conditions)
        .order_by(results_table.c.label)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(row._mapping) for row in rows]


def query_distinct_values(
    engine: Engine,
    column: str,
    filters: Optional[dict] = None,
) -> list[str]:
    """Get distinct values for a given column, optionally filtered."""
    col = results_table.c[column]
    stmt = select(col).distinct().order_by(col)

    if filters:
        conditions = _filter_conditions(filters)
        if conditions:
            stmt = stmt.where(*conditions)

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [row[0] for row in rows]


def query_checkpoint_comparison(
    engine: Engine,
    metric: str = "dice",
    filters: Optional[dict] = None,
    min_present: int = 100,
) -> list[dict]:
    """Compare checkpoints: voxel-weighted score per label per checkpoint.

    Applies the standard conventions: `present > min_present` filter, two-stage
    `crop_group` aggregation, and weighting by `evaluable` voxels (s0
    `total_voxels - unknown`). For each `(checkpoint, label, dataset, crop)`
    the best score across thresholds/postprocessing is taken first, then those
    are averaged across crop groups weighted by their evaluable voxel count.

    Returns list of dicts with keys: label, checkpoint, avg_score, num_groups.
    """
    nonempty_crops = _nonempty_crops_cte(min_present)
    conditions = [results_table.c.metric == metric]
    if filters:
        conditions.extend(_filter_conditions(filters))

    join_on = (
        (results_table.c.dataset == nonempty_crops.c.dataset)
        & (results_table.c.crop == nonempty_crops.c.crop)
        & (results_table.c.label == nonempty_crops.c.label)
    )

    # Stage 1: best score per (checkpoint, label, dataset, crop) — peak across thresholds/postproc.
    best_per_crop = (
        select(
            results_table.c.label,
            results_table.c.checkpoint,
            results_table.c.dataset,
            results_table.c.crop,
            nonempty_crops.c.crop_group,
            nonempty_crops.c.evaluable,
            func.max(results_table.c.score).label("best_score"),
        )
        .select_from(results_table.join(nonempty_crops, join_on))
        .where(*conditions)
        .group_by(
            results_table.c.label,
            results_table.c.checkpoint,
            results_table.c.dataset,
            results_table.c.crop,
            nonempty_crops.c.crop_group,
            nonempty_crops.c.evaluable,
        )
        .subquery()
    )

    # Stage 2: voxel-weighted score per crop_group within (label, checkpoint, dataset).
    per_group = (
        select(
            best_per_crop.c.label,
            best_per_crop.c.checkpoint,
            best_per_crop.c.dataset,
            best_per_crop.c.crop_group,
            func.sum(best_per_crop.c.best_score * best_per_crop.c.evaluable).label("sw"),
            func.sum(best_per_crop.c.evaluable).label("w"),
        )
        .group_by(
            best_per_crop.c.label,
            best_per_crop.c.checkpoint,
            best_per_crop.c.dataset,
            best_per_crop.c.crop_group,
        )
        .subquery()
    )

    # Stage 3: voxel-weighted macro per (label, checkpoint), counting groups.
    stmt = (
        select(
            per_group.c.label,
            per_group.c.checkpoint,
            (func.sum(per_group.c.sw) / func.sum(per_group.c.w)).label("avg_score"),
            func.count().label("num_groups"),
        )
        .group_by(per_group.c.label, per_group.c.checkpoint)
        .order_by(per_group.c.label, per_group.c.checkpoint)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(row._mapping) for row in rows]
