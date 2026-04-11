"""Database utilities for storing evaluation results.

Uses SQLAlchemy Core for backend-agnostic database access.
Supports SQLite (default) and PostgreSQL.
"""

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    Engine,
    Float,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    desc,
    func,
    insert,
    select,
)

logger = logging.getLogger(__name__)

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


def init_database(db_url: str) -> Engine:
    """Initialize database with schema and return an engine.

    Args:
        db_url: SQLAlchemy database URL.
            SQLite: "sqlite:///path/to/db.sqlite"
            PostgreSQL: "postgresql://user:pass@host/dbname"
    """
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(db_url)
    metadata.create_all(engine)
    logger.info(f"Database initialized at {db_url}")
    return engine


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

    with engine.begin() as conn:
        existing = conn.execute(select(results_table).where(*_where_clause(values))).first()

        if existing:
            conn.execute(results_table.update().where(*_where_clause(values)).values(score=score))
        else:
            conn.execute(insert(results_table).values(**values))


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
    }

    unique_conditions = [crops_table.c[col] == values[col] for col in CROPS_UNIQUE_COLUMNS]

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
                )
            )
        else:
            conn.execute(insert(crops_table).values(**values))



def _filter_conditions(filters: dict) -> list:
    """Build WHERE conditions from a filter dict, skipping None values."""
    conditions = []
    for col_name, value in filters.items():
        if value is not None:
            conditions.append(results_table.c[col_name] == value)
    return conditions


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
) -> list[dict]:
    """Find the best score per label for a given metric.

    Returns rows with: label, best score, and the checkpoint/threshold/dataset/crop that achieved it.
    """
    # Subquery: max score per label
    conditions = [results_table.c.metric == metric]
    if filters:
        conditions.extend(_filter_conditions(filters))

    max_subq = (
        select(results_table.c.label, func.max(results_table.c.score).label("best_score"))
        .where(*conditions)
        .group_by(results_table.c.label)
        .subquery()
    )

    # Join back to get full row details for the best score
    stmt = (
        select(results_table)
        .join(
            max_subq,
            (results_table.c.label == max_subq.c.label) & (results_table.c.score == max_subq.c.best_score),
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
) -> list[dict]:
    """Compare checkpoints: average best-per-crop score per label per checkpoint.

    For each (checkpoint, label, dataset, crop) combination, finds the best score
    across all thresholds/postprocessing, then averages those best scores.

    Returns list of dicts with keys: label, checkpoint, avg_score, num_crops.
    """
    conditions = [results_table.c.metric == metric]
    if filters:
        conditions.extend(_filter_conditions(filters))

    # Subquery: best score per (checkpoint, label, dataset, crop)
    best_per_crop = (
        select(
            results_table.c.label,
            results_table.c.checkpoint,
            results_table.c.dataset,
            results_table.c.crop,
            func.max(results_table.c.score).label("best_score"),
        )
        .where(*conditions)
        .group_by(
            results_table.c.label,
            results_table.c.checkpoint,
            results_table.c.dataset,
            results_table.c.crop,
        )
        .subquery()
    )

    # Average the best-per-crop scores
    stmt = (
        select(
            best_per_crop.c.label,
            best_per_crop.c.checkpoint,
            func.avg(best_per_crop.c.best_score).label("avg_score"),
            func.count().label("num_crops"),
        )
        .group_by(best_per_crop.c.label, best_per_crop.c.checkpoint)
        .order_by(best_per_crop.c.label, best_per_crop.c.checkpoint)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(row._mapping) for row in rows]
