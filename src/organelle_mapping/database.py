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
    Column("label", String, nullable=False),
    Column("threshold", Float, nullable=True),
    Column("metric", String, nullable=False),
    Column("score", Float, nullable=False),
    UniqueConstraint("run", "checkpoint", "dataset", "crop", "label", "threshold", "metric", name="uq_result"),
)

UNIQUE_COLUMNS = ["run", "checkpoint", "dataset", "crop", "label", "threshold", "metric"]


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
    label: str,
    metric: str,
    score: float,
    threshold: Optional[float] = None,
) -> None:
    """Insert or update a single evaluation result."""
    values = {
        "run": run,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "crop": crop,
        "label": label,
        "threshold": threshold,
        "metric": metric,
        "score": score,
    }

    with engine.begin() as conn:
        existing = conn.execute(
            select(results_table).where(*_where_clause(values))
        ).first()

        if existing:
            conn.execute(
                results_table.update()
                .where(*_where_clause(values))
                .values(score=score)
            )
        else:
            conn.execute(insert(results_table).values(**values))


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
    limit: int = 100,
    order_by: str = "score",
    sort_direction: str = "desc",
) -> list[dict]:
    """Query results with optional filters.

    Args:
        engine: SQLAlchemy engine.
        filters: Dict of column_name -> value to filter on. None values are ignored.
        limit: Maximum number of rows to return.
        order_by: Column name to sort by.
        sort_direction: "asc" for ascending, "desc" for descending.
    """
    stmt = select(results_table)
    if filters:
        conditions = _filter_conditions(filters)
        if conditions:
            stmt = stmt.where(*conditions)

    col = results_table.c[order_by]
    stmt = stmt.order_by(col if sort_direction == "asc" else desc(col)).limit(limit)

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
    """Compare checkpoints: average score per label per checkpoint.

    Returns list of dicts with keys: label, checkpoint, avg_score, num_crops.
    """
    conditions = [results_table.c.metric == metric]
    if filters:
        conditions.extend(_filter_conditions(filters))

    stmt = (
        select(
            results_table.c.label,
            results_table.c.checkpoint,
            func.avg(results_table.c.score).label("avg_score"),
            func.count().label("num_crops"),
        )
        .where(*conditions)
        .group_by(results_table.c.label, results_table.c.checkpoint)
        .order_by(results_table.c.label, results_table.c.checkpoint)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [dict(row._mapping) for row in rows]
