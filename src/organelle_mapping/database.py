"""Database utilities for storing evaluation results.

Uses SQLAlchemy Core for backend-agnostic database access.
Supports SQLite (default) and PostgreSQL.
"""

import logging
from pathlib import Path

from sqlalchemy import (
    Column,
    Engine,
    Float,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
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
    Column("threshold", Float, nullable=False),
    Column("metric", String, nullable=False),
    Column("score", Float, nullable=False),
    UniqueConstraint("run", "checkpoint", "dataset", "crop", "label", "threshold", "metric", name="uq_result"),
)

UNIQUE_COLUMNS = ["run", "checkpoint", "dataset", "crop", "label", "threshold", "metric"]


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
    threshold: float,
    metric: str,
    score: float,
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
        # Check if row exists
        stmt = select(results_table).where(
            *[results_table.c[col] == values[col] for col in UNIQUE_COLUMNS]
        )
        existing = conn.execute(stmt).first()

        if existing:
            # Update score
            update_stmt = (
                results_table.update()
                .where(*[results_table.c[col] == values[col] for col in UNIQUE_COLUMNS])
                .values(score=score)
            )
            conn.execute(update_stmt)
        else:
            conn.execute(insert(results_table).values(**values))
