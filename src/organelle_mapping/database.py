"""Database utilities for storing evaluation results."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def create_database_schema(db_path: str) -> None:
    """Create database table for evaluation results."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                run TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                dataset TEXT NOT NULL,
                crop TEXT NOT NULL,
                label TEXT NOT NULL,
                threshold REAL NOT NULL,
                metric TEXT NOT NULL,
                score REAL NOT NULL,
                UNIQUE(run, checkpoint, dataset, crop, label, threshold, metric)
            )
        """)

        conn.commit()
        logger.info(f"Database schema created/verified at {db_path}")


def init_database(db_path: str) -> None:
    """Initialize database with schema."""
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    create_database_schema(db_path)


def insert_result(
    db_path: str,
    run: str,
    checkpoint: str,
    dataset: str,
    crop: str,
    label: str,
    threshold: float,
    metric: str,
    score: float,
) -> None:
    """Insert or replace a single evaluation result."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO results
            (run, checkpoint, dataset, crop, label, threshold, metric, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (run, checkpoint, dataset, crop, label, threshold, metric, score),
        )
        conn.commit()
