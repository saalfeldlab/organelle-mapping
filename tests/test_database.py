"""Tests for database functionality."""

import tempfile
from pathlib import Path

from sqlalchemy import func, inspect, select

from organelle_mapping.database import init_database, insert_result, results_table


def test_database_creation():
    """Test database initialization and schema creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        # Check that file was created
        assert Path(f"{tmpdir}/test.db").exists()

        # Check schema
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "results" in tables

        columns = inspector.get_columns("results")
        column_names = [col["name"] for col in columns]
        expected_columns = ["run", "checkpoint", "dataset", "crop", "label", "threshold", "metric", "score"]
        assert column_names == expected_columns


def test_insert_result():
    """Test inserting evaluation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0] == ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)


def test_insert_overwrite():
    """Test that upsert overwrites existing results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.90)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0][7] == 0.90  # score should be updated


def test_multiple_results():
    """Test inserting multiple different results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        test_data = [
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85),
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "jaccard", 0.74),
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "er", 0.3, "dice", 0.72),
            ("run01", "checkpoint_2000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.88),
        ]

        for data in test_data:
            insert_result(engine, *data)

        with engine.connect() as conn:
            rows = conn.execute(
                select(results_table).order_by(
                    results_table.c.checkpoint, results_table.c.label, results_table.c.metric
                )
            ).fetchall()
            assert len(rows) == 4
            assert set(rows) == set(test_data)


def test_query_best_checkpoint_per_label():
    """Test querying for the best checkpoint per label."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        test_data = [
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.80),
            ("run01", "checkpoint_2000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.90),
            ("run01", "checkpoint_3000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85),
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "er", 0.3, "dice", 0.70),
            ("run01", "checkpoint_2000", "jrc_hela-2", "crop1", "er", 0.3, "dice", 0.75),
        ]
        for data in test_data:
            insert_result(engine, *data)

        with engine.connect() as conn:
            stmt = (
                select(
                    results_table.c.label,
                    results_table.c.checkpoint,
                    func.max(results_table.c.score).label("best_score"),
                )
                .where(results_table.c.metric == "dice")
                .group_by(results_table.c.label)
                .order_by(results_table.c.label)
            )
            best = conn.execute(stmt).fetchall()

            assert len(best) == 2
            assert best[0] == ("er", "checkpoint_2000", 0.75)
            assert best[1] == ("mito", "checkpoint_2000", 0.90)


def test_query_across_datasets():
    """Test querying results across different datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        test_data = [
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85),
            ("run01", "checkpoint_1000", "jrc_hela-3", "crop5", "mito", 0.5, "dice", 0.78),
        ]
        for data in test_data:
            insert_result(engine, *data)

        with engine.connect() as conn:
            stmt = (
                select(
                    results_table.c.dataset,
                    func.avg(results_table.c.score).label("avg_score"),
                )
                .where(results_table.c.metric == "dice")
                .group_by(results_table.c.dataset)
                .order_by(results_table.c.dataset)
            )
            per_dataset = conn.execute(stmt).fetchall()

            assert len(per_dataset) == 2
            assert per_dataset[0][0] == "jrc_hela-2"
            assert per_dataset[0][1] == 0.85
            assert per_dataset[1][0] == "jrc_hela-3"
            assert per_dataset[1][1] == 0.78
