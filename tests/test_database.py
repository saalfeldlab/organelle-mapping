"""Tests for database schema and insert/upsert functionality."""

import tempfile
from pathlib import Path

from sqlalchemy import inspect, select

from organelle_mapping.database import init_database, insert_result, results_table


def test_database_creation():
    """Test database initialization and schema creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)

        assert Path(f"{tmpdir}/test.db").exists()

        inspector = inspect(engine)
        assert "results" in inspector.get_table_names()

        columns = inspector.get_columns("results")
        column_names = [col["name"] for col in columns]
        expected_columns = [
            "run", "checkpoint", "dataset", "crop", "channel", "label",
            "postprocessing_type", "threshold", "metric", "score",
        ]
        assert column_names == expected_columns

        # Verify threshold is nullable
        threshold_col = next(c for c in columns if c["name"] == "threshold")
        assert threshold_col["nullable"] is True

        # Verify postprocessing_type is not nullable
        pp_col = next(c for c in columns if c["name"] == "postprocessing_type")
        assert pp_col["nullable"] is False


def test_insert_result():
    """Test inserting evaluation results with threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.85,
            threshold=0.5,
        )

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0] == (
                "run01",
                "checkpoint_1000",
                "jrc_hela-2",
                "crop1",
                "mito_binary",
                "mito",
                "threshold",
                0.5,
                "dice",
                0.85,
            )


def test_insert_result_without_threshold():
    """Test inserting evaluation results without threshold (NULL)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(
            engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1",
            "mito_binary", "mito", "dice", 0.85,
        )

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0] == (
                "run01",
                "checkpoint_1000",
                "jrc_hela-2",
                "crop1",
                "mito_binary",
                "mito",
                "threshold",
                None,
                "dice",
                0.85,
            )


def test_insert_overwrite():
    """Test that upsert overwrites existing results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.85,
            threshold=0.5,
        )
        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.90,
            threshold=0.5,
        )

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0][9] == 0.90  # score should be updated


def test_insert_overwrite_null_threshold():
    """Test that upsert works correctly with NULL threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.85,
        )
        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.90,
        )

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0][9] == 0.90


def test_multiple_results():
    """Test inserting multiple different results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.85,
            threshold=0.5,
        )
        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "jaccard",
            0.74,
            threshold=0.5,
        )
        insert_result(
            engine,
            "run01",
            "checkpoint_1000",
            "jrc_hela-2",
            "crop1",
            "er_binary",
            "er",
            "dice",
            0.72,
            threshold=0.3,
        )
        insert_result(
            engine,
            "run01",
            "checkpoint_2000",
            "jrc_hela-2",
            "crop1",
            "mito_binary",
            "mito",
            "dice",
            0.88,
            threshold=0.5,
        )

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 4


def test_different_thresholds_same_label():
    """Test that results at different thresholds are stored separately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.70, threshold=0.3)
        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.75, threshold=0.7)

        with engine.connect() as conn:
            rows = conn.execute(
                select(results_table)
                .where(results_table.c.channel == "mito_binary")
                .order_by(results_table.c.threshold)
            ).fetchall()
            assert len(rows) == 3
            assert [r[7] for r in rows] == [0.3, 0.5, 0.7]  # thresholds (index 7)
            assert [r[9] for r in rows] == [0.70, 0.85, 0.75]  # scores (index 9)
