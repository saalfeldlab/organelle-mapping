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

        assert Path(f"{tmpdir}/test.db").exists()

        inspector = inspect(engine)
        assert "results" in inspector.get_table_names()

        columns = inspector.get_columns("results")
        column_names = [col["name"] for col in columns]
        expected_columns = ["run", "checkpoint", "dataset", "crop", "label", "threshold", "metric", "score"]
        assert column_names == expected_columns

        # Verify threshold is nullable
        threshold_col = next(c for c in columns if c["name"] == "threshold")
        assert threshold_col["nullable"] is True


def test_insert_result():
    """Test inserting evaluation results with threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.85, threshold=0.5)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0] == ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)


def test_insert_result_without_threshold():
    """Test inserting evaluation results without threshold (NULL)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.85)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0] == ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", None, "dice", 0.85)


def test_insert_overwrite():
    """Test that upsert overwrites existing results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.90, threshold=0.5)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0][7] == 0.90  # score should be updated


def test_insert_overwrite_null_threshold():
    """Test that upsert works correctly with NULL threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.85)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.90)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 1
            assert rows[0][7] == 0.90


def test_multiple_results():
    """Test inserting multiple different results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "jaccard", 0.74, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "er", "dice", 0.72, threshold=0.3)
        insert_result(engine, "run01", "checkpoint_2000", "jrc_hela-2", "crop1", "mito", "dice", 0.88, threshold=0.5)

        with engine.connect() as conn:
            rows = conn.execute(select(results_table)).fetchall()
            assert len(rows) == 4


def test_different_thresholds_same_label():
    """Test that results at different thresholds are stored separately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito", "dice", 0.70, threshold=0.3)
        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito", "dice", 0.75, threshold=0.7)

        with engine.connect() as conn:
            rows = conn.execute(
                select(results_table)
                .where(results_table.c.label == "mito")
                .order_by(results_table.c.threshold)
            ).fetchall()
            assert len(rows) == 3
            assert [r[5] for r in rows] == [0.3, 0.5, 0.7]  # thresholds
            assert [r[7] for r in rows] == [0.70, 0.85, 0.75]  # scores


def test_query_best_checkpoint_per_label():
    """Test querying for the best checkpoint per label."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.80, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_2000", "jrc_hela-2", "crop1", "mito", "dice", 0.90, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_3000", "jrc_hela-2", "crop1", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "er", "dice", 0.70, threshold=0.3)
        insert_result(engine, "run01", "checkpoint_2000", "jrc_hela-2", "crop1", "er", "dice", 0.75, threshold=0.3)

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
        engine = init_database(f"sqlite:///{tmpdir}/test.db")

        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "checkpoint_1000", "jrc_hela-3", "crop5", "mito", "dice", 0.78, threshold=0.5)

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
