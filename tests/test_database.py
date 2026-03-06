"""Tests for database functionality."""

import sqlite3
import tempfile
from pathlib import Path

from organelle_mapping.database import init_database, insert_result


def test_database_creation():
    """Test database initialization and schema creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        # Initialize database
        init_database(db_path)

        # Check that file was created
        assert Path(db_path).exists()

        # Check schema
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            assert ("results",) in tables

            # Check columns
            columns = cursor.execute("PRAGMA table_info(results)").fetchall()
            column_names = [col[1] for col in columns]
            expected_columns = ["run", "checkpoint", "dataset", "crop", "label", "threshold", "metric", "score"]
            assert column_names == expected_columns


def test_insert_result():
    """Test inserting evaluation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        init_database(db_path)

        # Insert a result
        insert_result(db_path, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)

        # Check it was inserted
        with sqlite3.connect(db_path) as conn:
            results = conn.execute("SELECT * FROM results").fetchall()
            assert len(results) == 1
            assert results[0] == ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)


def test_insert_overwrite():
    """Test that INSERT OR REPLACE overwrites existing results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        init_database(db_path)

        # Insert initial result
        insert_result(db_path, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85)

        # Insert same key with different score
        insert_result(db_path, "run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.90)

        # Should only have one result with updated score
        with sqlite3.connect(db_path) as conn:
            results = conn.execute("SELECT * FROM results").fetchall()
            assert len(results) == 1
            assert results[0][7] == 0.90  # score should be updated


def test_multiple_results():
    """Test inserting multiple different results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        init_database(db_path)

        # Insert multiple results
        test_data = [
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85),
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "jaccard", 0.74),
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "er", 0.3, "dice", 0.72),
            ("run01", "checkpoint_2000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.88),
        ]

        for data in test_data:
            insert_result(db_path, *data)

        # Check all were inserted
        with sqlite3.connect(db_path) as conn:
            results = conn.execute("SELECT * FROM results ORDER BY checkpoint, label, metric").fetchall()
            assert len(results) == 4
            assert set(results) == set(test_data)


def test_query_best_checkpoint_per_label():
    """Test querying for the best checkpoint per label."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        init_database(db_path)

        # Insert results across checkpoints
        test_data = [
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.80),
            ("run01", "checkpoint_2000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.90),
            ("run01", "checkpoint_3000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85),
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "er", 0.3, "dice", 0.70),
            ("run01", "checkpoint_2000", "jrc_hela-2", "crop1", "er", 0.3, "dice", 0.75),
        ]
        for data in test_data:
            insert_result(db_path, *data)

        # Query best checkpoint per label
        with sqlite3.connect(db_path) as conn:
            best = conn.execute("""
                SELECT label, checkpoint, MAX(score) as best_score
                FROM results
                WHERE metric = 'dice'
                GROUP BY label
                ORDER BY label
            """).fetchall()

            assert len(best) == 2
            assert best[0] == ("er", "checkpoint_2000", 0.75)
            assert best[1] == ("mito", "checkpoint_2000", 0.90)


def test_query_across_datasets():
    """Test querying results across different datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        init_database(db_path)

        # Insert results from different datasets
        test_data = [
            ("run01", "checkpoint_1000", "jrc_hela-2", "crop1", "mito", 0.5, "dice", 0.85),
            ("run01", "checkpoint_1000", "jrc_hela-3", "crop5", "mito", 0.5, "dice", 0.78),
        ]
        for data in test_data:
            insert_result(db_path, *data)

        # Query average score per dataset
        with sqlite3.connect(db_path) as conn:
            per_dataset = conn.execute("""
                SELECT dataset, AVG(score) as avg_score
                FROM results
                WHERE metric = 'dice'
                GROUP BY dataset
                ORDER BY dataset
            """).fetchall()

            assert len(per_dataset) == 2
            assert per_dataset[0][0] == "jrc_hela-2"
            assert per_dataset[0][1] == 0.85
            assert per_dataset[1][0] == "jrc_hela-3"
            assert per_dataset[1][1] == 0.78
