"""Tests for query functions and CLI subcommands."""

import json
import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner
from pydantic import TypeAdapter

from organelle_mapping.config.evaluation import EvaluationConfig
from organelle_mapping.database import (
    init_database,
    insert_result,
    query_best_per_label,
    query_checkpoint_comparison,
    query_distinct_values,
    query_results,
)
from organelle_mapping.evaluation import checkpoint_name, cli, expected_per_crop
from organelle_mapping.query import format_output, query


def _populate_db(engine):
    """Insert a standard set of test data."""
    # run01: two checkpoints, two datasets, two labels
    insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.80, threshold=0.5)
    insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito_binary", "mito", "jaccard", 0.67, threshold=0.5)
    insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "er_binary", "er", "dice", 0.70, threshold=0.3)
    insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "er_binary", "er", "jaccard", 0.54, threshold=0.3)

    insert_result(engine, "run01", "ckpt_2000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.90, threshold=0.5)
    insert_result(engine, "run01", "ckpt_2000", "ds1", "crop1", "mito_binary", "mito", "jaccard", 0.82, threshold=0.5)
    insert_result(engine, "run01", "ckpt_2000", "ds1", "crop1", "er_binary", "er", "dice", 0.75, threshold=0.4)
    insert_result(engine, "run01", "ckpt_2000", "ds1", "crop1", "er_binary", "er", "jaccard", 0.60, threshold=0.4)

    # Different dataset
    insert_result(engine, "run01", "ckpt_1000", "ds2", "crop3", "mito_binary", "mito", "dice", 0.78, threshold=0.5)
    insert_result(engine, "run01", "ckpt_2000", "ds2", "crop3", "mito_binary", "mito", "dice", 0.88, threshold=0.6)

    # Different run
    insert_result(engine, "run02", "ckpt_500", "ds1", "crop1", "mito_binary", "mito", "dice", 0.72, threshold=0.5)


# --- query_results tests ---


def test_query_results_no_filters():
    """Test querying all results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_results(engine)
        assert len(rows) == 11


def test_query_results_with_filters():
    """Test querying with filters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_results(engine, filters={"run": "run01", "metric": "dice"})
        assert len(rows) == 6
        assert all(r["metric"] == "dice" for r in rows)
        assert all(r["run"] == "run01" for r in rows)


def test_query_results_ordering():
    """Test query result ordering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_results(engine, filters={"metric": "dice"}, order_by="score", sort_direction="desc")
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)


def test_query_results_limit():
    """Test query result limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_results(engine, limit=3)
        assert len(rows) == 3


def test_query_results_empty():
    """Test querying empty database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        rows = query_results(engine)
        assert rows == []


# --- query_best_per_label tests ---


def test_best_per_label():
    """Test finding best score per label."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_best_per_label(engine, metric="dice", filters={"run": "run01"})
        labels = {r["label"] for r in rows}
        assert labels == {"mito", "er"}

        # mito best should be 0.90 from ckpt_2000
        mito_rows = [r for r in rows if r["label"] == "mito"]
        assert any(r["score"] == 0.90 for r in mito_rows)

        # er best should be 0.75 from ckpt_2000
        er_rows = [r for r in rows if r["label"] == "er"]
        assert any(r["score"] == 0.75 for r in er_rows)


def test_best_per_label_with_three_checkpoints():
    """Test best per label correctly picks the top checkpoint among several."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.80, threshold=0.5)
        insert_result(engine, "run01", "ckpt_2000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.90, threshold=0.5)
        insert_result(engine, "run01", "ckpt_3000", "ds1", "crop1", "mito_binary", "mito", "dice", 0.85, threshold=0.5)
        insert_result(engine, "run01", "ckpt_1000", "ds1", "crop1", "er_binary", "er", "dice", 0.70, threshold=0.3)
        insert_result(engine, "run01", "ckpt_2000", "ds1", "crop1", "er_binary", "er", "dice", 0.75, threshold=0.3)

        rows = query_best_per_label(engine, metric="dice")

        mito_best = [r for r in rows if r["label"] == "mito"]
        assert any(r["score"] == 0.90 and r["checkpoint"] == "ckpt_2000" for r in mito_best)

        er_best = [r for r in rows if r["label"] == "er"]
        assert any(r["score"] == 0.75 and r["checkpoint"] == "ckpt_2000" for r in er_best)


def test_best_per_label_filtered():
    """Test best per label with checkpoint filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_best_per_label(engine, metric="dice", filters={"checkpoint": "ckpt_1000"})
        mito_rows = [r for r in rows if r["label"] == "mito"]
        assert any(r["score"] == 0.80 for r in mito_rows)


def test_best_per_label_empty():
    """Test best per label on empty database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        rows = query_best_per_label(engine, metric="dice")
        assert rows == []


# --- query_distinct_values tests ---


def test_distinct_values():
    """Test listing distinct values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        runs = query_distinct_values(engine, "run")
        assert runs == ["run01", "run02"]

        labels = query_distinct_values(engine, "label")
        assert labels == ["er", "mito"]

        channels = query_distinct_values(engine, "channel")
        assert channels == ["er_binary", "mito_binary"]

        checkpoints = query_distinct_values(engine, "checkpoint", filters={"run": "run01"})
        assert checkpoints == ["ckpt_1000", "ckpt_2000"]


def test_distinct_values_empty():
    """Test listing distinct values on empty database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        values = query_distinct_values(engine, "run")
        assert values == []


# --- query_checkpoint_comparison tests ---


def test_checkpoint_comparison():
    """Test comparing checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_checkpoint_comparison(engine, metric="dice", filters={"run": "run01"})
        assert len(rows) > 0

        # All rows should have label, checkpoint, avg_score, num_crops
        for row in rows:
            assert "label" in row
            assert "checkpoint" in row
            assert "avg_score" in row
            assert "num_crops" in row


def test_checkpoint_comparison_across_datasets():
    """Test that comparison aggregates across datasets correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        # mito has results in both ds1 and ds2 for ckpt_2000
        rows = query_checkpoint_comparison(engine, metric="dice", filters={"run": "run01", "label": "mito"})
        ckpt_2000 = next(r for r in rows if r["checkpoint"] == "ckpt_2000")
        # ds1: 0.90, ds2: 0.88 → avg = 0.89, num_crops = 2
        assert ckpt_2000["num_crops"] == 2
        assert abs(ckpt_2000["avg_score"] - 0.89) < 1e-6


def test_checkpoint_comparison_filtered():
    """Test comparing checkpoints with label filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = init_database(f"sqlite:///{tmpdir}/test.db")
        _populate_db(engine)

        rows = query_checkpoint_comparison(engine, metric="dice", filters={"run": "run01", "label": "mito"})
        assert all(r["label"] == "mito" for r in rows)
        checkpoints = [r["checkpoint"] for r in rows]
        assert "ckpt_1000" in checkpoints
        assert "ckpt_2000" in checkpoints


# --- format_output tests ---


def test_format_output_table():
    """Test table format output."""
    rows = [{"name": "mito", "score": 0.85}, {"name": "er", "score": 0.72}]
    output = format_output(rows, ["name", "score"], "table")
    assert "name" in output
    assert "score" in output
    assert "mito" in output
    assert "0.8500" in output


def test_format_output_csv():
    """Test CSV format output."""
    rows = [{"name": "mito", "score": 0.85}]
    output = format_output(rows, ["name", "score"], "csv")
    lines = [line.strip() for line in output.strip().split("\n")]
    assert lines[0] == "name,score"
    assert "mito" in lines[1]


def test_format_output_json():
    """Test JSON format output."""
    rows = [{"name": "mito", "score": 0.85, "extra": "ignored"}]
    output = format_output(rows, ["name", "score"], "json")
    parsed = json.loads(output)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "mito"
    assert parsed[0]["score"] == 0.85
    assert "extra" not in parsed[0]


def test_format_output_empty():
    """Test output with no results."""
    output = format_output([], ["name"], "table")
    assert "No results" in output


# --- CLI tests ---


def test_cli_best():
    """Test the 'best' CLI subcommand."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)
        _populate_db(engine)

        result = runner.invoke(query, ["--db-url", db_url, "best", "--metric", "dice"])
        assert result.exit_code == 0
        assert "mito" in result.output
        assert "er" in result.output


def test_cli_compare():
    """Test the 'compare' CLI subcommand."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)
        _populate_db(engine)

        result = runner.invoke(query, ["--db-url", db_url, "compare", "--run", "run01"])
        assert result.exit_code == 0
        assert "ckpt_1000" in result.output
        assert "ckpt_2000" in result.output


def test_cli_list():
    """Test the 'list' CLI subcommand."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)
        _populate_db(engine)

        result = runner.invoke(query, ["--db-url", db_url, "list", "runs"])
        assert result.exit_code == 0
        assert "run01" in result.output
        assert "run02" in result.output


def test_cli_scores():
    """Test the 'scores' CLI subcommand."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)
        _populate_db(engine)

        result = runner.invoke(query, ["--db-url", db_url, "scores", "--run", "run01", "--metric", "dice"])
        assert result.exit_code == 0
        assert "run01" in result.output


def test_cli_json_output():
    """Test JSON output format via CLI."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)
        _populate_db(engine)

        result = runner.invoke(query, ["--db-url", db_url, "--format", "json", "best", "--metric", "dice"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) > 0


def test_cli_csv_output():
    """Test CSV output format via CLI."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        engine = init_database(db_url)
        _populate_db(engine)

        result = runner.invoke(query, ["--db-url", db_url, "--format", "csv", "scores", "--limit", "3"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows


def test_cli_empty_database():
    """Test CLI with empty database."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{tmpdir}/test.db"
        init_database(db_url)

        result = runner.invoke(query, ["--db-url", db_url, "best"])
        assert result.exit_code == 0
        assert "No results" in result.output


# --- checkpoint_name tests ---


def test_checkpoint_name():
    """Test checkpoint name extraction from path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "run01" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "model_checkpoint_100000"
        ckpt_file.touch()

        assert checkpoint_name(str(ckpt_file)) == "model_checkpoint_100000"


def test_checkpoint_name_with_extension():
    """Test checkpoint name strips file extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "experiments" / "models"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "best_model.pt"
        ckpt_file.touch()

        assert checkpoint_name(str(ckpt_file)) == "best_model"


# --- expected_per_crop tests ---


def _make_minimal_eval_config(tmpdir, checkpoints=None):
    """Create a minimal EvaluationConfig for testing.

    Returns (eval_cfg, checkpoint_paths).
    """
    # Create checkpoint files
    if checkpoints is None:
        checkpoints = ["model_checkpoint_1000"]
    ckpt_dir = Path(tmpdir) / "run01" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_paths = []
    for ckpt in checkpoints:
        ckpt_file = ckpt_dir / ckpt
        ckpt_file.touch()
        ckpt_paths.append(str(ckpt_file))

    run_config = {
        "name": "test_run",
        "iterations": 1000,
        "targets": [
            {
                "loss_function": {"type": "mse"},
                "transforms": [
                    {"type": "binary", "source": "mito"},
                    {"type": "binary", "source": "er"},
                ],
            }
        ],
        "architecture": {
            "name": "standard_unet",
            "input_shape": [204, 204, 204],
            "output_shape": [64, 64, 64],
            "out_channels": 2,
            "in_channels": 1,
            "num_fmaps": 16,
            "fmap_inc_factor": 6,
        },
        "data": {
            "datasets": {
                "dummy": {
                    "em": {"data": "/fake/em.zarr", "group": "em", "contrast": [0, 255]},
                    "labels": {"data": "/fake/labels.zarr", "group": "labels", "crops": ["c1"]},
                },
            },
        },
        "augmentations": {"augmentations": []},
        "sampling": {"z": 8, "y": 8, "x": 8},
    }

    config_dict = {
        "experiment_run": run_config,
        "checkpoints": ckpt_paths,
        "data": {
            "datasets": {
                "ds1": {
                    "em": {"data": "/fake/em.zarr", "group": "em", "contrast": [0, 255]},
                    "labels": {"data": "/fake/labels.zarr", "group": "labels", "crops": ["crop1", "crop2,crop3"]},
                },
            },
        },
        "default_metrics": ["dice", "jaccard"],
        "default_postprocessing": [
            {"type": "threshold", "threshold": 0.3},
            {"type": "threshold", "threshold": 0.5},
        ],
        "eval_channels": [
            {"channel": "mito_binary"},
            {"channel": "er_binary"},
        ],
    }

    eval_cfg = TypeAdapter(EvaluationConfig).validate_python(config_dict)
    return eval_cfg, ckpt_paths


def test_expected_per_crop_count():
    """Test that expected_per_crop produces the correct number of keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_cfg, _ckpt_paths = _make_minimal_eval_config(tmpdir)

        keys = expected_per_crop(eval_cfg)

        # 2 channels x 2 postprocessing x 2 metrics = 8
        assert len(keys) == 8


def test_expected_per_crop_structure():
    """Test that each key tuple has the expected structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_cfg, _ckpt_paths = _make_minimal_eval_config(tmpdir)

        keys = expected_per_crop(eval_cfg)

        for key in keys:
            # (channel, pp_type, <param values...>, metric)
            assert len(key) == 4
            channel, pp_type, threshold, metric = key
            assert channel in ("mito_binary", "er_binary")
            assert pp_type == "threshold"
            assert threshold in (0.3, 0.5)
            assert metric in ("dice", "jaccard")


# --- status CLI tests ---


def _write_eval_config_file(tmpdir, ckpt_paths):
    """Write a minimal eval config YAML file for CLI testing."""
    config_file = Path(tmpdir) / "eval.yaml"
    config_dict = {
        "experiment_run": {
            "name": "test_run",
            "iterations": 1000,
            "targets": [
                {
                    "loss_function": {"type": "mse"},
                    "transforms": [{"type": "binary", "source": "mito"}],
                }
            ],
            "architecture": {
                "name": "standard_unet",
                "input_shape": [204, 204, 204],
                "output_shape": [64, 64, 64],
                "out_channels": 1,
                "in_channels": 1,
                "num_fmaps": 16,
                "fmap_inc_factor": 6,
            },
            "data": {
                "datasets": {
                    "dummy": {
                        "em": {"data": "/fake/em.zarr", "group": "em", "contrast": [0, 255]},
                        "labels": {"data": "/fake/labels.zarr", "group": "labels", "crops": ["c1"]},
                    },
                },
            },
            "augmentations": {"augmentations": []},
            "sampling": {"z": 8, "y": 8, "x": 8},
        },
        "checkpoints": ckpt_paths,
        "data": {
            "datasets": {
                "ds1": {
                    "em": {"data": "/fake/em.zarr", "group": "em", "contrast": [0, 255]},
                    "labels": {"data": "/fake/labels.zarr", "group": "labels", "crops": ["crop1"]},
                },
            },
        },
        "default_metrics": ["dice"],
        "default_postprocessing": [{"type": "threshold", "threshold": 0.5}],
        "eval_channels": [{"channel": "mito_binary"}],
    }
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    return config_file


def test_cli_status_empty_db():
    """Test status command with empty database."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create checkpoint files
        ckpt_dir = Path(tmpdir) / "run01" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "model_checkpoint_1000"
        ckpt_file.touch()
        ckpt_paths = [str(ckpt_file)]

        db_url = f"sqlite:///{tmpdir}/test.db"
        init_database(db_url)

        config_file = _write_eval_config_file(tmpdir, ckpt_paths)

        result = runner.invoke(cli, ["status", "-e", str(config_file), "--db-url", db_url])
        assert result.exit_code == 0, result.output
        assert "Expected:" in result.output
        assert "Missing:" in result.output
