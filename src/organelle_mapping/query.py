"""CLI subcommands for querying evaluation results from the database."""

import csv
import io
import json
import logging
from typing import Optional

import click

from organelle_mapping.database import (
    init_database,
    query_best_per_label,
    query_checkpoint_comparison,
    query_distinct_values,
    query_results,
)

logger = logging.getLogger(__name__)

# Mapping from user-friendly plural names to column names
COLUMN_ALIASES = {
    "runs": "run",
    "checkpoints": "checkpoint",
    "datasets": "dataset",
    "crops": "crop",
    "channels": "channel",
    "labels": "label",
    "postprocessing_types": "postprocessing_type",
    "metrics": "metric",
}


def format_output(rows: list[dict], columns: list[str], fmt: str) -> str:
    """Format query results as table, CSV, or JSON.

    Args:
        rows: List of dicts with query results.
        columns: Column names to include, in order.
        fmt: Output format — "table", "csv", or "json".
    """
    if not rows:
        return "No results found."

    if fmt == "json":
        filtered = [{col: row.get(col) for col in columns} for row in rows]
        return json.dumps(filtered, indent=2)

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue().rstrip()

    # Table format
    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val) if val is not None else ""
            widths[col] = max(widths[col], len(val_str))

    # Build header
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    separator = "  ".join("-" * widths[col] for col in columns)

    # Build rows
    lines = [header, separator]
    for row in rows:
        parts = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val) if val is not None else ""
            parts.append(val_str.ljust(widths[col]))
        lines.append("  ".join(parts))

    return "\n".join(lines)


@click.group()
@click.option("--db-url", required=True, help="SQLAlchemy database URL (e.g. 'sqlite:///results.db')")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    help="Output format (default: table)",
)
@click.pass_context
def query(ctx, db_url: str, output_format: str):
    """Query evaluation results database."""
    ctx.ensure_object(dict)
    ctx.obj["engine"] = init_database(db_url)
    ctx.obj["format"] = output_format


@query.command()
@click.option("--metric", default="dice", help="Metric to rank by (default: dice)")
@click.option("--run", "run_name", default=None, help="Filter by run name")
@click.option("--label", default=None, help="Filter by label")
@click.option("--dataset", default=None, help="Filter by dataset")
@click.option("--checkpoint", default=None, help="Filter by checkpoint")
@click.pass_context
def best(
    ctx, metric: str, run_name: Optional[str], label: Optional[str], dataset: Optional[str], checkpoint: Optional[str]
):
    """Find the best score per label."""
    engine = ctx.obj["engine"]
    fmt = ctx.obj["format"]

    filters = {}
    if run_name:
        filters["run"] = run_name
    if label:
        filters["label"] = label
    if dataset:
        filters["dataset"] = dataset
    if checkpoint:
        filters["checkpoint"] = checkpoint

    rows = query_best_per_label(engine, metric=metric, filters=filters or None)
    columns = ["label", "channel", "score", "checkpoint", "postprocessing_type", "threshold", "dataset", "crop", "run"]
    click.echo(format_output(rows, columns, fmt))


@query.command()
@click.option("--metric", default="dice", help="Metric to compare (default: dice)")
@click.option("--run", "run_name", default=None, help="Filter by run name")
@click.option("--label", default=None, help="Filter by label")
@click.option("--dataset", default=None, help="Filter by dataset")
@click.pass_context
def compare(ctx, metric: str, run_name: Optional[str], label: Optional[str], dataset: Optional[str]):
    """Compare checkpoints side-by-side (average score per label per checkpoint)."""
    engine = ctx.obj["engine"]
    fmt = ctx.obj["format"]

    filters = {}
    if run_name:
        filters["run"] = run_name
    if label:
        filters["label"] = label
    if dataset:
        filters["dataset"] = dataset

    rows = query_checkpoint_comparison(engine, metric=metric, filters=filters or None)
    columns = ["label", "checkpoint", "avg_score", "num_crops"]
    click.echo(format_output(rows, columns, fmt))


@query.command("list")
@click.argument("column", type=click.Choice(list(COLUMN_ALIASES.keys())))
@click.option("--run", "run_name", default=None, help="Filter by run name")
@click.option("--checkpoint", default=None, help="Filter by checkpoint")
@click.option("--dataset", default=None, help="Filter by dataset")
@click.option("--label", default=None, help="Filter by label")
@click.pass_context
def list_values(
    ctx, column: str, run_name: Optional[str], checkpoint: Optional[str], dataset: Optional[str], label: Optional[str]
):
    """List distinct values for a column (runs, checkpoints, datasets, crops, labels, metrics)."""
    engine = ctx.obj["engine"]
    fmt = ctx.obj["format"]

    actual_column = COLUMN_ALIASES[column]

    filters = {}
    if run_name:
        filters["run"] = run_name
    if checkpoint:
        filters["checkpoint"] = checkpoint
    if dataset:
        filters["dataset"] = dataset
    if label:
        filters["label"] = label

    values = query_distinct_values(engine, actual_column, filters=filters or None)

    if not values:
        click.echo("No results found.")
        return

    if fmt == "json":
        click.echo(json.dumps(values, indent=2))
    elif fmt == "csv":
        click.echo(actual_column)
        for v in values:
            click.echo(v)
    else:
        click.echo(f"{actual_column}")
        click.echo("-" * max(len(actual_column), *(len(str(v)) for v in values)))
        for v in values:
            click.echo(v)


@query.command()
@click.option("--run", "run_name", default=None, help="Filter by run name")
@click.option("--checkpoint", default=None, help="Filter by checkpoint")
@click.option("--dataset", default=None, help="Filter by dataset")
@click.option("--crop", default=None, help="Filter by crop")
@click.option("--label", default=None, help="Filter by label")
@click.option("--metric", default=None, help="Filter by metric name")
@click.option("--limit", default=100, help="Maximum number of results (default: 100)")
@click.option("--order-by", default="score", help="Column to sort by (default: score)")
@click.option(
    "--sort", "sort_direction", type=click.Choice(["asc", "desc"]), default="desc", help="Sort order (default: desc)"
)
@click.pass_context
def scores(
    ctx,
    run_name: Optional[str],
    checkpoint: Optional[str],
    dataset: Optional[str],
    crop: Optional[str],
    label: Optional[str],
    metric: Optional[str],
    limit: int,
    order_by: str,
    sort_direction: str,
):
    """Dump raw scores with flexible filtering."""
    engine = ctx.obj["engine"]
    fmt = ctx.obj["format"]

    filters = {}
    if run_name:
        filters["run"] = run_name
    if checkpoint:
        filters["checkpoint"] = checkpoint
    if dataset:
        filters["dataset"] = dataset
    if crop:
        filters["crop"] = crop
    if label:
        filters["label"] = label
    if metric:
        filters["metric"] = metric

    rows = query_results(engine, filters=filters or None, limit=limit, order_by=order_by, sort_direction=sort_direction)
    columns = [
        "run",
        "checkpoint",
        "dataset",
        "crop",
        "channel",
        "label",
        "postprocessing_type",
        "threshold",
        "metric",
        "score",
    ]
    click.echo(format_output(rows, columns, fmt))
