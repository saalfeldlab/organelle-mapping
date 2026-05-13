"""Standard plots from the evaluation database.

Subcommands:
- iteration:   metric vs training iteration, faceted by channel, lines per run
- threshold:   metric vs threshold for one (run, checkpoint), lines per channel
- leaderboard: best per-channel (run, ckpt, threshold) shown as a bar chart
- val-vs-all:  paired scatter of /val vs /all scores at matched (base run, ckpt, channel)

All commands filter out crops with `present <= --min-present` (defaults to 100)
to avoid being fooled by all-negative crops.
"""

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _crops_pairs_from_yaml(path: str | None) -> list[tuple[str, str]]:
    """Parse a DataConfig-style yaml and return all (dataset, crop) pairs (comma-entries split)."""
    if not path:
        return []
    with open(path) as f:
        cfg = yaml.safe_load(f)
    pairs: list[tuple[str, str]] = []
    for ds_name, info in cfg["datasets"].items():
        for entry in info["labels"]["crops"]:
            for sub in entry.split(","):
                pairs.append((ds_name, sub.strip()))
    return pairs


def _nonempty_crops_cte(crops_pairs: list[tuple[str, str]] | None) -> tuple[str, list[str]]:
    """Build the nonempty_crops CTE, optionally restricted to a list of (dataset, crop) pairs.

    Returns the CTE SQL and any leading positional args (for the VALUES tuple).
    The trailing `?` in `HAVING MAX(present) > ?` is filled by the caller as before.

    The `evaluable` column is the per-(dataset, crop, label) voxel count at s0
    that's not masked (= total_voxels - unknown = present + absent). Used as the
    aggregation weight: bigger evaluable region → more influence on the macro.
    """
    inner_select = """
        SELECT c.dataset, c.crop, c.label,
               COALESCE(MAX(c.crop_group), c.crop) AS crop_group,
               MAX(CASE WHEN c.scale_level = 's0' THEN c.total_voxels - c.unknown END) AS evaluable
    """
    if not crops_pairs:
        cte = f"""
        WITH nonempty_crops AS (
          {inner_select}
          FROM crops c
          GROUP BY c.dataset, c.crop, c.label
          HAVING MAX(c.present) > ?
        )
        """
        return cte, []
    placeholders = ",".join(["(?,?)"] * len(crops_pairs))
    flat = [v for p in crops_pairs for v in p]
    cte = f"""
    WITH crop_filter AS (SELECT column1 AS dataset, column2 AS crop FROM (VALUES {placeholders})),
    nonempty_crops AS (
      {inner_select}
      FROM crops c JOIN crop_filter cf ON cf.dataset=c.dataset AND cf.crop=c.crop
      GROUP BY c.dataset, c.crop, c.label
      HAVING MAX(c.present) > ?
    )
    """
    return cte, flat


# Two-stage aggregation:
#   1) inner per_group: SUM(score * evaluable), SUM(evaluable) within (dataset, crop_group)
#   2) outer macro:     SUM(sw) / SUM(w) across groups
# Equivalent to a single voxel-weighted mean over all sub-crops, but expressed in two
# stages so crop_group structure stays explicit. Weighting by evaluable voxels means
# bigger annotated regions contribute proportionally more to the macro — comma-separated
# sub-crops naturally combine into their joint annotation size.


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _checkpoint_iter(ckpt: str) -> int:
    """Extract iteration number from a checkpoint name like 'model_checkpoint_50000'."""
    digits = "".join(c if c.isdigit() else " " for c in ckpt).split()
    return int(digits[-1]) if digits else 0


def _split_of(run: str) -> tuple[str, str] | tuple[None, None]:
    """Split a run name like 'run13/val' into (base, split)."""
    if "/" in run:
        base, split = run.rsplit("/", 1)
        if split in ("val", "all"):
            return base, split
    return None, None


def _channel_filter(channels: str | None) -> tuple[str, list[str]]:
    if not channels:
        return "", []
    chans = [c.strip() for c in channels.split(",") if c.strip()]
    return f"AND r.channel IN ({','.join(['?'] * len(chans))})", chans


def _dataset_filter(datasets: str | None) -> tuple[str, list[str]]:
    if not datasets:
        return "", []
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    return f"AND r.dataset IN ({','.join(['?'] * len(ds))})", ds


@click.group()
@click.option("--db", "db_path", required=True, type=click.Path(exists=True), help="Path to eval SQLite DB.")
@click.option("--min-present", default=100, type=int, help="Drop crops with present <= this (default 100).")
@click.option("--out-dir", default="./plots", type=click.Path(), help="Output directory for PNGs.")
@click.option("--datasets", default=None, help="Comma-separated dataset names to restrict to (default = all).")
@click.option(
    "--crops-yaml",
    "crops_yamls",
    multiple=True,
    type=click.Path(exists=True),
    help=(
        "DataConfig yaml whose (dataset, crop) entries restrict the analysis. "
        "Repeatable: in `iteration`, each yaml becomes one curve set (overlaid by linestyle). "
        "Other subcommands use the first yaml only."
    ),
)
@click.pass_context
def cli(
    ctx: click.Context,
    db_path: str,
    min_present: int,
    out_dir: str,
    datasets: str | None,
    crops_yamls: tuple[str, ...],
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["db"] = db_path
    ctx.obj["min_present"] = min_present
    ctx.obj["out_dir"] = Path(out_dir)
    ctx.obj["datasets"] = datasets
    # crops_sets: list of (label, pairs). Empty list means "no crop filter".
    ctx.obj["crops_sets"] = [(Path(p).stem, _crops_pairs_from_yaml(p)) for p in crops_yamls]
    # First set populates crops_pairs / crops_label for subcommands that take a single set.
    if ctx.obj["crops_sets"]:
        ctx.obj["crops_label"], ctx.obj["crops_pairs"] = ctx.obj["crops_sets"][0]
        logger.info(
            f"Restricting analysis to {len(ctx.obj['crops_pairs'])} pairs from {crops_yamls[0]}"
            + (f" (+ {len(crops_yamls) - 1} more sets)" if len(crops_yamls) > 1 else "")
        )
    else:
        ctx.obj["crops_label"] = None
        ctx.obj["crops_pairs"] = []
    ctx.obj["out_dir"].mkdir(parents=True, exist_ok=True)


def _query_iteration(
    ctx: click.Context,
    crops_pairs: list[tuple[str, str]],
    metric: str,
    run_pattern: str,
    channels: str | None,
) -> dict[tuple[str, str, int], float]:
    """Return {(run, channel, iter) -> best-threshold score} for the given crops_pairs."""
    chan_clause, chan_args = _channel_filter(channels)
    ds_clause, ds_args = _dataset_filter(ctx.obj.get("datasets"))
    cte, cte_args = _nonempty_crops_cte(crops_pairs)
    sql = cte + f"""
        , per_group AS (
          SELECT r.run, r.checkpoint, r.channel, r.threshold, r.dataset, v.crop_group,
                 SUM(r.score * v.evaluable) AS sw, SUM(v.evaluable) AS w
          FROM results r
          JOIN nonempty_crops v ON v.dataset=r.dataset AND v.crop=r.crop AND v.label=r.label
          WHERE r.metric = ? AND r.postprocessing_type = 'threshold' AND r.run LIKE ?
            {chan_clause} {ds_clause}
          GROUP BY r.run, r.checkpoint, r.channel, r.threshold, r.dataset, v.crop_group
        )
        SELECT run, checkpoint, channel, threshold, SUM(sw) / SUM(w) AS score
        FROM per_group
        GROUP BY run, checkpoint, channel, threshold
        """
    conn = _connect(ctx.obj["db"])
    rows = conn.execute(
        sql, [*cte_args, ctx.obj["min_present"], metric, run_pattern, *chan_args, *ds_args]
    ).fetchall()
    conn.close()

    best: dict[tuple[str, str, int], float] = {}
    for r in rows:
        key = (r["run"], r["channel"], _checkpoint_iter(r["checkpoint"]))
        s = r["score"]
        if key not in best or s > best[key]:
            best[key] = s
    return best


_OVERLAY_STYLES = [
    ("-", "o"),
    ("--", "s"),
    (":", "^"),
    ("-.", "D"),
]


@cli.command("iteration")
@click.option("--metric", default="dice")
@click.option("--run-pattern", default="%/val", help="SQL LIKE pattern on run name (default '%/val').")
@click.option("--channels", default=None, help="Comma-separated channels; default = all.")
@click.option("--out-name", default=None, help="Output filename (default 'iteration_<metric>.png').")
@click.pass_context
def iteration_cmd(
    ctx: click.Context,
    metric: str,
    run_pattern: str,
    channels: str | None,
    out_name: str | None,
) -> None:
    """Plot metric vs training iteration, one subplot per channel.

    For each (run, ckpt, channel) the best threshold is taken — that's the
    "achievable score at this checkpoint" curve.

    If multiple --crops-yaml are passed at the global level, each becomes one
    curve set (overlaid via linestyle/marker) so you can compare e.g. training-
    crop fit vs. held-out generalization.
    """
    crops_sets = ctx.obj.get("crops_sets") or [("all-crops", [])]
    if len(crops_sets) > len(_OVERLAY_STYLES):
        logger.warning(f"Only the first {len(_OVERLAY_STYLES)} crop sets will be plotted distinctly")
        crops_sets = crops_sets[: len(_OVERLAY_STYLES)]

    queried = [(label, _query_iteration(ctx, pairs, metric, run_pattern, channels)) for label, pairs in crops_sets]
    queried = [(lbl, data) for lbl, data in queried if data]
    if not queried:
        logger.warning("No data returned")
        return

    chans: set[str] = set()
    runs: set[str] = set()
    for _, data in queried:
        chans |= {k[1] for k in data}
        runs |= {k[0] for k in data}
    chans_list = sorted(chans)
    runs_list = sorted(runs)

    cols = min(3, len(chans_list))
    rows_n = (len(chans_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 3.5 * rows_n), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")
    colors = {run: cmap(i % 10) for i, run in enumerate(runs_list)}

    for ax, ch in zip(axes, chans_list, strict=False):
        for (label, data), (linestyle, marker) in zip(queried, _OVERLAY_STYLES, strict=False):
            for run in runs_list:
                pts = sorted([(it, s) for (r_, c_, it), s in data.items() if r_ == run and c_ == ch])
                if pts:
                    its, scs = zip(*pts, strict=False)
                    ax.plot(its, scs, marker=marker, linestyle=linestyle, color=colors[run], markersize=4)
        ax.set_title(ch)
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(chans_list) :]:
        ax.set_visible(False)

    run_handles = [plt.Line2D([0], [0], color=colors[r], label=r) for r in runs_list]
    style_handles = []
    if len(queried) > 1:
        style_handles = [
            plt.Line2D([0], [0], color="black", linestyle=ls, marker=mk, label=label)
            for (label, _), (ls, mk) in zip(queried, _OVERLAY_STYLES, strict=False)
        ]
    fig.legend(
        handles=run_handles + style_handles,
        loc="upper center",
        ncol=min(len(run_handles) + len(style_handles), 6),
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout()
    out = ctx.obj["out_dir"] / (out_name or f"iteration_{metric}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


@cli.command("threshold")
@click.option("--run", "run_name", required=True)
@click.option("--checkpoint", required=True)
@click.option("--metric", default="dice")
@click.option("--channels", default=None, help="Comma-separated channels; default = all.")
@click.option("--out-name", default=None)
@click.pass_context
def threshold_cmd(
    ctx: click.Context, run_name: str, checkpoint: str, metric: str, channels: str | None, out_name: str | None
) -> None:
    """Plot metric vs threshold for a single (run, ckpt), one line per channel."""
    chan_clause, chan_args = _channel_filter(channels)
    ds_clause, ds_args = _dataset_filter(ctx.obj.get("datasets"))
    cte, cte_args = _nonempty_crops_cte(ctx.obj.get("crops_pairs"))
    sql = cte + f"""
        , per_group AS (
          SELECT r.channel, r.threshold, r.dataset, v.crop_group,
                 SUM(r.score * v.evaluable) AS sw, SUM(v.evaluable) AS w
          FROM results r
          JOIN nonempty_crops v ON v.dataset=r.dataset AND v.crop=r.crop AND v.label=r.label
          WHERE r.metric = ? AND r.postprocessing_type = 'threshold'
            AND r.run = ? AND r.checkpoint = ?
            {chan_clause} {ds_clause}
          GROUP BY r.channel, r.threshold, r.dataset, v.crop_group
        )
        SELECT channel, threshold, SUM(sw) / SUM(w) AS score
        FROM per_group
        GROUP BY channel, threshold
        ORDER BY channel, threshold
        """
    conn = _connect(ctx.obj["db"])
    rows = conn.execute(
        sql, [*cte_args, ctx.obj["min_present"], metric, run_name, checkpoint, *chan_args, *ds_args]
    ).fetchall()
    conn.close()
    if not rows:
        logger.warning("No data returned")
        return

    by_chan: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by_chan[r["channel"]].append((r["threshold"], r["score"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    for ch, pts in sorted(by_chan.items()):
        pts.sort()
        ts, scs = zip(*pts, strict=False)
        ax.plot(ts, scs, "o-", label=ch, markersize=4)
    ax.set_xlabel("threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"{run_name} / {checkpoint}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    safe = f"{run_name.replace('/', '_')}_{checkpoint}_{metric}_threshold.png"
    out = ctx.obj["out_dir"] / (out_name or safe)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {out}")


@cli.command("leaderboard")
@click.option("--metric", default="dice")
@click.option("--run-pattern", default="%/val", help="SQL LIKE pattern on run name.")
@click.option("--out-name", default=None)
@click.pass_context
def leaderboard_cmd(ctx: click.Context, metric: str, run_pattern: str, out_name: str | None) -> None:
    """Best per-channel (run, ckpt, threshold) score as a horizontal bar chart."""
    ds_clause, ds_args = _dataset_filter(ctx.obj.get("datasets"))
    cte, cte_args = _nonempty_crops_cte(ctx.obj.get("crops_pairs"))
    sql = cte + f"""
        , per_group AS (
          SELECT r.channel, r.run, r.checkpoint, r.threshold, r.dataset, v.crop_group,
                 SUM(r.score * v.evaluable) AS sw, SUM(v.evaluable) AS w
          FROM results r
          JOIN nonempty_crops v ON v.dataset=r.dataset AND v.crop=r.crop AND v.label=r.label
          WHERE r.metric = ? AND r.postprocessing_type = 'threshold' AND r.run LIKE ?
            {ds_clause}
          GROUP BY r.channel, r.run, r.checkpoint, r.threshold, r.dataset, v.crop_group
        )
        SELECT channel, run, checkpoint, threshold, SUM(sw) / SUM(w) AS score
        FROM per_group
        GROUP BY channel, run, checkpoint, threshold
        """
    conn = _connect(ctx.obj["db"])
    rows = conn.execute(sql, [*cte_args, ctx.obj["min_present"], metric, run_pattern, *ds_args]).fetchall()
    conn.close()

    by_chan: dict[str, dict] = {}
    for r in rows:
        prev = by_chan.get(r["channel"])
        if prev is None or r["score"] > prev["score"]:
            by_chan[r["channel"]] = dict(r)

    items = sorted(by_chan.items(), key=lambda kv: kv[1]["score"])
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(items))))
    labels = [
        f'{ch}  [{v["run"].rsplit("/", 1)[0]} @ {_checkpoint_iter(v["checkpoint"])} t={v["threshold"]}]'
        for ch, v in items
    ]
    scores = [v["score"] for _, v in items]
    ax.barh(labels, scores)
    for i, s in enumerate(scores):
        ax.text(s + 0.005, i, f"{s:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(f"Best per-channel {metric} (run pattern: {run_pattern})")
    fig.tight_layout()
    out = ctx.obj["out_dir"] / (out_name or f"leaderboard_{metric}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {out}")


@cli.command("val-vs-all")
@click.option("--metric", default="dice")
@click.option("--channels", default=None, help="Comma-separated channels; default = all.")
@click.option("--out-name", default=None)
@click.pass_context
def val_vs_all_cmd(ctx: click.Context, metric: str, channels: str | None, out_name: str | None) -> None:
    """Scatter of /val vs /all scores at matched (base run, ckpt, channel).

    Useful for visualizing how much training-set leakage inflates /all numbers.
    """
    chan_clause, chan_args = _channel_filter(channels)
    ds_clause, ds_args = _dataset_filter(ctx.obj.get("datasets"))
    cte, cte_args = _nonempty_crops_cte(ctx.obj.get("crops_pairs"))
    sql = cte + f"""
        , per_group AS (
          SELECT r.run, r.checkpoint, r.channel, r.threshold, r.dataset, v.crop_group,
                 SUM(r.score * v.evaluable) AS sw, SUM(v.evaluable) AS w
          FROM results r
          JOIN nonempty_crops v ON v.dataset=r.dataset AND v.crop=r.crop AND v.label=r.label
          WHERE r.metric = ? AND r.postprocessing_type = 'threshold'
            {chan_clause} {ds_clause}
          GROUP BY r.run, r.checkpoint, r.channel, r.threshold, r.dataset, v.crop_group
        )
        SELECT run, checkpoint, channel, threshold, SUM(sw) / SUM(w) AS score
        FROM per_group
        GROUP BY run, checkpoint, channel, threshold
        """
    conn = _connect(ctx.obj["db"])
    rows = conn.execute(sql, [*cte_args, ctx.obj["min_present"], metric, *chan_args, *ds_args]).fetchall()
    conn.close()

    best: dict[tuple[str, str, str], float] = {}
    for r in rows:
        k = (r["run"], r["checkpoint"], r["channel"])
        if k not in best or r["score"] > best[k]:
            best[k] = r["score"]

    paired: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for (run, ckpt, ch), s in best.items():
        base, split = _split_of(run)
        if base is None:
            continue
        paired[(base, ckpt, ch)][split] = s

    pts = [(k, v["val"], v["all"]) for k, v in paired.items() if "val" in v and "all" in v]
    if not pts:
        logger.warning("No matched val/all pairs found")
        return

    chans_present = sorted({p[0][2] for p in pts})
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(chans_present)}

    fig, ax = plt.subplots(figsize=(7, 7))
    for (_, _, ch), v, a in pts:
        ax.scatter(v, a, color=colors[ch], alpha=0.7, s=30)
    val_scores = [p[1] for p in pts]
    all_scores = [p[2] for p in pts]
    lo = min(min(val_scores), min(all_scores))
    hi = max(max(val_scores), max(all_scores))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel(f"{metric} on /val (held-out)")
    ax.set_ylabel(f"{metric} on /all (training-set leaked)")
    ax.set_title("/all vs /val per (base run, ckpt, channel)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], label=c, markersize=8)
        for c in chans_present
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = ctx.obj["out_dir"] / (out_name or f"val_vs_all_{metric}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {out}")


@cli.command("threshold-compare")
@click.option("--channel", required=True, help="Single channel to plot.")
@click.option(
    "--setup",
    "setups",
    multiple=True,
    required=True,
    help="Setup as 'run:checkpoint'. Repeat for each setup to overlay (e.g. --setup run13/val:model_checkpoint_50000).",
)
@click.option("--metric", default="dice")
@click.option(
    "--facet-finetune",
    is_flag=True,
    default=False,
    help="Put from-scratch runs and finetuned runs (name contains '_ft-') in two side-by-side "
    "subplots, sharing a colour across panels for matched base/finetune pairs.",
)
@click.option(
    "--color-by",
    type=click.Choice(["run-pair", "iteration"]),
    default="run-pair",
    help="'run-pair' (default): one colour per base/finetune pair, /val solid vs /all dashed. "
    "'iteration': colour curves by training iteration (sequential cmap + colorbar), linestyle "
    "= solid for base runs / dashed for finetuned ('_ft-') runs — handy for plotting many "
    "checkpoints of one run as a trajectory.",
)
@click.option("--out-name", default=None)
@click.pass_context
def threshold_compare_cmd(
    ctx: click.Context,
    channel: str,
    setups: tuple[str, ...],
    metric: str,
    facet_finetune: bool,
    color_by: str,
    out_name: str | None,
) -> None:
    """Plot metric vs threshold for ONE channel across MULTIPLE (run, ckpt) setups.

    Inverse of `threshold` cmd: that one shows many channels for a single setup;
    this one shows many setups for a single channel — useful for picking a
    deployment setup on a specific organelle. `/val` and `/all` of the same base
    run share a colour (solid vs dashed); `--facet-finetune` additionally splits
    from-scratch vs finetuned runs into two panels; `--color-by iteration` instead
    colours by training iteration (with a colorbar) for trajectory-style plots.
    """
    parsed = []
    for s in setups:
        if ":" not in s:
            raise click.BadParameter(f"setup '{s}' must be 'run:checkpoint'")
        run, ckpt = s.split(":", 1)
        parsed.append((run, ckpt))

    ds_clause, ds_args = _dataset_filter(ctx.obj.get("datasets"))
    pair_clause = " OR ".join(["(r.run = ? AND r.checkpoint = ?)"] * len(parsed))
    pair_args: list[str] = []
    for run, ckpt in parsed:
        pair_args.extend([run, ckpt])

    cte, cte_args = _nonempty_crops_cte(ctx.obj.get("crops_pairs"))
    sql = cte + f"""
        , per_group AS (
          SELECT r.run, r.checkpoint, r.threshold, r.dataset, v.crop_group,
                 SUM(r.score * v.evaluable) AS sw, SUM(v.evaluable) AS w
          FROM results r
          JOIN nonempty_crops v ON v.dataset=r.dataset AND v.crop=r.crop AND v.label=r.label
          WHERE r.metric = ? AND r.postprocessing_type = 'threshold'
            AND r.channel = ?
            AND ({pair_clause})
            {ds_clause}
          GROUP BY r.run, r.checkpoint, r.threshold, r.dataset, v.crop_group
        )
        SELECT run, checkpoint, threshold, SUM(sw) / SUM(w) AS score
        FROM per_group
        GROUP BY run, checkpoint, threshold
        ORDER BY run, checkpoint, threshold
        """
    conn = _connect(ctx.obj["db"])
    rows = conn.execute(
        sql, [*cte_args, ctx.obj["min_present"], metric, channel, *pair_args, *ds_args]
    ).fetchall()
    conn.close()
    if not rows:
        logger.warning("No data returned")
        return

    by_setup: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by_setup[(r["run"], r["checkpoint"])].append((r["threshold"], r["score"]))

    def _base_of(run: str) -> str:
        base, _ = _split_of(run)
        return base if base is not None else run

    def _is_finetune(run: str) -> bool:
        return "_ft-" in _base_of(run)

    def _colour_key(run: str) -> str:
        """Matched base/finetune runs share a key: 'run14_ft-run11' -> 'run11'."""
        base = _base_of(run)
        return base.split("_ft-", 1)[1] if "_ft-" in base else base

    # Colour groups: matched base/finetune pairs share a colour; order = first appearance.
    colour_order: list[str] = []
    for run, _ in parsed:
        k = _colour_key(run)
        if k not in colour_order:
            colour_order.append(k)
    cmap = plt.get_cmap("tab10")
    group_colours = {k: cmap(i % 10) for i, k in enumerate(colour_order)}

    base_iters: dict[str, dict[str, int]] = defaultdict(dict)
    for run, ckpt in parsed:
        _, split = _split_of(run)
        base_iters[_base_of(run)][split or ""] = _checkpoint_iter(ckpt)

    split_ls = {"val": "-", "all": "--"}
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    # For --color-by iteration: sequential cmap over the iteration range present.
    all_iters = [_checkpoint_iter(c) for _, c in parsed]
    it_norm = plt.Normalize(min(all_iters), max(all_iters)) if all_iters else None
    seq_cmap = plt.get_cmap("viridis")

    def _base_label(b: str) -> str:
        its = base_iters[b]
        if len(set(its.values())) == 1:
            return f"{b} @ {next(iter(its.values())) // 1000}k"
        return b + " (" + ", ".join(f"{s or '?'}@{v // 1000}k" for s, v in sorted(its.items())) + ")"

    def _plot_panel(ax, panel_setups: list[tuple[str, str]]) -> None:  # noqa: ANN001
        seen: dict[tuple[str, str], int] = defaultdict(int)
        bases_here: list[str] = []
        splits_here: set[str] = set()
        for run, ckpt in panel_setups:
            pts = by_setup.get((run, ckpt))
            if not pts:
                continue
            pts.sort()
            ts, scs = zip(*pts, strict=False)
            _, split = _split_of(run)
            base = _base_of(run)
            if base not in bases_here:
                bases_here.append(base)
            if split:
                splits_here.add(split)
            if color_by == "iteration":
                color = seq_cmap(it_norm(_checkpoint_iter(ckpt))) if it_norm is not None else None
                ls = "--" if _is_finetune(run) else "-"
                mk = "o"
            else:
                mk = markers[seen[(base, split or "")] % len(markers)]
                seen[(base, split or "")] += 1
                color = group_colours[_colour_key(run)]
                ls = split_ls.get(split or "", "-")
            ax.plot(ts, scs, marker=mk, linestyle=ls, color=color, markersize=4)
        ax.set_xlabel("threshold")
        ax.grid(True, alpha=0.3)
        if color_by == "iteration":
            handles = [
                plt.Line2D([0], [0], color="black", linestyle=("--" if "_ft-" in b else "-"), label=b)
                for b in bases_here
            ]
        else:
            handles = [
                plt.Line2D([0], [0], color=group_colours[_colour_key(b)], label=_base_label(b)) for b in bases_here
            ]
            if "val" in splits_here:
                handles.append(plt.Line2D([0], [0], color="black", linestyle="-", label="/val (held-out)"))
            if "all" in splits_here:
                handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="/all (incl. trained-on)"))
        ax.legend(handles=handles, loc="best", fontsize=8)

    ds_tag = f" ({ctx.obj['datasets']})" if ctx.obj.get("datasets") else ""
    title = f"{channel} — {metric} vs threshold{ds_tag}"
    scratch = [(r, c) for r, c in parsed if not _is_finetune(r)]
    finetuned = [(r, c) for r, c in parsed if _is_finetune(r)]

    if facet_finetune and scratch and finetuned:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, layout="constrained")
        _plot_panel(axes[0], scratch)
        axes[0].set_title("from scratch")
        axes[0].set_ylabel(metric)
        _plot_panel(axes[1], finetuned)
        axes[1].set_title("finetuned")
        fig.suptitle(title)
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        _plot_panel(ax, parsed)
        ax.set_ylabel(metric)
        ax.set_title(title)
        axes_list = [ax]

    if color_by == "iteration" and it_norm is not None:
        sm = plt.cm.ScalarMappable(norm=it_norm, cmap=seq_cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes_list, label="training iteration")
    out = ctx.obj["out_dir"] / (out_name or f"threshold_compare_{channel}_{metric}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    cli(obj={})
