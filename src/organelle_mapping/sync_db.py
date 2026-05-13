"""Snapshot a local SQLite results DB to another path (e.g. NFS) via VACUUM INTO.

The destination is a self-contained .db file in default rollback-journal mode,
with no ``-wal``/``-shm`` sidecars — safe to place on NFS. The source can have
active writers during the snapshot.
"""

import logging
from pathlib import Path

import click

from organelle_mapping.database import init_database, vacuum_into

logger = logging.getLogger("organelle_mapping.sync_db")


@click.command()
@click.option("--src", "src_url", type=str, required=True, help="Source SQLite URL (e.g. 'sqlite:///local/results.db').")
@click.option(
    "--dest",
    "dest_path",
    type=click.Path(),
    required=True,
    help="Destination .db file path (e.g. '/nrs/.../results.db').",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Replace the destination if it already exists.",
)
def cli(src_url: str, dest_path: str, *, overwrite: bool) -> None:
    """Snapshot a SQLite DB at ``--src`` to ``--dest`` via VACUUM INTO."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    dest = Path(dest_path)
    if dest.exists():
        if not overwrite:
            raise click.BadParameter(f"destination {dest} already exists; pass --overwrite to replace it")
        dest.unlink()

    engine = init_database(src_url)
    logger.info("Snapshotting %s -> %s", src_url, dest)
    vacuum_into(engine, str(dest))
    logger.info("Snapshot complete: %s", dest)


if __name__ == "__main__":
    cli()
