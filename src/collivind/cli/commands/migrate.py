"""`collivind migrate`: move all data between modes, ids and graph intact."""

import sys
from dataclasses import replace
from typing import NoReturn

import click

from collivind.config import CollivindConfig, load_config
from collivind.engine.migrate import Stores, embedding_space, migrate
from collivind.storage.factory import _resolve_provider, create_all_backends

# Not "remote": it shares the [qdrant]/[neo4j] sections with docker, so with
# one config.toml docker<->remote would migrate a store onto itself.
MODES = click.Choice(["docker", "embedded"])


def mode_config(base: CollivindConfig, mode: str) -> CollivindConfig:
    """`base` switched to `mode`, with per-backend provider overrides cleared.

    An override outranks the mode, so leaving one in place could resolve
    `--from` and `--to` to the very same store.
    """
    return replace(
        base,
        mode=mode,
        qdrant=replace(base.qdrant, provider=""),
        neo4j=replace(base.neo4j, provider=""),
        embeddings=replace(base.embeddings, provider=""),
    )


def _fail(message: str) -> NoReturn:
    click.secho(message, fg="red", err=True)
    sys.exit(1)


def _open(config: CollivindConfig, role: str) -> Stores:
    try:
        return Stores(*create_all_backends(config))
    except Exception as e:
        _fail(f"{role} mode {config.mode!r} is not reachable; nothing was written. {e}")


def _space(config: CollivindConfig, stores: Stores):
    provider = _resolve_provider(config, "embeddings")
    return embedding_space(provider, config.embeddings.model, stores.embedder.dimension)


@click.command(name="migrate")
@click.option("--from", "from_mode", required=True, type=MODES, help="Mode to copy from (never modified)")
@click.option("--to", "to_mode", required=True, type=MODES, help="Mode to copy into")
@click.option(
    "--dry-run", is_flag=True, help="Count what would be copied; writes no records (stores may create empty schema)"
)
def migrate_cmd(from_mode, to_mode, dry_run):
    """Copy ALL memories, entities, relationships and vectors between modes.

    Ids are preserved, so entity links, version chains and memory-to-memory
    edges survive. Copy-only and idempotent: re-running upserts by id.
    """
    if from_mode == to_mode:
        _fail(f"--from and --to are both {from_mode!r}; nothing to migrate.")
    base = load_config()
    src_cfg, dst_cfg = mode_config(base, from_mode), mode_config(base, to_mode)
    src = _open(src_cfg, "Source")
    dst = _open(dst_cfg, "Target")
    try:
        src_space, dst_space = _space(src_cfg, src), _space(dst_cfg, dst)
        reembed = src_space != dst_space
        click.echo(f"Migrating {from_mode} -> {to_mode}{' (dry run)' if dry_run else ''}")
        if reembed:
            click.secho(
                f"Embedding models differ ({src_space} vs {dst_space}): re-embedding every vector "
                f"with the target's provider.",
                fg="yellow",
            )
        else:
            click.echo(f"Vectors copied as stored (same embedding space: {dst_space[1]}, {dst_space[2]}d).")
        report = migrate(src, dst, reembed=reembed, dry_run=dry_run)
    finally:
        for stores in (src, dst):
            stores.vector.close()
            stores.graph.close()

    for edge in report.dangling:
        click.secho(
            f"skipped dangling edge {edge.id} ({edge.source_id} -{edge.type.value}-> {edge.target_id}): "
            f"an endpoint does not exist in the source",
            fg="yellow",
        )
    click.echo(f"{'kind':<14}{'source':>8}{'target':>8}{'missing':>9}")
    for c in report.counts:
        click.echo(f"{c.kind:<14}{c.source:>8}{c.target:>8}{c.missing:>9}")
    if dry_run:
        click.secho("Dry run: nothing written. 'missing' is what a real run would add.", fg="green")
    elif report.ok:
        click.secho(f"Every source record is present in {to_mode}. Source left untouched.", fg="green")
    else:
        _fail("Verification failed: some source records are missing from the target (see 'missing').")
