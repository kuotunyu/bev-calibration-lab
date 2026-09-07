"""Freeze verified native records with explicit shortage diagnostics."""

from pathlib import Path
from typing import Annotated

import typer

from bevcalib.cli.data import Version
from bevcalib.cli.runtime import checked

app = typer.Typer(no_args_is_help=True)


@app.command()
def freeze(
    dataroot: Annotated[Path, typer.Option()],
    version: Annotated[Version, typer.Option()],
    protocol: Annotated[Path, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    from bevcalib.cohort.manifest import save_manifest
    from bevcalib.cohort.protocol import resolve_protocol
    from bevcalib.cohort.splits import freeze_scene_cohort
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    with checked():
        if version != Version.trainval:
            raise ValueError("formal cohort freeze requires v1.0-trainval")
        resolved = resolve_protocol(protocol)
        installation = resolve_installation(dataroot, version.value)
        installation.preflight()
        manifests = freeze_scene_cohort(
            installation.scene_records(),
            protocol_hash=resolved.protocol_hash,
            dataset_version=resolved.dataset_version,
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        for role, manifest in manifests.items():
            save_manifest(manifest, output_dir / f"{role}.json")
            shortage = sum(item.requested - item.selected for item in manifest.allocation)
            typer.echo(f"{role}: selected={len(manifest.scenes)} shortage={shortage}")
