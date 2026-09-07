"""Explicit test injection and shell-only local runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
import yaml


@dataclass(frozen=True)
class Runtime:
    synthetic_fixture: bool = False
    model_factory: Callable[[], Any] | None = None
    repository_root: Path | None = None


def runtime(context: typer.Context) -> Runtime:
    return context.ensure_object(Runtime)


def dataroot() -> Path:
    value = os.environ.get("NUSCENES_ROOT")
    if not value:
        raise ValueError("NUSCENES_ROOT must name a local v1.0-trainval installation")
    return Path(value)


@contextmanager
def checked() -> Iterator[None]:
    try:
        yield
    except (ValueError, OSError, yaml.YAMLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
