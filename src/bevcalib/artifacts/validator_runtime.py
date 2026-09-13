"""Compare this validator process and checkout with separately frozen expectations."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from bevcalib.cohort.manifest import Digest

SOURCE_ROOTS = ("src", "configs", "schemas", "scripts", ".agents/skills")
SOURCE_FILES = ("pyproject.toml", "uv.lock")


class SourceSnapshot(TypedDict):
    files: dict[str, str]
    sha256: str


class ExpectedValidator(BaseModel):
    """Reviewed validator identity; distinct from the original raw-run producer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-validator-runtime-expectations/v1"]
    python_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: Digest
    source_sha256: Digest


def source_snapshot(root: Path) -> SourceSnapshot:
    """Hash actual bytes, including untracked/ignored inputs, excluding bytecode caches.

    This is a fixed source/configuration scope, not an installed-dependency audit.
    Generate the expected digest before the run from the separately reviewed copy.
    """
    if not (root / "src").is_dir():
        raise ValueError("validator source directory is missing")
    paths = []
    for name in SOURCE_ROOTS:
        parts = Path(name).parts
        for depth in range(1, len(parts) + 1):
            component = root.joinpath(*parts[:depth])
            if component.is_symlink() or component.is_junction():
                raise ValueError(
                    f"validator source symlink or junction is not allowed: {component}"
                )
        directory = root / name
        paths.append(directory)
        paths.extend(directory.rglob("*"))
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"required source file is missing: {name}")
        paths.append(path)
    files = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink() or path.is_junction():
            raise ValueError(
                f"validator source symlink or junction is not allowed: {relative.as_posix()}"
            )
        if path.is_file():
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(canonical).hexdigest()}


def validate_validator_runtime(expected: ExpectedValidator) -> dict[str, str]:
    """Check this imported module's checkout, not a caller-supplied lookalike path.

    Returns no success on mismatches. Matching source bytes do not attest to loaded
    third-party packages, monkeypatches, a hostile process, or historical run state.
    """
    root = Path(__file__).resolve().parents[3]
    try:
        identity = (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("validator Git checkout identity unavailable") from exc
    if len(identity) != 2 or Path(identity[0]).resolve() != root:
        raise ValueError("validator Git checkout identity differs from imported source")
    source = source_snapshot(root)
    actual = {
        "python_version": platform.python_version(),
        "commit": identity[1],
        "lock_sha256": source["files"]["uv.lock"],
        "source_sha256": source["sha256"],
        "executable": str(Path(sys.executable).resolve()),
    }
    for field in ("python_version", "commit", "lock_sha256", "source_sha256"):
        if actual[field] != getattr(expected, field):
            raise ValueError(f"frozen validator runtime expectation mismatch: {field}")
    return actual
