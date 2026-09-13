"""Fail-closed validation for release identity, archives, and checksums."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import io
import re
import stat
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from bevcalib import __version__

PACKAGE_NAME = "bev-calibration-lab"
_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)")
_TAG = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)")


def _config(project: Path) -> dict[str, object]:
    try:
        return tomllib.loads(project.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read project metadata: {error}") from error


def verify_backend(project: Path) -> None:
    """Require every build dependency to be exactly pinned and installed."""
    try:
        requirements = _config(project)["build-system"]["requires"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("project build-system requirements are missing") from error
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("backend dependencies require exact version pins")
    for requirement in requirements:
        match = _PIN.fullmatch(requirement) if isinstance(requirement, str) else None
        if match is None:
            raise ValueError("backend dependencies require exact version pins")
        name, expected = match.groups()
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"installed backend is unavailable: {name}") from error
        if installed != expected:
            raise ValueError(f"installed backend {name}=={installed} does not match {requirement}")
        print(f"backend: {requirement}")


def _safe(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return (
        bool(normalized)
        and not path.is_absolute()
        and not re.match(r"^[A-Za-z]:/", normalized)
        and ".." not in path.parts
    )


def _check_metadata(payload: bytes, name: str, version: str) -> None:
    message = BytesParser().parsebytes(payload)
    if message.get_all("Name") != [name] or message.get_all("Version") != [version]:
        raise ValueError("archive metadata name or version does not match project")


def _wheel(path: Path, stem: str, name: str, version: str) -> None:
    expected = f"{stem}.dist-info/METADATA"
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if any(not _safe(item.filename) for item in members):
                raise ValueError("wheel contains an unsafe archive member path")
            if any(
                item.create_system == 3
                and (mode := stat.S_IFMT(item.external_attr >> 16))
                and mode not in {stat.S_IFREG, stat.S_IFDIR}
                for item in members
            ):
                raise ValueError("wheel contains an unsafe archive member type")
            found = [item for item in members if item.filename.endswith(".dist-info/METADATA")]
            if len(found) != 1 or found[0].filename != expected or found[0].is_dir():
                raise ValueError("wheel metadata must be one canonical metadata file")
            _check_metadata(archive.read(found[0]), name, version)
    except (zipfile.BadZipFile, OSError, RuntimeError) as error:
        raise ValueError(f"wheel is malformed: {error}") from error


def _sdist(path: Path, stem: str, name: str, version: str) -> None:
    expected = f"{stem}/PKG-INFO"
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if any(not _safe(item.name) for item in members):
                raise ValueError("sdist contains an unsafe archive member path")
            if any(not (item.isfile() or item.isdir()) for item in members):
                raise ValueError("sdist contains an unsafe archive member type")
            found = [item for item in members if item.name == expected]
            if len(found) != 1 or not found[0].isfile():
                raise ValueError("sdist metadata must be one canonical metadata file")
            stream = archive.extractfile(found[0])
            assert stream is not None
            _check_metadata(stream.read(), name, version)
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"sdist is malformed: {error}") from error


def _write_sdist_temporary(path: Path, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    return temporary


def normalize_sdist(dist_dir: Path, epoch: int) -> None:
    """Rewrite one safe sdist with deterministic gzip, tar, and PAX timestamps."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch <= 2**32 - 1:
        raise ValueError("epoch must be an integer in the unsigned 32-bit range")
    try:
        candidates = list(dist_dir.glob("*.tar.gz"))
    except OSError as error:
        raise ValueError(f"cannot read distribution directory: {error}") from error
    if len(candidates) != 1:
        raise ValueError("distribution directory must contain exactly one sdist")
    path = candidates[0]
    if path.is_symlink():
        raise ValueError("sdist must not be a symlink")
    if not path.is_file():
        raise ValueError("sdist must be a regular file")

    try:
        with tarfile.open(path, "r:gz") as source:
            members = source.getmembers()
            if any(not _safe(member.name) for member in members):
                raise ValueError("sdist contains an unsafe archive member path")
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise ValueError("sdist contains an unsafe archive member type")
            payloads: list[bytes | None] = []
            for member in members:
                stream = source.extractfile(member) if member.isfile() else None
                assert stream is not None if member.isfile() else stream is None
                payloads.append(None if stream is None else stream.read())
            global_pax = dict(source.pax_headers)
        for field in ("atime", "ctime", "mtime"):
            global_pax.pop(field, None)
        output = io.BytesIO()
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=epoch) as compressed,
            tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT, pax_headers=global_pax
            ) as target,
        ):
            for member, payload in zip(members, payloads, strict=True):
                normalized = copy.copy(member)
                normalized.mtime = epoch
                normalized.pax_headers = dict(member.pax_headers)
                for field in ("atime", "ctime", "mtime"):
                    normalized.pax_headers.pop(field, None)
                target.addfile(normalized, None if payload is None else io.BytesIO(payload))
    except (tarfile.TarError, OSError, EOFError) as error:
        raise ValueError(f"sdist is malformed: {error}") from error
    try:
        temporary = _write_sdist_temporary(path, output.getvalue())
    except OSError as error:
        raise ValueError(f"cannot atomically replace sdist: {error}") from error
    try:
        temporary.replace(path)
    except OSError as error:
        raise ValueError(f"cannot atomically replace sdist: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    print(f"normalized: {path.name} ({epoch})")


def verify_release(
    project: Path, dist_dir: Path, tag: str, installed_version: str, runtime_version: str
) -> None:
    """Validate every release input before creating portable checksums."""
    try:
        metadata = _config(project)["project"]
        name, version = metadata["name"], metadata["version"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("project name and version are missing") from error
    if (
        name != PACKAGE_NAME
        or not isinstance(version, str)
        or _TAG.fullmatch(f"v{version}") is None
    ):
        raise ValueError("project name or version is invalid")
    match = _TAG.fullmatch(tag)
    if match is None or (match.group(1), installed_version, runtime_version) != (
        version,
        version,
        version,
    ):
        raise ValueError(
            "release identity disagrees across project, tag, installed, or runtime version"
        )
    stem = f"bev_calibration_lab-{version}"
    names = {f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"}
    try:
        entries = list(dist_dir.iterdir())
    except OSError as error:
        raise ValueError(f"cannot read distribution directory: {error}") from error
    if any(entry.name not in names | {"SHA256SUMS"} for entry in entries):
        raise ValueError("distribution directory contains unexpected members")
    artifacts = [dist_dir / item for item in sorted(names)]
    if any(not path.exists() or not path.is_file() for path in artifacts):
        raise ValueError("distribution set is missing or has wrong canonical names")
    if any(path.is_symlink() for path in artifacts):
        raise ValueError("distribution inputs must not be symlinks")
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    sdist = next(path for path in artifacts if path.name.endswith(".tar.gz"))
    _wheel(wheel, stem, name, version)
    _sdist(sdist, stem, name, version)
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in artifacts
    )
    manifest = dist_dir / "SHA256SUMS"
    if manifest.is_symlink():
        raise ValueError("checksum manifest must be a regular non-symlink file")
    if manifest.exists():
        if not manifest.is_file():
            raise ValueError("checksum manifest must be a regular non-symlink file")
        try:
            existing = manifest.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"existing checksum manifest is malformed: {error}") from error
        if existing != checksums:
            raise ValueError("existing checksum manifest is stale or malformed")
    else:
        try:
            with manifest.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(checksums)
        except FileExistsError as error:
            raise ValueError("checksum manifest appeared during creation") from error
    print(f"verified: {name} {version} ({tag})")
    print(checksums, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    verify = commands.add_parser("verify")
    verify.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    verify.add_argument("--dist-dir", type=Path, required=True)
    verify.add_argument("--tag", required=True)
    normalize = commands.add_parser("normalize-sdist")
    normalize.add_argument("--dist-dir", type=Path, required=True)
    normalize.add_argument("--epoch", type=int, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "backend":
            verify_backend(arguments.project)
        elif arguments.command == "verify":
            verify_release(
                arguments.project,
                arguments.dist_dir,
                arguments.tag,
                importlib.metadata.version(PACKAGE_NAME),
                __version__,
            )
        else:
            normalize_sdist(arguments.dist_dir, arguments.epoch)
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        print(f"release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
