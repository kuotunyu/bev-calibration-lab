"""Release verifier behavior tests using real archive fixtures."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import runpy
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

from bevcalib import release

PAYLOAD = b"Metadata-Version: 2.4\nName: bev-calibration-lab\nVersion: 1.0.0\n"


def timestamped_sdist(directory: Path, timestamp: int) -> Path:
    directory.mkdir()
    path = directory / "package-1.0.0.tar.gz"
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="source-name.tar", mode="wb", fileobj=raw, mtime=timestamp
        ) as zipped,
        tarfile.open(
            fileobj=zipped,
            mode="w",
            format=tarfile.PAX_FORMAT,
            pax_headers={
                "atime": f"{timestamp - 2}.5",
                "ctime": f"{timestamp - 1}.5",
                "mtime": f"{timestamp}.5",
                "global-comment": "kept",
            },
        ) as archive,
    ):
        root = tarfile.TarInfo("package-1.0.0")
        root.type = tarfile.DIRTYPE
        root.mode = 0o751
        root.uid = 12
        root.gid = 34
        root.uname = "builder"
        root.gname = "release"
        root.mtime = timestamp
        root.pax_headers = {
            "atime": f"{timestamp - 2}.25",
            "ctime": f"{timestamp - 1}.25",
            "mtime": f"{timestamp}.25",
            "comment": "kept",
        }
        archive.addfile(root)
        member = tarfile.TarInfo("package-1.0.0/payload.bin")
        member.mode = 0o640
        member.uid = 56
        member.gid = 78
        member.uname = "owner"
        member.gname = "group"
        member.mtime = timestamp + 1
        member.pax_headers = {
            "atime": f"{timestamp - 1}.75",
            "ctime": f"{timestamp}.75",
            "mtime": f"{timestamp + 1}.75",
            "comment": "also kept",
        }
        member.size = 4
        archive.addfile(member, io.BytesIO(b"data"))
    return path


def archive_snapshot(path: Path) -> list[tuple[object, ...]]:
    with tarfile.open(path, "r:gz") as archive:
        return [
            (
                member.name,
                member.type,
                member.mode,
                member.uid,
                member.gid,
                member.uname,
                member.gname,
                member.linkname,
                member.size,
                None if member.isdir() else archive.extractfile(member).read(),  # type: ignore[union-attr]
                {
                    key: value
                    for key, value in member.pax_headers.items()
                    if key not in {"atime", "ctime", "mtime"}
                },
            )
            for member in archive.getmembers()
        ]


def test_normalize_sdist_makes_timestamp_variants_byte_identical(tmp_path: Path) -> None:
    first = timestamped_sdist(tmp_path / "first", 1_700_000_000)
    second = timestamped_sdist(tmp_path / "second", 1_700_000_123)
    before = archive_snapshot(first)

    release.normalize_sdist(first.parent, 123456789)
    release.normalize_sdist(second.parent, 123456789)

    assert first.read_bytes() == second.read_bytes()
    assert archive_snapshot(first) == before
    compressed = first.read_bytes()
    assert int.from_bytes(compressed[4:8], "little") == 123456789
    assert compressed[3] & 0x08 == 0
    with tarfile.open(first, "r:gz") as archive:
        assert [member.mtime for member in archive.getmembers()] == [123456789, 123456789]
        assert all(
            {"atime", "ctime", "mtime"}.isdisjoint(member.pax_headers)
            for member in archive.getmembers()
        )
        assert archive.pax_headers == {"global-comment": "kept"}


def test_normalize_sdist_replace_failure_preserves_source_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = timestamped_sdist(tmp_path / "dist", 1_700_000_000)
    original = path.read_bytes()

    def fail_replace(candidate: Path, target: Path) -> Path:
        raise OSError("replace denied")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(ValueError, match="replace denied"):
        release.normalize_sdist(path.parent, 42)

    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_normalize_sdist_temporary_creation_failure_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = timestamped_sdist(tmp_path / "dist", 1_700_000_000)
    original = path.read_bytes()

    def fail_temporary(*args: object, **kwargs: object) -> object:
        raise OSError("temporary denied")

    monkeypatch.setattr(release.tempfile, "NamedTemporaryFile", fail_temporary)
    with pytest.raises(ValueError, match="temporary denied"):
        release.normalize_sdist(path.parent, 42)

    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("epoch", [-1, 2**32, True])
def test_normalize_sdist_rejects_invalid_epoch_before_overwrite(tmp_path: Path, epoch: int) -> None:
    path = timestamped_sdist(tmp_path / "dist", 1_700_000_000)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="epoch"):
        release.normalize_sdist(path.parent, epoch)
    assert path.read_bytes() == original


@pytest.mark.parametrize("count", [0, 2])
def test_normalize_sdist_requires_exactly_one_archive(tmp_path: Path, count: int) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    for index in range(count):
        path = timestamped_sdist(tmp_path / f"source-{index}", 1_700_000_000 + index)
        path.replace(dist / f"package-{index}.tar.gz")
    with pytest.raises(ValueError, match="exactly one"):
        release.normalize_sdist(dist, 0)


@pytest.mark.parametrize(
    "defect", ["unreadable-dir", "symlink", "nonregular", "unsafe", "member-type", "malformed"]
)
def test_normalize_sdist_rejects_unsafe_input_before_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    path = dist / "package.tar.gz"
    if defect == "unreadable-dir":
        path.write_bytes(b"untouched")

        def unreadable(candidate: Path, pattern: str) -> object:
            raise OSError("denied")

        monkeypatch.setattr(Path, "glob", unreadable)
    elif defect == "malformed":
        path.write_bytes(b"bad")
    elif defect == "nonregular":
        path.mkdir()
    else:
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("../escape" if defect == "unsafe" else "package/link")
            if defect == "member-type":
                member.type = tarfile.SYMTYPE
                member.linkname = "target"
            else:
                member.size = 1
            archive.addfile(member, None if defect == "member-type" else io.BytesIO(b"x"))
    original = path.read_bytes() if path.is_file() else None
    if defect == "symlink":
        monkeypatch.setattr(Path, "is_symlink", lambda candidate: candidate == path)
    with pytest.raises(
        ValueError, match=r"distribution directory|symlink|regular|unsafe|type|malformed"
    ):
        release.normalize_sdist(dist, 0)
    if original is not None:
        assert path.read_bytes() == original


def test_normalize_sdist_cli_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = timestamped_sdist(tmp_path / "dist", 1_700_000_000)
    assert release.main(["normalize-sdist", "--dist-dir", str(path.parent), "--epoch", "42"]) == 0
    assert "normalized:" in capsys.readouterr().out
    assert release.main(["normalize-sdist", "--dist-dir", str(path.parent), "--epoch", "-1"]) == 1
    assert "epoch" in capsys.readouterr().err


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname="bev-calibration-lab"\nversion="1.0.0"\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "bev_calibration_lab-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("bev_calibration_lab-1.0.0.dist-info/METADATA", PAYLOAD)
        archive.writestr("bevcalib/__init__.py", "")
    sdist = dist / "bev_calibration_lab-1.0.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("bev_calibration_lab-1.0.0/PKG-INFO")
        member.size = len(PAYLOAD)
        archive.addfile(member, io.BytesIO(PAYLOAD))
        directory = tarfile.TarInfo("bev_calibration_lab-1.0.0/src")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
    return project, wheel, sdist


def verify(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    release.verify_release(project, tmp_path / "dist", "v1.0.0", "1.0.0", "1.0.0")


def test_valid_release_writes_sorted_portable_checksums_and_accepts_matching_manifest(
    tmp_path: Path,
) -> None:
    _, wheel, sdist = fixture(tmp_path)
    verify(tmp_path)
    expected = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted((wheel, sdist), key=lambda path: path.name)
    )
    assert (tmp_path / "dist" / "SHA256SUMS").read_text() == expected
    verify(tmp_path)


@pytest.mark.parametrize(
    ("tag", "installed", "runtime"),
    [
        ("1.0.0", "1.0.0", "1.0.0"),
        ("v1.0", "1.0.0", "1.0.0"),
        ("v1.0.0", "0.1.0", "1.0.0"),
        ("v1.0.0", "1.0.0", "0.1.0"),
    ],
)
def test_identity_mismatch_fails_before_manifest(
    tmp_path: Path, tag: str, installed: str, runtime: str
) -> None:
    project, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        release.verify_release(project, tmp_path / "dist", tag, installed, runtime)
    assert not (tmp_path / "dist" / "SHA256SUMS").exists()


@pytest.mark.parametrize(("name", "version"), [("other", "1.0.0"), ("bev-calibration-lab", "one")])
def test_project_identity_is_validated(tmp_path: Path, name: str, version: str) -> None:
    project, _, _ = fixture(tmp_path)
    project.write_text(f'[project]\nname="{name}"\nversion="{version}"\n')
    with pytest.raises(ValueError, match="project"):
        release.verify_release(project, tmp_path / "dist", "v1.0.0", "1.0.0", "1.0.0")


@pytest.mark.parametrize("defect", ["missing", "wrong", "extra"])
def test_distribution_set_is_exact(tmp_path: Path, defect: str) -> None:
    _, wheel, _ = fixture(tmp_path)
    if defect == "missing":
        wheel.unlink()
    elif defect == "wrong":
        wheel.rename(wheel.parent / "wrong-1.0.0-py3-none-any.whl")
    else:
        (wheel.parent / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="distribution"):
        verify(tmp_path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong-name", "wrong-version"])
def test_archive_metadata_is_canonical_unique_and_matching(
    tmp_path: Path, kind: str, defect: str
) -> None:
    _, wheel, sdist = fixture(tmp_path)
    payload = (
        PAYLOAD.replace(b"bev-calibration-lab", b"other")
        if defect == "wrong-name"
        else PAYLOAD.replace(b"1.0.0", b"9.9.9")
        if defect == "wrong-version"
        else PAYLOAD
    )
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "w") as archive:
            if defect != "missing":
                archive.writestr("bev_calibration_lab-1.0.0.dist-info/METADATA", payload)
                if defect == "duplicate":
                    archive.writestr("other.dist-info/METADATA", payload)
    else:
        with tarfile.open(sdist, "w:gz") as archive:
            if defect != "missing":
                member = tarfile.TarInfo("bev_calibration_lab-1.0.0/PKG-INFO")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                if defect == "duplicate":
                    member = tarfile.TarInfo("bev_calibration_lab-1.0.0/PKG-INFO")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="metadata"):
        verify(tmp_path)


def test_sdist_accepts_nested_egg_info_beside_canonical_metadata(tmp_path: Path) -> None:
    _, _, sdist = fixture(tmp_path)
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "bev_calibration_lab-1.0.0/PKG-INFO",
            "bev_calibration_lab-1.0.0/src/bev_calibration_lab.egg-info/PKG-INFO",
        ):
            member = tarfile.TarInfo(name)
            member.size = len(PAYLOAD)
            archive.addfile(member, io.BytesIO(PAYLOAD))
    verify(tmp_path)


def test_sdist_rejects_duplicate_exact_canonical_metadata(tmp_path: Path) -> None:
    _, _, sdist = fixture(tmp_path)
    with tarfile.open(sdist, "w:gz") as archive:
        for _ in range(2):
            member = tarfile.TarInfo("bev_calibration_lab-1.0.0/PKG-INFO")
            member.size = len(PAYLOAD)
            archive.addfile(member, io.BytesIO(PAYLOAD))
    with pytest.raises(ValueError, match="canonical metadata"):
        verify(tmp_path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "safe\\..\\escape"])
def test_unsafe_paths_are_refused(tmp_path: Path, kind: str, name: str) -> None:
    _, wheel, sdist = fixture(tmp_path)
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(name, "x")
    else:
        with tarfile.open(sdist, "w:gz") as archive:
            metadata = tarfile.TarInfo("bev_calibration_lab-1.0.0/PKG-INFO")
            metadata.size = len(PAYLOAD)
            archive.addfile(metadata, io.BytesIO(PAYLOAD))
            member = tarfile.TarInfo(name)
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        verify(tmp_path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_unsafe_member_types_are_refused(tmp_path: Path, kind: str) -> None:
    _, wheel, sdist = fixture(tmp_path)
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "a") as archive:
            member = zipfile.ZipInfo("link")
            member.create_system = 3
            member.external_attr = 0o120777 << 16
            archive.writestr(member, "target")
    else:
        with tarfile.open(sdist, "w:gz") as archive:
            metadata = tarfile.TarInfo("bev_calibration_lab-1.0.0/PKG-INFO")
            metadata.size = len(PAYLOAD)
            archive.addfile(metadata, io.BytesIO(PAYLOAD))
            tar_member = tarfile.TarInfo("bev_calibration_lab-1.0.0/link")
            tar_member.type = tarfile.SYMTYPE
            tar_member.linkname = "x"
            archive.addfile(tar_member)
    with pytest.raises(ValueError, match="type"):
        verify(tmp_path)


def test_wheel_fifo_member_is_refused(tmp_path: Path) -> None:
    _, wheel, _ = fixture(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        fifo = zipfile.ZipInfo("fifo")
        fifo.create_system = 3
        fifo.external_attr = 0o010644 << 16
        archive.writestr(fifo, "")
    with pytest.raises(ValueError, match="type"):
        verify(tmp_path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_malformed_archives_are_validation_errors(tmp_path: Path, kind: str) -> None:
    _, wheel, sdist = fixture(tmp_path)
    (wheel if kind == "wheel" else sdist).write_bytes(b"bad")
    with pytest.raises(ValueError, match="malformed"):
        verify(tmp_path)


def test_distribution_symlink_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, wheel, _ = fixture(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == wheel or original(path))
    with pytest.raises(ValueError, match="symlink"):
        verify(tmp_path)


@pytest.mark.parametrize("contents", [b"stale", b"\xff"])
def test_stale_or_malformed_manifest_is_refused(tmp_path: Path, contents: bytes) -> None:
    _, wheel, _ = fixture(tmp_path)
    manifest = wheel.parent / "SHA256SUMS"
    manifest.write_text("stale")
    manifest.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)
    assert manifest.read_bytes() == contents


@pytest.mark.parametrize("state", ["matching", "stale", "dangling"])
def test_checksum_manifest_symlinks_are_refused_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    _, wheel, _ = fixture(tmp_path)
    manifest = wheel.parent / "SHA256SUMS"
    target = tmp_path / "outside.txt"
    if state != "dangling":
        target.write_text("unchanged")
        manifest.write_text("matching" if state == "matching" else "stale")
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == manifest)
    with pytest.raises(ValueError, match="checksum manifest"):
        verify(tmp_path)
    assert target.read_text() == "unchanged" if target.exists() else not target.exists()


def test_checksum_manifest_creation_refuses_a_racing_new_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, wheel, _ = fixture(tmp_path)
    manifest = wheel.parent / "SHA256SUMS"
    original_open = Path.open

    def racing_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == manifest and args and args[0] == "x":
            raise FileExistsError(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(ValueError, match="appeared during creation"):
        verify(tmp_path)


def test_nonregular_checksum_manifest_is_refused(tmp_path: Path) -> None:
    _, wheel, _ = fixture(tmp_path)
    manifest = wheel.parent / "SHA256SUMS"
    manifest.mkdir()
    with pytest.raises(ValueError, match="regular non-symlink"):
        verify(tmp_path)


def test_backend_exact_pins_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text('[build-system]\nrequires=["setuptools==80.9.0", "wheel==0.45.1"]\n')
    monkeypatch.setattr(
        importlib.metadata, "version", {"setuptools": "80.9.0", "wheel": "0.45.1"}.__getitem__
    )
    release.verify_backend(project)
    project.write_text('[build-system]\nrequires=["setuptools>=80"]\n')
    with pytest.raises(ValueError, match="exact"):
        release.verify_backend(project)
    project.write_text('[build-system]\nrequires=["setuptools==80.9.0"]\n')
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0")
    with pytest.raises(ValueError, match="does not match"):
        release.verify_backend(project)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )
    with pytest.raises(ValueError, match="unavailable"):
        release.verify_backend(project)


@pytest.mark.parametrize("text", ["", "[build-system]\nrequires=[]\n"])
def test_backend_requires_a_nonempty_requirements_list(tmp_path: Path, text: str) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(text)
    with pytest.raises(ValueError, match=r"requirements|exact"):
        release.verify_backend(project)


def test_unreadable_or_incomplete_project_and_missing_dist_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="read project"):
        release.verify_release(tmp_path / "missing", tmp_path, "v1.0.0", "1.0.0", "1.0.0")
    project = tmp_path / "project.toml"
    project.write_text("[project]\nname='bev-calibration-lab'\n")
    with pytest.raises(ValueError, match="name and version"):
        release.verify_release(project, tmp_path, "v1.0.0", "1.0.0", "1.0.0")
    project.write_text("[project]\nname='bev-calibration-lab'\nversion='1.0.0'\n")
    with pytest.raises(ValueError, match="distribution directory"):
        release.verify_release(project, tmp_path / "missing", "v1.0.0", "1.0.0", "1.0.0")


@pytest.mark.parametrize(("name", "expected"), [("", False), ("safe/path", True)])
def test_member_path_boundary(name: str, expected: bool) -> None:
    assert release._safe(name) is expected


def test_cli_success_failure_and_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, _, _ = fixture(tmp_path)
    project.write_text(project.read_text() + '[build-system]\nrequires=["wheel==0.45.1"]\n')
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.45.1" if name == "wheel" else "1.0.0"
    )
    monkeypatch.setattr(release, "__version__", "1.0.0")
    assert release.main(["backend", "--project", str(project)]) == 0
    assert (
        release.main(
            [
                "verify",
                "--project",
                str(project),
                "--dist-dir",
                str(tmp_path / "dist"),
                "--tag",
                "v1.0.0",
            ]
        )
        == 0
    )
    assert (
        release.main(
            [
                "verify",
                "--project",
                "missing",
                "--dist-dir",
                str(tmp_path / "dist"),
                "--tag",
                "v1.0.0",
            ]
        )
        == 1
    )
    assert "release check failed" in capsys.readouterr().err
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release",
            "verify",
            "--project",
            "missing",
            "--dist-dir",
            str(tmp_path / "dist"),
            "--tag",
            "v1.0.0",
        ],
    )
    assert release.__file__ is not None
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(release.__file__, run_name="__main__")
    assert stopped.value.code == 1
