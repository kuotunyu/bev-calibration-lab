"""The Pages assembler packages validated report and synthetic demo assets."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

DOCUMENTS = ("metrics", "intervals", "recovery", "timing", "exclusions")
FIGURES = ("recovery-by-fault-level", "bev-error-by-range")


def _inputs(root: Path) -> tuple[Path, Path]:
    claims = root / "claims.yaml"
    claims.write_bytes(b"claims\n")
    artifacts = root / "artifacts"
    artifacts.mkdir()
    for name in DOCUMENTS:
        (artifacts / f"{name}.json").write_bytes(f'{{"name":"{name}"}}\n'.encode())
    return claims, artifacts


def _formal_builder(
    claims: Path,
    artifacts: Path,
    output: Path,
    *,
    repository_root: Path,
    include_figures: bool,
) -> Path:
    assert claims.is_file() and artifacts.is_dir() and repository_root.is_dir()
    assert include_figures is True
    output.mkdir()
    (output / "evidence").mkdir()
    (output / "figures").mkdir()
    (output / "claims.yaml").write_bytes(claims.read_bytes())
    for name in DOCUMENTS:
        (output / "evidence" / f"{name}.json").write_bytes(
            (artifacts / f"{name}.json").read_bytes()
        )
    for name in FIGURES:
        (output / "figures" / f"{name}.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg"><title>{name}</title></svg>\n',
            encoding="utf-8",
            newline="\n",
        )
    index = output / "index.html"
    index.write_text(
        '<!doctype html><a href="claims.yaml">claims</a>'
        '<img src="figures/recovery-by-fault-level.svg">'
        '<a href="https://example.invalid">external</a><a href="#top">top</a></html>\n',
        encoding="utf-8",
        newline="\n",
    )
    return index


def _install_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bevcalib.report.site.build_formal_report", _formal_builder)
    monkeypatch.setattr(
        "bevcalib.report.site.build_explorer",
        lambda: "<!doctype html><html><p>synthetic</p><p>MIT</p></html>\n",
    )


def test_build_site_packages_fixed_assets_links_and_deterministic_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an omitted demo/link, asset, checksum, or unstable output byte."""
    from bevcalib.report.site import build_site

    _install_seams(monkeypatch)
    claims, artifacts = _inputs(tmp_path)
    before = {path: path.read_bytes() for path in (claims, *artifacts.iterdir())}

    first = build_site(claims, artifacts, tmp_path / "site-a", repository_root=tmp_path)
    second = build_site(claims, artifacts, tmp_path / "site-b", repository_root=tmp_path)

    assert first == tmp_path / "site-a" / "index.html"
    assert {path: path.read_bytes() for path in before} == before
    assert 'href="demo/calibration-explorer.html"' in first.read_text(encoding="utf-8")
    assert "synthetic" in (first.parent / "demo/calibration-explorer.html").read_text(
        encoding="utf-8"
    )
    inventory = json.loads((first.parent / "site-inventory.json").read_bytes())
    expected = {
        "claims.yaml",
        "demo/calibration-explorer.html",
        "index.html",
        *(f"evidence/{name}.json" for name in DOCUMENTS),
        *(f"figures/{name}.svg" for name in FIGURES),
    }
    assert inventory == {
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256((first.parent / relative).read_bytes()).hexdigest(),
            }
            for relative in sorted(expected)
        ],
        "schema_version": "bev-calibration-site-inventory/1",
    }
    for relative in expected | {"site-inventory.json"}:
        assert (first.parent / relative).read_bytes() == (second.parent / relative).read_bytes()


def test_build_site_refuses_existing_output_before_calling_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches accidental overwrite of an existing publication directory."""
    from bevcalib.report.site import build_site

    _install_seams(monkeypatch)
    claims, artifacts = _inputs(tmp_path)
    output = tmp_path / "site"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        build_site(claims, artifacts, output, repository_root=tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_build_site_refuses_missing_required_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a formal builder returning an incomplete Pages payload."""
    from bevcalib.report.site import build_site

    claims, artifacts = _inputs(tmp_path)

    def incomplete(*args, **kwargs):  # type: ignore[no-untyped-def]
        path = _formal_builder(*args, **kwargs)
        (path.parent / "figures/recovery-by-fault-level.svg").unlink()
        return path

    monkeypatch.setattr("bevcalib.report.site.build_formal_report", incomplete)
    monkeypatch.setattr("bevcalib.report.site.build_explorer", lambda: "demo")
    with pytest.raises(ValueError, match="required site asset"):
        build_site(claims, artifacts, tmp_path / "site", repository_root=tmp_path)


def test_build_site_refuses_extra_nested_file_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a deployed file that is absent from the checksum inventory."""
    from bevcalib.report.site import build_site

    claims, artifacts = _inputs(tmp_path)

    def extra(*args, **kwargs):  # type: ignore[no-untyped-def]
        path = _formal_builder(*args, **kwargs)
        nested = path.parent / "unexpected" / "payload.txt"
        nested.parent.mkdir()
        nested.write_text("untracked deployment payload", encoding="utf-8")
        return path

    monkeypatch.setattr("bevcalib.report.site.build_formal_report", extra)
    monkeypatch.setattr("bevcalib.report.site.build_explorer", lambda: "demo")
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="unexpected site asset"):
        build_site(claims, artifacts, output, repository_root=tmp_path)
    assert not (output / "site-inventory.json").exists()


def test_build_site_refuses_symlink_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a publication tree entry whose bytes are outside the tree."""
    from bevcalib.report.site import build_site

    claims, artifacts = _inputs(tmp_path)
    original = Path.is_symlink

    def linked(path: Path) -> bool:
        return path.name == "claims.yaml" or original(path)

    monkeypatch.setattr("bevcalib.report.site.build_formal_report", _formal_builder)
    monkeypatch.setattr("bevcalib.report.site.build_explorer", lambda: "demo")
    monkeypatch.setattr(Path, "is_symlink", linked)
    output = tmp_path / "site"
    with pytest.raises(ValueError, match=r"nonregular site asset: claims\.yaml"):
        build_site(claims, artifacts, output, repository_root=tmp_path)
    assert not (output / "site-inventory.json").exists()


def test_build_site_refuses_broken_declared_local_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an HTML link whose local target is absent from the package."""
    from bevcalib.report.site import build_site

    claims, artifacts = _inputs(tmp_path)

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        path = _formal_builder(*args, **kwargs)
        path.write_text('<a href="missing.html">missing</a></html>\n', encoding="utf-8")
        return path

    monkeypatch.setattr("bevcalib.report.site.build_formal_report", broken)
    monkeypatch.setattr("bevcalib.report.site.build_explorer", lambda: "demo")
    with pytest.raises(ValueError, match="broken local link"):
        build_site(claims, artifacts, tmp_path / "site", repository_root=tmp_path)


def test_build_site_refuses_formal_page_without_closing_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches insertion into a malformed or unexpectedly changed formal page."""
    from bevcalib.report.site import build_site

    claims, artifacts = _inputs(tmp_path)

    def malformed(*args, **kwargs):  # type: ignore[no-untyped-def]
        path = _formal_builder(*args, **kwargs)
        path.write_text("<!doctype html><p>unfinished", encoding="utf-8")
        return path

    monkeypatch.setattr("bevcalib.report.site.build_formal_report", malformed)
    monkeypatch.setattr("bevcalib.report.site.build_explorer", lambda: "demo")
    with pytest.raises(ValueError, match="closing html"):
        build_site(claims, artifacts, tmp_path / "site", repository_root=tmp_path)


def test_build_site_refuses_inputs_changed_during_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches concurrent replacement of any registry or formal input byte."""
    from bevcalib.report.site import build_site

    claims, artifacts = _inputs(tmp_path)

    def mutating(*args, **kwargs):  # type: ignore[no-untyped-def]
        path = _formal_builder(*args, **kwargs)
        claims.write_bytes(b"changed\n")
        return path

    monkeypatch.setattr("bevcalib.report.site.build_formal_report", mutating)
    monkeypatch.setattr("bevcalib.report.site.build_explorer", lambda: "demo")
    with pytest.raises(ValueError, match="input changed during site assembly"):
        build_site(claims, artifacts, tmp_path / "site", repository_root=tmp_path)


def test_main_builds_site_from_command_line_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a module CLI that does not route its three required paths."""
    from bevcalib.report import site

    _install_seams(monkeypatch)
    claims, artifacts = _inputs(tmp_path)
    output = tmp_path / "site"
    assert (
        site.main(
            [
                "--claims",
                str(claims),
                "--artifacts-dir",
                str(artifacts),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(output / "index.html")
    assert (output / "demo/calibration-explorer.html").is_file()


def test_module_entry_point_exits_after_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a missing executable module guard for ``python -m`` use."""
    import bevcalib.report.explorer as explorer
    import bevcalib.report.formal as formal
    import bevcalib.report.site as site

    claims, artifacts = _inputs(tmp_path)
    output = tmp_path / "site"
    monkeypatch.setattr(formal, "build_formal_report", _formal_builder)
    monkeypatch.setattr(explorer, "build_explorer", lambda: "demo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "site.py",
            "--claims",
            str(claims),
            "--artifacts-dir",
            str(artifacts),
            "--output-dir",
            str(output),
        ],
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(site.__file__, run_name="__main__")
    assert capsys.readouterr().out.strip() == str(output / "index.html")
