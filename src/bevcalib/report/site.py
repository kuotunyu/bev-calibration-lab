"""Assemble the Pages site: a small landing page, the full formal report and the explorer.

The landing page at the site root states the result and links onward. The complete
formal report lives at `evidence/index.html`, next to the five source documents it
copies; the claims registry, the formal figures and the explorer keep the URLs they
had when the report itself was the root page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bevcalib.report.explorer import build_explorer
from bevcalib.report.formal import build_formal_report
from bevcalib.report.landing import build_overview

_DOCUMENTS = ("metrics", "intervals", "recovery", "timing", "exclusions")
_FIGURES = ("recovery-by-fault-level", "bev-error-by-range")
_REPORT = "evidence/index.html"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_snapshot(claims_path: Path, artifacts_dir: Path) -> dict[Path, str]:
    paths = (claims_path, *(artifacts_dir / f"{name}.json" for name in _DOCUMENTS))
    return {path: _sha256(path) for path in paths}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        names = {"href"} if tag == "a" else {"src"} if tag in {"img", "script"} else set()
        self.targets.extend(value for name, value in attrs if name in names and value is not None)


def _validate_local_links(site_root: Path, pages: Sequence[Path]) -> None:
    root = site_root.resolve()
    for page in pages:
        parser = _LinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for reference in parser.targets:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                raise ValueError(f"broken local link in {page.relative_to(site_root)}: {reference}")


def _required_assets(output_dir: Path) -> tuple[Path, ...]:
    relative = (
        "analysis/operating-envelope.json",
        "analysis/operating-envelope.svg",
        "claims.yaml",
        "demo/calibration-explorer.html",
        "index.html",
        _REPORT,
        *(f"evidence/{name}.json" for name in _DOCUMENTS),
        *(f"figures/{name}.svg" for name in _FIGURES),
    )
    paths = tuple(output_dir / name for name in relative)
    entries = tuple(output_dir.rglob("*"))
    invalid = [
        path.relative_to(output_dir).as_posix()
        for path in entries
        if path.is_symlink() or (not path.is_file() and not path.is_dir())
    ]
    if invalid:
        raise ValueError("nonregular site asset: " + ", ".join(invalid))
    actual = {path.relative_to(output_dir).as_posix() for path in entries if path.is_file()}
    expected = set(relative)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError("missing required site asset: " + ", ".join(missing))
    extra = sorted(actual - expected)
    if extra:
        raise ValueError("unexpected site asset: " + ", ".join(extra))
    return paths


def _write_inventory(output_dir: Path, assets: Sequence[Path]) -> Path:
    files = [
        {"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path)}
        for path in sorted(assets, key=lambda item: item.relative_to(output_dir).as_posix())
    ]
    target = output_dir / "site-inventory.json"
    target.write_text(
        json.dumps(
            {"files": files, "schema_version": "bev-calibration-site-inventory/1"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def build_site(
    claims_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
) -> Path:
    """Build one deterministic Pages tree from already validated formal inputs."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    before = _input_snapshot(claims_path, artifacts_dir)
    report = build_formal_report(
        claims_path,
        artifacts_dir,
        output_dir,
        repository_root=repository_root,
        include_figures=True,
        page=_REPORT,
    )
    demo = output_dir / "demo" / "calibration-explorer.html"
    demo.parent.mkdir()
    demo.write_text(build_explorer(), encoding="utf-8", newline="\n")
    report_text = report.read_text(encoding="utf-8")
    marker = "</html>"
    if marker not in report_text:
        raise ValueError("formal report has no closing html element")
    link = (
        '<p><a href="../index.html">Project overview</a> · '
        '<a href="../demo/calibration-explorer.html">Open synthetic calibration explorer</a></p>'
    )
    report.write_text(report_text.replace(marker, link + marker, 1), encoding="utf-8", newline="\n")
    for relative, text in build_overview(
        claims_path, artifacts_dir, repository_root=repository_root
    ).items():
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    index = output_dir / "index.html"
    assets = _required_assets(output_dir)
    _validate_local_links(output_dir, (index, report, demo))
    if _input_snapshot(claims_path, artifacts_dir) != before:
        raise ValueError("input changed during site assembly")
    _write_inventory(output_dir, assets)
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = build_site(
        arguments.claims,
        arguments.artifacts_dir,
        arguments.output_dir,
        repository_root=Path.cwd(),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
