"""Filesystem acceptance uses real audit/snapshots and a small figure mapping seam."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import yaml
from tests.contract.test_formal_publication import formal_claims_path as formal_claims_path
from tests.contract.test_formal_publication import formal_directory as formal_directory
from tests.unit.artifacts.test_formal_documents import payloads as payloads


@pytest.fixture
def small_mapping(formal_claims_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    import bevcalib.report.formal_figures as figures
    from bevcalib.report.figure_data import FigureCell, FigurePanel, FigurePoint
    from bevcalib.report.scalar_binding import bind_scalar

    def mapped(rows, hashes, claims):  # type: ignore[no-untyped-def]
        assert next(iter(rows)).document == "metrics"
        source = next(iter(claims.values()))
        value = source.report_binding.expected_value
        identifier = bind_scalar("metrics", source.metric_path, value, hashes["metrics"], claims)
        point = FigurePoint(
            "identity",
            0,
            value,
            None,
            None,
            "metrics",
            hashes["metrics"],
            (FigureCell("value", source.metric_path, value, identifier),),
            None,
        )
        return {
            "recovery-by-fault-level": (
                FigurePanel("yaw:absolute", "degrees", "percent", ((-1, "-1"), (1, "1")), (point,)),
            ),
            "bev-error-by-range": (
                FigurePanel(
                    "yaw:0", "GT range bin (metres)", "metres", ((0, "0-10"), (4, "80+")), (point,)
                ),
            ),
        }

    monkeypatch.setattr(figures, "figure_panels", mapped)
    return figures


def test_figure_package_copies_verified_sources_and_reproduces_exact_bytes(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    small_mapping,
) -> None:  # type: ignore[no-untyped-def]
    original = {path.name: path.read_bytes() for path in formal_directory.iterdir()}
    first = small_mapping.build_formal_figures(
        formal_claims_path, formal_directory, tmp_path / "first", repository_root=tmp_path
    )
    second = small_mapping.build_formal_figures(
        formal_claims_path, formal_directory, tmp_path / "second", repository_root=tmp_path
    )
    assert {path.name for path in first} == {
        "recovery-by-fault-level.svg",
        "bev-error-by-range.svg",
    }
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    for path in first:
        root = ET.fromstring(path.read_bytes())
        assert "Synthetic" in "".join(root.itertext())
    for name, raw in original.items():
        assert (tmp_path / "first" / "evidence" / name).read_bytes() == raw
        assert (formal_directory / name).read_bytes() == raw
    assert (tmp_path / "first" / "claims.yaml").read_bytes() == formal_claims_path.read_bytes()


def test_formal_report_can_embed_the_same_audited_figures(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    small_mapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    import bevcalib.report.formal as report
    from bevcalib.analysis.claims import load_registry
    from bevcalib.analysis.formal_claims import PublicationRow

    source = load_registry(formal_claims_path).claims[0]
    row = PublicationRow(
        "metrics",
        "Synthetic row",
        "percent",
        (("value", source.metric_path, source.report_binding.expected_value),),
    )
    monkeypatch.setattr(report, "publication_rows", lambda _: iter((row,)))
    page = report.build_formal_report(
        formal_claims_path,
        formal_directory,
        tmp_path / "site",
        repository_root=tmp_path,
        include_figures=True,
    )
    text = page.read_text(encoding="utf-8")
    for name in ("recovery-by-fault-level", "bev-error-by-range"):
        assert f"figures/{name}.svg" in text
        assert (page.parent / "figures" / f"{name}.svg").is_file()
    assert (page.parent / "claims.yaml").read_bytes() == formal_claims_path.read_bytes()


@pytest.mark.parametrize(
    "case", ["missing-file", "wrong-value", "source-change", "registry-change", "existing"]
)
def test_bad_or_changed_sources_and_existing_output_are_not_published(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    small_mapping,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "output"
    if case == "missing-file":
        (formal_directory / "timing.json").unlink()
    elif case == "wrong-value":
        body = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
        body["claims"][0]["report_binding"]["expected_value"] += 1
        formal_claims_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    elif case == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("retain", encoding="utf-8")

    original_render = small_mapping.render_svg

    def render(*args, **kwargs):  # type: ignore[no-untyped-def]
        if case == "registry-change":
            with formal_claims_path.open("a", encoding="utf-8") as handle:
                handle.write("\n# concurrent writer\n")
        elif case == "source-change":
            from bevcalib.artifacts.result_documents import digest

            source = formal_directory / "metrics.json"
            body = json.loads(source.read_bytes())
            body["identity"]["measurement_identity_sha256"] = "f" * 64
            body["document_sha256"] = digest(
                {key: value for key, value in body.items() if key != "document_sha256"}
            )
            source.write_text(json.dumps(body), encoding="utf-8")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(small_mapping, "render_svg", render)
    with pytest.raises((ValueError, FileExistsError)):
        small_mapping.build_formal_figures(
            formal_claims_path, formal_directory, output, repository_root=tmp_path
        )
    if case == "existing":
        assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]
        assert (output / "keep.txt").read_text(encoding="utf-8") == "retain"
    else:
        assert not output.exists()
