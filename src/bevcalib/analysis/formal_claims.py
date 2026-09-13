"""Fixed publication fields and exact scalar claims for the formal result set."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bevcalib.analysis.claims import (
    ALLOWED_EVIDENCE_TYPES,
    ALLOWED_STATUSES,
    CLAIM_REQUIRED_FIELDS,
    ClaimsRegistryV1,
    ClaimV1,
    ReportScalarBinding,
)
from bevcalib.analysis.policy import METRIC_UNITS
from bevcalib.artifacts.documents import DOCUMENT_TYPES, FormalArtifactSet, load_formal_artifact_set

# Fixed before viewing a publication subset. All axes/levels and methods remain.
# Signed/per-axis pose descriptors remain in the linked full metrics document.
REPORT_METRICS = (
    "rotation_geodesic_deg",
    "translation_norm_cm",
    "pixel_frame_p50_px",
    "pixel_frame_p90_px",
    "edge_score_px",
    *(name for name in METRIC_UNITS if name.startswith("bev_frame_mean_m/")),
)
SUPPORT_FIELDS = ("frames", "total_frames", "scenes", "total_scenes", "objects", "total_objects")


@dataclass(frozen=True)
class PublicationRow:
    document: str
    label: str
    unit: str
    cells: tuple[tuple[str, str, Any], ...]
    reason: str | None = None


def pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def claim_id(document: str, pointer: str) -> str:
    return f"formal.{document}.{hashlib.sha256(pointer.encode('utf-8')).hexdigest()}"


def publication_rows(artifacts: FormalArtifactSet) -> Iterator[PublicationRow]:
    """Select explicit report fields, never estimates based on their observed value."""

    def estimate_row(
        document: str, label: str, unit: str, base: str, value: dict[str, Any]
    ) -> PublicationRow:
        fields = ("before", "after", "improvement") if "before" in value else ("value",)
        cells = [(field, f"{base}/{field}", value[field]) for field in fields]
        if "interval" in value:
            for field in ("low", "high"):
                interval = value["interval"]
                cells.append(
                    (
                        f"interval {field}",
                        f"{base}/interval/{field}",
                        None if interval is None else interval[field],
                    )
                )
        support = value["support"]
        for field in SUPPORT_FIELDS:
            if support[field] is not None:
                cells.append((field, f"{base}/support/{field}", support[field]))
        for reason, count in sorted(support["exclusions"].items()):
            cells.append(
                (f"excluded: {reason}", f"{base}/support/exclusions/{pointer_token(reason)}", count)
            )
        return PublicationRow(document, label, unit, tuple(cells), value["reason"])

    for label, conditions in sorted(artifacts.metrics.model_dump(mode="json")["runs"].items()):
        for condition, metrics in sorted(conditions.items()):
            for metric in REPORT_METRICS:
                base = f"/runs/{pointer_token(label)}/{pointer_token(condition)}/{pointer_token(metric)}"
                yield estimate_row(
                    "metrics",
                    f"{label} · {condition} · {metric}",
                    METRIC_UNITS[metric],
                    base,
                    metrics[metric],
                )
    for label, conditions in sorted(
        artifacts.intervals.model_dump(mode="json")["comparisons"].items()
    ):
        for condition, metrics in sorted(conditions.items()):
            for metric in REPORT_METRICS:
                base = f"/comparisons/{pointer_token(label)}/{pointer_token(condition)}/{pointer_token(metric)}"
                yield estimate_row(
                    "intervals",
                    f"{label} · {condition} · {metric}",
                    METRIC_UNITS[metric],
                    base,
                    metrics[metric],
                )
    recovery = artifacts.recovery.model_dump(mode="json")
    for group in ("runs", "comparisons"):
        for label, conditions in sorted(recovery[group].items()):
            for condition, value in sorted(conditions.items()):
                base = f"/{group}/{pointer_token(label)}/{pointer_token(condition)}"
                unit = (
                    "percent; improvement in percentage points"
                    if group == "comparisons"
                    else "percent"
                )
                yield estimate_row("recovery", f"{label} · {condition}", unit, base, value)
    for offset, value in sorted(
        artifacts.timing.model_dump(mode="json")["offsets"].items(), key=lambda item: int(item[0])
    ):
        base = f"/offsets/{pointer_token(offset)}"
        fields: tuple[str, ...] = (
            "requested_offset_ms",
            "total",
            "selected",
            "valid",
            "invalid",
            "valid_fraction",
        )
        cells = [(field, f"{base}/{field}", value[field]) for field in fields]
        for field in ("realized_offset_ms", "absolute_error_ms"):
            for statistic in ("count", "mean", "median", "p90"):
                cells.append(
                    (f"{field} {statistic}", f"{base}/{field}/{statistic}", value[field][statistic])
                )
        for reason, count in sorted(value["reasons"].items()):
            cells.append((reason, f"{base}/reasons/{pointer_token(reason)}", count))
        yield PublicationRow(
            "timing",
            f"identity timing stress · {offset}",
            "offsets: ms; counts; fraction",
            tuple(cells),
        )
    for label, conditions in sorted(artifacts.exclusions.model_dump(mode="json")["runs"].items()):
        for condition, value in sorted(conditions.items()):
            base = f"/runs/{pointer_token(label)}/{pointer_token(condition)}"
            fields = (
                "total",
                "valid",
                "invalid",
                "projection_input_points",
                "within_row_pixel_error_count",
            )
            cells = [(field, f"{base}/{field}", value[field]) for field in fields]
            for reason, count in sorted(value["reasons"].items()):
                cells.append((reason, f"{base}/reasons/{pointer_token(reason)}", count))
            yield PublicationRow(
                "exclusions", f"{label} · {condition}", "counts; global row validity", tuple(cells)
            )


def load_publication(artifacts_dir: Path, repository_root: Path) -> FormalArtifactSet:
    root = repository_root.resolve()
    if not artifacts_dir.resolve().is_relative_to(root) or any(
        not (artifacts_dir / f"{name}.json").resolve().is_relative_to(root)
        for name in DOCUMENT_TYPES
    ):
        raise ValueError("formal artifact path escapes repository")
    return load_formal_artifact_set(artifacts_dir)


def generate_formal_claims(
    artifacts_dir: Path, output_path: Path, *, repository_root: Path
) -> Path:
    artifacts = load_publication(artifacts_dir, repository_root)
    if not output_path.resolve().is_relative_to(repository_root.resolve()):
        raise ValueError("formal registry path escapes repository")
    if output_path.exists():
        raise FileExistsError("formal registry already exists")
    identity = artifacts.metrics.identity
    artifact_paths = {
        name: (artifacts_dir / f"{name}.json")
        .resolve()
        .relative_to(repository_root.resolve())
        .as_posix()
        for name in DOCUMENT_TYPES
    }
    claims = {}
    for row in publication_rows(artifacts):
        document = getattr(artifacts, row.document)
        for _, pointer, value in row.cells:
            if value is not None:
                identifier = claim_id(row.document, pointer)
                claims[identifier] = ClaimV1(
                    claim_id=identifier,
                    text=f"Formal scalar: {value}",
                    evidence_type=identity.evidence_type,
                    protocol_hash=identity.protocol_hash,
                    dataset_manifest_hash=identity.dataset_manifest_hash,
                    artifact_path=artifact_paths[row.document],
                    metric_path=pointer,
                    status="verified",
                    report_binding=ReportScalarBinding(
                        expected_summary_sha256=document.document_sha256, expected_value=value
                    ),
                )
    registry = ClaimsRegistryV1(
        allowed_evidence_types=ALLOWED_EVIDENCE_TYPES,
        claim_required_fields=CLAIM_REQUIRED_FIELDS,
        allowed_statuses=ALLOWED_STATUSES,
        claims=tuple(claims[key] for key in sorted(claims)),
    )
    contents = yaml.dump(
        registry.model_dump(mode="json"),
        Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper),
        sort_keys=False,
        allow_unicode=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".formal-claims-", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
        os.link(temporary, output_path)
    finally:
        Path(temporary).unlink()
    return output_path
