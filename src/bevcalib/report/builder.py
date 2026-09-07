"""Deterministic report rendering from one safe summary and exact scalar claims."""

from __future__ import annotations

from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from typing import Any

from bevcalib.analysis.claims import _resolve_json_pointer, _text_numbers, load_registry
from bevcalib.artifacts.summary import load_safe_summary
from bevcalib.report.figures import validity_bar


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def build_report(
    claims_path: Path, artifacts_dir: Path, output_dir: Path, *, repository_root: Path | None = None
) -> Path:
    from jinja2 import Environment

    root = (repository_root or Path.cwd()).resolve()
    artifact_path = (artifacts_dir / "calibration_summary.json").resolve()
    summary = load_safe_summary(artifact_path)
    document = summary.model_dump(mode="json")
    registry = load_registry(claims_path)
    if len({claim.claim_id for claim in registry.claims}) != len(registry.claims):
        raise ValueError("duplicate claim ID in report registry")
    verified = sorted(
        (claim for claim in registry.claims if claim.status == "verified"),
        key=lambda claim: claim.claim_id,
    )
    scalar_claims = {}
    contexts = []
    for claim in verified:
        if (
            root / claim.artifact_path
        ).resolve() != artifact_path or not artifact_path.is_relative_to(root):
            raise ValueError("report claim references the wrong artifact")
        if (
            claim.protocol_hash != summary.protocol_hash
            or claim.dataset_manifest_hash != summary.dataset_manifest_hash
        ):
            raise ValueError("report claim provenance differs from safe summary")
        allowed = {"synthetic"} if summary.evidence_type == "synthetic" else {"observed", "derived"}
        if claim.evidence_type not in allowed:
            raise ValueError("claim evidence label is incompatible with safe summary")
        try:
            value = _resolve_json_pointer(document, claim.metric_path)
        except LookupError as exc:
            raise ValueError("report claim pointer does not exist") from exc
        numbers = _text_numbers(claim.text)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if any(number != Decimal(str(value)) for number in numbers):
                raise ValueError("claim text number differs from its exact scalar")
            if claim.metric_path in scalar_claims:
                raise ValueError("displayed scalar has multiple verified claims")
            scalar_claims[claim.metric_path] = {
                "value": value,
                "id": claim.claim_id,
                "text": claim.text,
                "evidence": claim.evidence_type,
            }
        else:
            if numbers:
                raise ValueError("nonscalar claims cannot authorize displayed numbers")
            contexts.append({"id": claim.claim_id, "text": claim.text})
    used = set()

    def metric(pointer: str) -> dict[str, Any]:
        if pointer not in scalar_claims:
            raise ValueError(f"unclaimed displayed scalar: {pointer}")
        used.add(pointer)
        return scalar_claims[pointer]

    sections = []
    for label, run in sorted(document["runs"].items()):
        base = "/runs/" + _pointer_token(label)
        section = {
            "method": run["method"],
            "seed": None if run["seed"] is None else metric(base + "/seed"),
            "extrinsic": [],
            "timing": [],
            "ranges": [],
        }
        for key, condition in run["conditions"].items():
            path = base + "/conditions/" + _pointer_token(key)
            validity = condition["validity"]
            row = {
                "axis": condition["fault_axis"],
                "level": metric(path + "/fault_level"),
                "valid": metric(path + "/validity/valid"),
                "invalid": metric(path + "/validity/invalid"),
                "total": metric(path + "/validity/total"),
                "rate": metric(path + "/validity/invalid_rate"),
                "reasons": [
                    {
                        "text": reason,
                        "count": metric(path + "/validity/reasons/" + _pointer_token(reason)),
                    }
                    for reason in validity["reasons"]
                ],
            }
            row["bar"] = validity_bar(row["rate"]["value"], row["rate"]["id"])
            section["timing" if condition["fault_axis"] == "time" else "extrinsic"].append(row)
        for name, group in run["conditions"]["yaw:0"]["ground_contact_by_range"].items():
            path = base + "/conditions/yaw:0/ground_contact_by_range/" + _pointer_token(name)
            section["ranges"].append(
                {
                    "name": name,
                    "total": metric(path + "/total"),
                    "count": metric(path + "/count"),
                    "invalid": metric(path + "/invalid"),
                    "mean": None if group["mean"] is None else metric(path + "/mean"),
                    "reasons": [
                        {
                            "text": reason,
                            "count": metric(path + "/reasons/" + _pointer_token(reason)),
                        }
                        for reason in group["reasons"]
                    ],
                }
            )
        sections.append(section)
    extras = [value for key, value in scalar_claims.items() if key not in used]
    template = (
        files("bevcalib.report").joinpath("templates/index.html.j2").read_text(encoding="utf-8")
    )
    html = (
        Environment(autoescape=True, keep_trailing_newline=True)
        .from_string(template)
        .render(sections=sections, extras=extras, contexts=contexts, evidence=summary.evidence_type)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "index.html"
    output.write_text(html, encoding="utf-8", newline="\n")
    return output
