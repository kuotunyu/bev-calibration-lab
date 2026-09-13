"""Strict public-claim vocabulary, filtering, and evidence audit.

The vocabulary is shared across the three portfolio repositories and is enforced
here rather than imported, so a repository cannot quietly widen it for itself.
The audit answers one question: can this exact sentence be reproduced from a
committed artifact right now?
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    model_validator,
)

ALLOWED_EVIDENCE_TYPES = ("observed", "derived", "synthetic", "illustrative")
CLAIM_REQUIRED_FIELDS = (
    "claim_id",
    "text",
    "evidence_type",
    "protocol_hash",
    "dataset_manifest_hash",
    "artifact_path",
    "metric_path",
    "status",
)
ALLOWED_STATUSES = ("draft", "verified", "rejected", "superseded")

_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


class ReportScalarBinding(BaseModel):
    """The exact safe artifact and scalar that a report claim actually verified."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_value: StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]


class ClaimV1(BaseModel):
    """One immutable public statement linked to an exact metric in an exact artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    text: str = Field(min_length=1)
    evidence_type: Literal["observed", "derived", "synthetic", "illustrative"]
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_path: str = Field(min_length=1)
    metric_path: str = Field(pattern=r"^/")
    status: Literal["draft", "verified", "rejected", "superseded"]
    # Optional for the legacy audit contract; mandatory for displayed report scalars.
    report_binding: ReportScalarBinding | None = None


class ClaimsRegistryV1(BaseModel):
    """This repository's copy of the approved cross-repository claim vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_evidence_types: tuple[str, ...]
    claim_required_fields: tuple[str, ...]
    allowed_statuses: tuple[str, ...]
    claims: tuple[ClaimV1, ...]

    @model_validator(mode="after")
    def validate_shared_vocabulary(self) -> Self:
        """Refuse a registry that has drifted from the vocabulary the portfolio agreed on."""

        if self.allowed_evidence_types != ALLOWED_EVIDENCE_TYPES:
            raise ValueError("allowed_evidence_types must match the approved vocabulary")
        if self.claim_required_fields != CLAIM_REQUIRED_FIELDS:
            raise ValueError("claim_required_fields must match the approved vocabulary")
        if self.allowed_statuses != ALLOWED_STATUSES:
            raise ValueError("allowed_statuses must match the approved vocabulary")
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("duplicate claim ID in registry")
        return self


def load_registry(claims_path: Path) -> ClaimsRegistryV1:
    """Load and strictly validate one claims registry document."""

    raw_value = yaml.load(
        claims_path.read_text(encoding="utf-8"),
        Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
    )
    return ClaimsRegistryV1.model_validate(raw_value)


def verified_claims(claims_path: Path) -> tuple[ClaimV1, ...]:
    """Return only the claims permitted to reach a public README or report."""

    return tuple(claim for claim in load_registry(claims_path).claims if claim.status == "verified")


def _resolve_json_pointer(document: object, pointer: str) -> object:
    current = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and (
            token == "0" or (not token.startswith("0") and token.isdecimal())
        ):
            index = int(token)
            if index >= len(current):
                raise LookupError(pointer)
            current = current[index]
        else:
            raise LookupError(pointer)
    return current


def _text_numbers(text: str) -> tuple[Decimal, ...]:
    return tuple(
        Decimal(match.group().removesuffix("%")) for match in _NUMBER_PATTERN.finditer(text)
    )


def _json_numbers(value: object) -> set[Decimal]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, int | float | Decimal):
        return {Decimal(str(value))}
    numbers: set[Decimal] = set()
    if isinstance(value, dict):
        for child in value.values():
            numbers.update(_json_numbers(child))
    elif isinstance(value, list):
        for child in value:
            numbers.update(_json_numbers(child))
    return numbers


def audit_claims(claims_path: Path, repository_root: Path) -> tuple[str, ...]:
    """Report every claim whose exact evidence cannot be reproduced locally.

    Returns violations rather than raising, because the useful output is the
    complete list: fixing one claim per run is how the second stale number ships.
    """

    try:
        registry = load_registry(claims_path)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        return (f"claims registry is invalid: {exc}",)

    from bevcalib.analysis.formal_claims import load_publication
    from bevcalib.artifacts.documents import DOCUMENT_TYPES, FormalArtifactSet

    formal_schemas = {
        get_args(cls.model_fields["schema_version"].annotation)[0]: name
        for name, cls in DOCUMENT_TYPES.items()
    }
    formal_namespaces = {schema.split("/")[0] for schema in formal_schemas}
    root = repository_root.resolve()
    violations: list[str] = []
    artifacts: dict[Path, Any] = {}
    resolved_paths: dict[str, Path] = {}
    formal_sets: dict[Path, FormalArtifactSet] = {}
    formal_failures: dict[Path, str] = {}
    validated_formal: dict[Path, Any] = {}
    for claim in registry.claims:
        if claim.artifact_path not in resolved_paths:
            resolved_paths[claim.artifact_path] = (root / claim.artifact_path).resolve()
        artifact_path = resolved_paths[claim.artifact_path]
        if not artifact_path.is_relative_to(root):
            violations.append(
                f"{claim.claim_id}: artifact path escapes repository: {claim.artifact_path}"
            )
            continue
        if artifact_path not in artifacts and not artifact_path.is_file():
            violations.append(f"{claim.claim_id}: artifact does not exist: {claim.artifact_path}")
            continue

        try:
            if artifact_path not in artifacts:
                artifacts[artifact_path] = json.loads(
                    artifact_path.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                )
            artifact = artifacts[artifact_path]
        except (OSError, UnicodeDecodeError, ValueError):
            violations.append(f"{claim.claim_id}: artifact is not valid UTF-8 JSON")
            continue

        if not isinstance(artifact, dict):
            violations.append(f"{claim.claim_id}: artifact root must be an object")
            continue
        schema = artifact.get("schema_version")
        formal_name = formal_schemas.get(schema) if isinstance(schema, str) else None
        if formal_name is None and (
            (isinstance(schema, str) and schema.split("/")[0] in formal_namespaces)
            or (artifact_path.stem in DOCUMENT_TYPES and "identity" in artifact)
        ):
            violations.append(f"{claim.claim_id}: formal schema is missing or unsupported")
            continue
        identity = artifact
        if formal_name is not None:
            try:
                directory = artifact_path.parent
                if artifact_path.name != f"{formal_name}.json":
                    raise ValueError("formal document filename differs from its schema")
                if directory in formal_failures:
                    raise ValueError(formal_failures[directory])
                if directory not in formal_sets:
                    try:
                        formal_sets[directory] = load_publication(directory, root)
                    except (OSError, UnicodeDecodeError, ValueError) as exc:
                        formal_failures[directory] = str(exc)
                        raise
                if artifact_path not in validated_formal:
                    document = getattr(formal_sets[directory], formal_name).model_dump(mode="json")
                    snapshot = (
                        DOCUMENT_TYPES[formal_name].model_validate(artifact).model_dump(mode="json")
                    )
                    if snapshot != document:
                        raise ValueError("formal artifact changed during audit")
                    validated_formal[artifact_path] = document
                identity = validated_formal[artifact_path]["identity"]
                allowed = (
                    {"synthetic"}
                    if identity["evidence_type"] == "synthetic"
                    else {"observed", "derived"}
                )
                if claim.evidence_type not in allowed:
                    raise ValueError("formal claim evidence label is incompatible")
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                violations.append(f"{claim.claim_id}: formal evidence invalid: {exc}")
                continue
        if identity.get("protocol_hash") != claim.protocol_hash:
            violations.append(f"{claim.claim_id}: protocol hash mismatch")
        if identity.get("dataset_manifest_hash") != claim.dataset_manifest_hash:
            violations.append(f"{claim.claim_id}: dataset manifest hash mismatch")

        try:
            metric = _resolve_json_pointer(artifact, claim.metric_path)
        except LookupError:
            violations.append(
                f"{claim.claim_id}: metric JSON pointer does not exist: {claim.metric_path}"
            )
            continue

        if formal_name is not None:
            binding = claim.report_binding
            if (
                isinstance(metric, bool)
                or not isinstance(metric, int | float)
                or binding is None
                or binding.expected_summary_sha256 != artifact["document_sha256"]
                or Decimal(str(binding.expected_value)) != Decimal(str(metric))
            ):
                violations.append(f"{claim.claim_id}: formal scalar binding differs from evidence")
                continue

        metric_numbers = _json_numbers(metric)
        for number in _text_numbers(claim.text):
            if number not in metric_numbers:
                violations.append(f"{claim.claim_id}: claim number is absent from metric: {number}")

    return tuple(violations)
