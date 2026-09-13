"""One source/path/claim audit boundary for formal tables and figures."""

from pathlib import Path

from bevcalib.analysis.claims import ClaimV1, audit_claims, load_registry
from bevcalib.analysis.formal_claims import load_publication
from bevcalib.artifacts.documents import DOCUMENT_TYPES, FormalArtifactSet


def load_display_evidence(
    claims_path: Path,
    artifacts_dir: Path,
    *,
    repository_root: Path,
) -> tuple[FormalArtifactSet, bytes, dict[tuple[str, str], ClaimV1]]:
    artifacts = load_publication(artifacts_dir, repository_root)
    registry_bytes = claims_path.read_bytes()
    registry = load_registry(claims_path)
    if not registry.claims:
        raise ValueError("formal report requires nonempty claims")
    violations = audit_claims(claims_path, repository_root)
    if violations:
        raise ValueError("formal claim audit failed: " + "; ".join(violations[:5]))
    scalar_claims = {}
    resolved_paths = {}
    source_directory = artifacts_dir.resolve()
    for claim in registry.claims:
        if claim.status != "verified":
            continue
        if claim.artifact_path not in resolved_paths:
            resolved_paths[claim.artifact_path] = (repository_root / claim.artifact_path).resolve()
        source = resolved_paths[claim.artifact_path]
        if source.parent != source_directory or source.stem not in DOCUMENT_TYPES:
            raise ValueError("formal report claim references the wrong artifact")
        key = (source.stem, claim.metric_path)
        if key in scalar_claims:
            raise ValueError("formal scalar has multiple verified claims")
        scalar_claims[key] = claim
    return artifacts, registry_bytes, scalar_claims


def original_documents(
    artifacts: FormalArtifactSet,
    registry_bytes: bytes,
    claims_path: Path,
    artifacts_dir: Path,
) -> dict[str, bytes]:
    """Recheck loaded snapshots after rendering, before creating any output path."""
    originals = {}
    for name in DOCUMENT_TYPES:
        raw = (artifacts_dir / f"{name}.json").read_bytes()
        if DOCUMENT_TYPES[name].model_validate_json(raw) != getattr(artifacts, name):
            raise ValueError("formal source changed during report construction")
        originals[name] = raw
    if claims_path.read_bytes() != registry_bytes:
        raise ValueError("formal registry changed during report construction")
    return originals
