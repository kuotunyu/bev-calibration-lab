"""Deterministic checked-in JSON Schema generation for the artifact contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .envelope import PortfolioArtifactEnvelopeV1, canonical_json_bytes
from .result_documents import RunCompleteV2, SceneResultDocumentV2
from .results import CalibrationResultV1, CalibrationResultV2
from .run_record import RunRecordV1

_SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("calibration_result_v1.json", CalibrationResultV1),
    ("calibration_result_v2.json", CalibrationResultV2),
    ("scene_result_v2.json", SceneResultDocumentV2),
    ("run_complete_v2.json", RunCompleteV2),
    ("portfolio_artifact_envelope_v1.json", PortfolioArtifactEnvelopeV1),
    ("run_record_v1.json", RunRecordV1),
)


def contract_schema_documents() -> dict[str, bytes]:
    """Return model-derived schemas in stable filename order and byte format."""

    return {
        filename: canonical_json_bytes(model.model_json_schema()) + b"\n"
        for filename, model in _SCHEMA_MODELS
    }


def write_contract_schemas(repository_root: Path) -> tuple[Path, ...]:
    """Regenerate every checked-in schema under one repository root.

    The schemas are generated from the models rather than written by hand, so a
    schema cannot describe a contract the code no longer implements.
    """

    schema_directory = repository_root / "schemas"
    schema_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, document in contract_schema_documents().items():
        path = schema_directory / filename
        path.write_bytes(document)
        written.append(path)
    return tuple(written)
