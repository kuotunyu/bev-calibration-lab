from __future__ import annotations

import tomllib
from pathlib import Path

import bevcalib

PROJECT_ROOT = Path(__file__).parents[2]


def test_source_declares_the_release_identity_and_build_backends() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["version"] == "1.0.0"
    assert bevcalib.__version__ == "1.0.0"
    assert metadata["build-system"] == {
        "requires": ["setuptools==84.0.0", "wheel==0.45.1"],
        "build-backend": "setuptools.build_meta",
    }
    assert {"setuptools==84.0.0", "wheel==0.45.1"} <= set(metadata["dependency-groups"]["dev"])
