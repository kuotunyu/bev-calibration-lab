import fnmatch
import tomllib
from pathlib import Path


def test_mutation_scope_retains_the_pure_core_and_formal_pairing() -> None:
    root = Path(__file__).parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["mutmut"]["only_mutate"]
    required = [
        "src/bevcalib/analysis/estimands.py",
        "src/bevcalib/analysis/aggregate.py",
        *(
            path.relative_to(root).as_posix()
            for family in ("geometry", "perturbations", "operators", "correctors", "metrics")
            for path in (root / "src" / "bevcalib" / family).glob("*.py")
        ),
    ]
    assert all((root / path).is_file() for path in required)
    assert all(any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns) for path in required)


def test_mutation_sandbox_copies_non_source_inputs_consumed_by_contract_tests() -> None:
    root = Path(__file__).parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    copied_roots = {Path(path) for path in config["tool"]["mutmut"]["also_copy"]}
    consumed = (
        Path(".agents/skills/verifying-nuscenes-transforms/SKILL.md"),
        Path(".agents/skills/verifying-nuscenes-transforms/scripts/validate_transform_chain.py"),
        Path(".github/workflows/ci.yml"),
        Path(".github/workflows/pages.yml"),
        Path(".github/workflows/release.yml"),
        Path("README.md"),
        Path("README.en.md"),
        Path("NOTICE"),
        Path("CITATION.cff"),
    )

    assert all((root / path).is_file() for path in consumed)
    assert all(any(path.is_relative_to(copied) for copied in copied_roots) for path in consumed)
