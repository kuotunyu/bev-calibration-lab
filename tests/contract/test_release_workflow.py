from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV = "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
LOCK = "uv lock --check"
SYNC = "uv sync --frozen --all-groups --all-extras --python 3.12.13"
BACKEND = "uv run --frozen python -m bevcalib.release backend --project pyproject.toml"
FULL_GATE = "uv run --frozen python -m bevcalib.dev verify"
WHEEL = "dist-a/bev_calibration_lab-1.0.0-py3-none-any.whl"
SDIST = "dist-a/bev_calibration_lab-1.0.0.tar.gz"
CHECKSUMS = "dist-a/SHA256SUMS"


def _workflow() -> dict[str, Any]:
    # BaseLoader preserves GitHub's `on` key instead of resolving it as YAML 1.1 true.
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _runs(job: dict[str, Any]) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def _run_containing(runs: list[str], fragment: str) -> str:
    matches = [run for run in runs if fragment in run]
    assert len(matches) == 1
    return matches[0]


def test_release_is_tag_only_and_the_release_job_alone_can_write_contents() -> None:
    workflow = _workflow()

    assert workflow["on"] == {"push": {"tags": ["v*"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"release"}
    assert workflow["jobs"]["release"]["permissions"] == {"contents": "write"}


def test_release_uses_the_pinned_reproducible_environment_and_full_gate() -> None:
    release = _workflow()["jobs"]["release"]

    assert release["runs-on"] == "ubuntu-latest"
    assert release["defaults"]["run"]["shell"] == "bash"
    assert release["env"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    checkout = next(step for step in release["steps"] if step.get("uses") == CHECKOUT)
    assert checkout["with"] == {"fetch-depth": "0", "persist-credentials": "false"}
    setup = next(step for step in release["steps"] if step.get("uses") == SETUP_UV)
    assert setup["with"] == {"version": "0.11.18", "python-version": "3.12.13"}

    runs = _runs(release)
    assert runs.index(LOCK) < runs.index(SYNC) < runs.index(BACKEND) < runs.index(FULL_GATE)


def test_builds_are_bound_to_the_exact_tag_commit_and_epoch() -> None:
    runs = _runs(_workflow()["jobs"]["release"])
    identity = _run_containing(runs, '>> "$GITHUB_ENV"')

    assert 'test "$GITHUB_REF_TYPE" = "tag"' in identity
    assert 'test "$GITHUB_REF" = "refs/tags/$GITHUB_REF_NAME"' in identity
    assert 'git rev-parse "$GITHUB_REF^{commit}"' in identity
    assert "git rev-parse HEAD" in identity
    assert 'git log -1 --format=%ct "$GITHUB_REF^{commit}"' in identity
    assert '>> "$GITHUB_ENV"' in identity
    assert "${{" not in identity

    first_build = _run_containing(runs, "--outdir dist-a")
    second_build = _run_containing(runs, "--outdir dist-b")
    first_verify = _run_containing(
        runs, 'verify --project pyproject.toml --tag "$GITHUB_REF_NAME" --dist-dir dist-a'
    )
    second_verify = _run_containing(
        runs, 'verify --project pyproject.toml --tag "$GITHUB_REF_NAME" --dist-dir dist-b'
    )
    compare = _run_containing(runs, "cmp dist-a/SHA256SUMS")
    release = _run_containing(runs, "gh release create")
    assert "python -m build --no-isolation" in first_build
    assert "python -m build --no-isolation" in second_build
    assert '--tag "$GITHUB_REF_NAME"' in first_verify
    assert '--tag "$GITHUB_REF_NAME"' in second_verify
    assert compare.strip() == "cmp dist-a/SHA256SUMS dist-b/SHA256SUMS"
    assert (
        runs.index(identity)
        < runs.index(first_build)
        < runs.index(first_verify)
        < runs.index(second_build)
        < runs.index(second_verify)
        < runs.index(compare)
        < runs.index(release)
    )


def test_each_build_is_normalized_with_the_tag_epoch_before_verification() -> None:
    runs = _runs(_workflow()["jobs"]["release"])
    identity = _run_containing(runs, '>> "$GITHUB_ENV"')
    first_build = _run_containing(runs, "--outdir dist-a")
    second_build = _run_containing(runs, "--outdir dist-b")
    first_normalize = _run_containing(runs, "normalize-sdist --dist-dir dist-a")
    second_normalize = _run_containing(runs, "normalize-sdist --dist-dir dist-b")
    first_verify = _run_containing(
        runs, 'verify --project pyproject.toml --tag "$GITHUB_REF_NAME" --dist-dir dist-a'
    )
    second_verify = _run_containing(
        runs, 'verify --project pyproject.toml --tag "$GITHUB_REF_NAME" --dist-dir dist-b'
    )

    assert first_normalize == (
        "uv run --frozen python -m bevcalib.release normalize-sdist "
        '--dist-dir dist-a --epoch "$SOURCE_DATE_EPOCH"'
    )
    assert second_normalize == (
        "uv run --frozen python -m bevcalib.release normalize-sdist "
        '--dist-dir dist-b --epoch "$SOURCE_DATE_EPOCH"'
    )
    assert (
        runs.index(identity)
        < runs.index(first_build)
        < runs.index(first_normalize)
        < runs.index(first_verify)
        < runs.index(second_build)
        < runs.index(second_normalize)
        < runs.index(second_verify)
    )


def test_release_publishes_only_the_three_verified_versioned_assets() -> None:
    release = _run_containing(_runs(_workflow()["jobs"]["release"]), "gh release create")

    assert 'gh release create "$GITHUB_REF_NAME"' in release
    assert "--verify-tag" in release
    assert "--notes-file docs/release-notes/v1.0.0.md" in release
    for asset in (WHEEL, SDIST, CHECKSUMS):
        assert release.count(asset) == 1
    assert "dist/*" not in release
    assert "git tag" not in release
    asset_lines = [line.strip().rstrip("\\").strip() for line in release.splitlines()]
    assert [line for line in asset_lines if line.startswith("dist-")] == [WHEEL, SDIST, CHECKSUMS]


def test_sdist_rebuilds_the_same_wheel_before_release() -> None:
    runs = _runs(_workflow()["jobs"]["release"])
    manifest_compare = _run_containing(runs, "cmp dist-a/SHA256SUMS")
    rebuild = _run_containing(runs, "tar --extract")
    rebuilt_compare = _run_containing(runs, "cmp dist-a/bev_calibration_lab-1.0.0")
    release = _run_containing(runs, "gh release create")

    assert 'tar --extract --gzip --file "$GITHUB_WORKSPACE/' + SDIST + '"' in rebuild
    assert '"$RUNNER_TEMP/sdist-source/bev_calibration_lab-1.0.0"' in rebuild
    assert "uv run --frozen python -m build --wheel --no-isolation" in rebuild
    assert "python -m pip install" not in rebuild
    assert rebuilt_compare.strip() == (
        "cmp dist-a/bev_calibration_lab-1.0.0-py3-none-any.whl "
        "dist-from-sdist/bev_calibration_lab-1.0.0-py3-none-any.whl"
    )
    assert (
        runs.index(manifest_compare)
        < runs.index(rebuild)
        < runs.index(rebuilt_compare)
        < runs.index(release)
    )


def test_built_wheel_is_checked_from_a_clean_environment_outside_checkout() -> None:
    runs = _runs(_workflow()["jobs"]["release"])
    clean_install = _run_containing(runs, "wheel-install")
    rebuilt_compare = _run_containing(runs, "cmp dist-a/bev_calibration_lab-1.0.0")
    release = _run_containing(runs, "gh release create")

    assert 'python -m venv "$RUNNER_TEMP/wheel-install"' in clean_install
    assert (
        'UV_PROJECT_ENVIRONMENT="$RUNNER_TEMP/wheel-install" uv sync --frozen '
        "--all-groups --all-extras --no-install-project "
        '--python "$RUNNER_TEMP/wheel-install/bin/python"'
    ) in clean_install
    assert 'cd "$RUNNER_TEMP"' in clean_install
    assert (
        '"$RUNNER_TEMP/wheel-install/bin/python" -m pip install --no-deps '
        '"$GITHUB_WORKSPACE/' + WHEEL + '"'
    ) in clean_install
    assert "pip install --upgrade" not in clean_install
    assert clean_install.index("uv sync --frozen") < clean_install.index("pip install --no-deps")
    assert 'version("bev-calibration-lab") == bevcalib.__version__ == "1.0.0"' in clean_install
    assert 'os.environ["GITHUB_WORKSPACE"]' in clean_install
    assert '"$RUNNER_TEMP/wheel-install/bin/bev-calib" --help' in clean_install
    assert '"$RUNNER_TEMP/wheel-install/bin/python" -m pip check' in clean_install
    assert runs.index(rebuilt_compare) < runs.index(clean_install) < runs.index(release)
