from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV = "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
UPLOAD_PAGES = "actions/upload-pages-artifact@7b1f4a764d45c48632c6b24a0339c27f5614fb0b"
DEPLOY_PAGES = "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e"


def _workflow(name: str) -> dict[str, Any]:
    # BaseLoader preserves GitHub's `on` key instead of resolving it as YAML 1.1 true.
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _runs(job: dict[str, Any]) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def _run_containing(runs: list[str], fragment: str) -> str:
    return next(run for run in runs if fragment in run)


def _assert_reproducible_environment(job: dict[str, Any]) -> None:
    assert job["defaults"]["run"]["shell"] == "bash"
    assert job["env"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }

    steps = job["steps"]
    checkout = next(step for step in steps if step.get("uses") == CHECKOUT)
    assert checkout["with"]["persist-credentials"] == "false"
    setup = next(step for step in steps if step.get("uses") == SETUP_UV)
    assert setup["with"]["version"] == "0.11.18"
    assert setup["with"]["python-version"] == "3.12.13"

    runs = _runs(job)
    lock_index = runs.index("uv lock --check")
    sync_index = runs.index("uv sync --frozen --all-groups --all-extras --python 3.12.13")
    assert lock_index < sync_index


def test_ci_runs_the_full_gate_on_linux_and_windows_from_the_frozen_lock() -> None:
    workflow = _workflow("ci.yml")

    assert workflow["permissions"] == {"contents": "read"}
    verify = workflow["jobs"]["verify"]
    assert verify["strategy"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert verify["runs-on"] == "${{ matrix.os }}"
    _assert_reproducible_environment(verify)

    runs = _runs(verify)
    sync_index = runs.index("uv sync --frozen --all-groups --all-extras --python 3.12.13")
    backend_index = runs.index(
        "uv run --frozen python -m bevcalib.release backend --project pyproject.toml"
    )
    gate_index = runs.index("uv run --frozen python -m bevcalib.dev verify")
    assert sync_index < backend_index < gate_index


def test_ci_cancels_superseded_runs_only_for_pull_requests() -> None:
    # A newer push to main must not cancel a run that has already started for an
    # earlier push; a newer pull-request push may.
    assert _workflow("ci.yml")["concurrency"] == {
        "group": "ci-${{ github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


def test_ci_rebuilds_and_installs_reproducible_packages_after_the_full_gate() -> None:
    runs = _runs(_workflow("ci.yml")["jobs"]["verify"])

    gate = runs.index("uv run --frozen python -m bevcalib.dev verify")
    identity = _run_containing(runs, "SOURCE_DATE_EPOCH")
    assert "git log -1 --format=%ct HEAD" in identity
    assert 'echo "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" >> "$GITHUB_ENV"' in identity

    first_build = runs.index("uv run --frozen python -m build --no-isolation --outdir dist-a")
    first_normalize = runs.index(
        "uv run --frozen python -m bevcalib.release normalize-sdist "
        '--dist-dir dist-a --epoch "$SOURCE_DATE_EPOCH"'
    )
    first_verify = next(
        run for run in runs if "bevcalib.release verify" in run and "--dist-dir dist-a" in run
    )
    second_build = runs.index("uv run --frozen python -m build --no-isolation --outdir dist-b")
    second_normalize = runs.index(
        "uv run --frozen python -m bevcalib.release normalize-sdist "
        '--dist-dir dist-b --epoch "$SOURCE_DATE_EPOCH"'
    )
    second_verify = next(
        run for run in runs if "bevcalib.release verify" in run and "--dist-dir dist-b" in run
    )
    compare = runs.index("cmp dist-a/SHA256SUMS dist-b/SHA256SUMS")
    assert gate < runs.index(identity) < first_build < first_normalize < runs.index(first_verify)
    assert runs.index(first_verify) < second_build < second_normalize < runs.index(second_verify)
    assert runs.index(second_verify) < compare
    assert '--tag "v1.0.0"' in first_verify
    assert '--tag "v1.0.0"' in second_verify

    rebuild = _run_containing(runs, "dist-from-sdist")
    assert "bev_calibration_lab-1.0.0.tar.gz" in rebuild
    assert "python -m build --wheel --no-isolation" in rebuild
    assert "bev_calibration_lab-1.0.0-py3-none-any.whl" in rebuild

    install = _run_containing(runs, "wheel-install")
    assert "uv venv" in install
    locked_sync = (
        'VIRTUAL_ENV="$native_env" uv sync --active --frozen --all-groups --all-extras '
        '--no-install-project --python "$clean_python"'
    )
    wheel_install = 'uv pip install --python "$clean_python" --no-deps "$wheel_path"'
    assert locked_sync in install
    assert 'wheel_path="$GITHUB_WORKSPACE/dist-a/' in install
    assert wheel_install in install
    assert install.index(locked_sync) < install.index(wheel_install)
    assert 'cd "$external_root"' in install
    assert install.index('cd "$external_root"') < install.index('"$clean_python" -c')
    assert "bevcalib.__version__" in install
    assert 'version("bev-calibration-lab")' in install
    assert "bevcalib.__file__" in install
    assert "location.is_relative_to(clean)" in install
    assert "not location.is_relative_to(checkout)" in install
    assert 'native_env="$RUNNER_TEMP/wheel-install"' in install
    windows_branch = install.split('if [[ "$RUNNER_OS" == "Windows" ]]', maxsplit=1)[1].split(
        "else", maxsplit=1
    )[0]
    assert "cygpath -u" in windows_branch
    assert "native_env=" not in windows_branch
    assert "bev-calib" in install and "--help" in install
    assert "uv pip check" in install


def test_pages_builds_the_site_from_the_frozen_lock_without_rerunning_the_gate() -> None:
    workflow = _workflow("pages.yml")

    assert workflow["on"] == {"push": {"branches": ["main"]}, "workflow_dispatch": ""}
    assert workflow["permissions"] == {"contents": "read"}
    build = workflow["jobs"]["build"]
    assert build["permissions"] == {"contents": "read"}
    _assert_reproducible_environment(build)

    runs = _runs(build)
    # Pages does not wait for CI; CI runs the full gate on each push to main in a
    # separate workflow. The site build audits every claim against the committed
    # evidence before it writes anything, so a registry that does not match the
    # evidence still stops the deploy.
    assert not any("bevcalib.dev verify" in run for run in runs)
    sync_index = runs.index("uv sync --frozen --all-groups --all-extras --python 3.12.13")
    backend_index = runs.index(
        "uv run --frozen python -m bevcalib.release backend --project pyproject.toml"
    )
    site_index = runs.index(
        "uv run --frozen python -m bevcalib.report.site --claims docs/claims.yaml "
        "--artifacts-dir docs/evidence/nuscenes_calibration_v1 --output-dir site"
    )
    assert sync_index < backend_index < site_index

    upload = next(step for step in build["steps"] if step.get("uses") == UPLOAD_PAGES)
    assert upload["with"] == {"path": "site"}


def test_pages_deploy_has_only_deployment_permissions_and_needs_the_build() -> None:
    deploy = _workflow("pages.yml")["jobs"]["deploy"]

    assert deploy["needs"] == "build"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    assert deploy["steps"] == [
        {
            "name": "Deploy to GitHub Pages",
            "id": "deployment",
            "uses": DEPLOY_PAGES,
        }
    ]
