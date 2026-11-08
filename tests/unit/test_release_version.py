import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("tag", "version", "succeeds"),
    [
        ("v0.6.7", "0.6.7", True),
        ("0.6.7", "0.6.7", True),
        ("v0.7.0rc1", "0.7.0c1", True),
        ("v0.6.6b", "0.6.6", False),
        ("v06.6c", "0.6.6c", False),
        ("not-a-version", "0.6.7", False),
    ],
)
def test_release_version_check(
    tmp_path: Path, tag: str, version: str, succeeds: bool
) -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github/workflows/publish.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text())
    steps = workflow["jobs"]["build"]["steps"]
    script = next(
        step["run"] for step in steps if step.get("name") == "Check release version"
    )
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    env = os.environ | {
        "RELEASE_TAG": tag,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
