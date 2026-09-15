import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_submodule_matches_lock_and_required_context():
    lock = json.loads((ROOT / "DEPENDENCY_LOCK.json").read_text(encoding="utf-8"))["exact_sew"]
    submodule = ROOT / lock["submodule_path"]
    assert submodule.is_dir()
    actual = subprocess.check_output(
        ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == lock["commit"]
    for relative, expected in lock["required_context_sha256"].items():
        digest = hashlib.sha256((submodule / relative).read_bytes()).hexdigest()
        assert digest == expected

