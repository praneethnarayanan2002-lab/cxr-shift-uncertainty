import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxrshift.utils import load_config  # noqa: E402


@pytest.fixture
def smoke_cfg(tmp_path):
    return load_config(ROOT / "configs" / "smoke.yaml", [f"output_dir={tmp_path}"])
