from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
sample_text = (FIXTURES / "sample_doc.md").read_text(encoding="utf-8")


@pytest.fixture
def test_data():
    return FIXTURES
