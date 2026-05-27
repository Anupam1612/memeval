"""Tests for scenario loading."""


import pytest

from memeval.scenarios.loader import load_builtin_scenarios, load_scenario


def test_load_builtin_scenarios():
    scenarios = load_builtin_scenarios()
    assert len(scenarios) >= 5
    names = {s.name for s in scenarios}
    assert "Basic Recall" in names
    assert "Preference Update" in names
    assert "Privacy Isolation" in names


def test_scenario_structure():
    scenarios = load_builtin_scenarios()
    for s in scenarios:
        assert s.name
        assert s.description
        assert s.version
        assert isinstance(s.dimensions_tested, list)
        assert len(s.dimensions_tested) > 0
        assert len(s.setup) + len(s.steps) > 0


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_scenario("/nonexistent/path.yaml")
