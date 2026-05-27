"""YAML scenario loader for memeval."""

from __future__ import annotations

from pathlib import Path

import yaml

from memeval.scenarios.types import Scenario


def load_scenario(path: str | Path) -> Scenario:
    """Load a single scenario from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    data = yaml.safe_load(path.read_text())
    return _parse_scenario(data, source_path=str(path))


def load_scenarios_from_dir(directory: str | Path) -> list[Scenario]:
    """Load all scenario YAML files from a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            scenarios.append(load_scenario(path))
        except Exception as e:
            raise ValueError(f"Error loading scenario {path}: {e}") from e

    return scenarios


def load_builtin_scenarios() -> list[Scenario]:
    """Load the built-in scenario suite shipped with memeval."""
    builtin_dir = Path(__file__).parent.parent / "datasets" / "builtin"
    if not builtin_dir.exists():
        return []
    return load_scenarios_from_dir(builtin_dir)


def _parse_scenario(data: dict, source_path: str | None = None) -> Scenario:
    """Parse a raw YAML dict into a Scenario object."""
    return Scenario(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        version=data.get("version", "1.0"),
        memory_types_tested=data.get("memory_types_tested", []),
        dimensions_tested=data.get("dimensions_tested", []),
        config=data.get("config", {}),
        setup=data.get("setup", []),
        steps=data.get("steps", []),
        thresholds=data.get("thresholds", {}),
        source_path=source_path,
    )
