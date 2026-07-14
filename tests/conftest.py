"""Central test-tier classification.

Existing deterministic tests are unit tests by default. Expensive tests must
opt in explicitly with ``native``/``gpu``/``integration``/``slow`` markers.
Plain ``pytest`` still collects every test; profile scripts select subsets.
"""

from __future__ import annotations

import pytest


EXPENSIVE_MARKERS = frozenset({"native", "gpu", "integration", "slow"})


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        names = {marker.name for marker in item.iter_markers()}
        if not names.intersection(EXPENSIVE_MARKERS):
            item.add_marker(pytest.mark.unit)
        elif "gpu" in names and "native" not in names:
            item.add_marker(pytest.mark.native)
