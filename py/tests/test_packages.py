from importlib import import_module

import pytest


@pytest.mark.parametrize("package", ["brier_sim", "brier_ml", "brier_explain"])
def test_packages_import(package: str) -> None:
    assert import_module(package).__name__ == package
