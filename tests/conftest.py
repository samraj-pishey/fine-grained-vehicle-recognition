import pytest

from vehicle_recognition.config import get_data_paths


@pytest.fixture(scope="session")
def data_paths():
    try:
        p = get_data_paths()
    except RuntimeError:
        pytest.skip("DATA_ROOT not configured")
    if not (p.devkit / "cars_meta.mat").exists():
        pytest.skip("Stanford Cars devkit not present under DATA_ROOT")
    return p
