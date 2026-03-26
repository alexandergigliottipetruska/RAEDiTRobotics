import pytest

DEFAULT_HDF5 = (
    "c:/Users/naqee/OneDrive/Desktop/CSC415 Project"
    "/data/unified/robomimic/lift/ph_abs_v15.hdf5"
)


def pytest_addoption(parser):
    parser.addoption("--hdf5", default=DEFAULT_HDF5, help="Path to unified HDF5 file")


@pytest.fixture(scope="session")
def hdf5_path(request):
    return request.config.getoption("--hdf5")
