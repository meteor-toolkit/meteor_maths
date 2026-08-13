"""meteor_maths - meteor_maths is a python package with mathematical algorithms for use in earth observation data and tools."""

__all__ = []
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meteor_maths")
except PackageNotFoundError:
    __version__ = "unknown"
