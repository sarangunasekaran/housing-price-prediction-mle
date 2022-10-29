# Add vendored libs to the path
import os
import sys

# silence warnings
from .core.base_utils import silence_common_warnings as _silence_warnings
from .version import version

__version__ = version


_silence_warnings()


_HERE = os.path.abspath(os.path.dirname(__file__))
_VENDOR_PATH = os.path.join(_HERE, "_vendor")
if os.path.exists(_VENDOR_PATH):
    sys.path.append(_VENDOR_PATH)
