import sys
import os
from pathlib import Path

_ROOT_DIR = Path(os.getcwd()).parent


def init_notebook():
    sys.path.append(str(_ROOT_DIR))
    return _ROOT_DIR


init_notebook()
