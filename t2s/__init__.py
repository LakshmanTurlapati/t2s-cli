"""
T2S - Text to SQL CLI
A powerful terminal-based Text-to-SQL converter with AI model integration.

Created by Lakshman Turlapati
Repository: https://github.com/lakshmanturlapati/t2s-cli
"""

import sys
from pathlib import Path

try:
    # Try to read version from pyproject.toml
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        # Fallback for Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None
    
    if tomllib:
        # Get the path to pyproject.toml
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
            __version__ = pyproject_data.get("project", {}).get("version", "0.2.0")
        else:
            __version__ = "0.2.0"
    else:
        __version__ = "0.2.0"
        
except Exception:
    # Fallback if anything goes wrong
    __version__ = "0.2.0"

__author__ = "Lakshman Turlapati"
__email__ = "lakshmanturlapati@gmail.com"
__description__ = "Terminal-based Text-to-SQL converter with AI model integration"
__url__ = "https://github.com/lakshmanturlapati/t2s-cli"

from .core.engine import T2SEngine
from .core.config import Config

__all__ = ["T2SEngine", "Config"] 