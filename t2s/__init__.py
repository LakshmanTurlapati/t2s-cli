"""
T2S - Text to SQL CLI
A powerful terminal-based Text-to-SQL converter with AI model integration.

Created by Lakshman Turlapati
Repository: https://github.com/lakshmanturlapati/t2s-cli
"""

__version__ = "0.1.2"
__author__ = "Lakshman Turlapati"
__email__ = "lakshmanturlapati@gmail.com"
__description__ = "Terminal-based Text-to-SQL converter with AI model integration"
__url__ = "https://github.com/lakshmanturlapati/t2s-cli"

from .core.engine import T2SEngine
from .core.config import Config

__all__ = ["T2SEngine", "Config"] 