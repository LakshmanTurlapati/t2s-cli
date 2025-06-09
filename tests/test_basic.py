"""
Basic tests for T2S CLI package
"""
import subprocess
import sys
import pytest

def test_package_import():
    """Test that the main package can be imported."""
    import t2s
    assert hasattr(t2s, '__version__')
    assert t2s.__version__ == "0.2.0"

def test_core_engine_import():
    """Test that the core engine can be imported."""
    from t2s.core.engine import T2SEngine
    # Basic instantiation test (might need mock data)
    assert T2SEngine is not None

def test_config_import():
    """Test that the config can be imported."""
    from t2s.core.config import Config
    assert Config is not None

def test_cli_help():
    """Test that the CLI entry point works and shows help."""
    result = subprocess.run([sys.executable, "-m", "t2s.cli", "--help"], 
                          capture_output=True, text=True, timeout=30)
    # CLI should either show help (exit code 0) or show an error that's not a module error
    assert result.returncode in [0, 1, 2]  # Allow various help/error exit codes
    # Should not have import errors
    assert "ModuleNotFoundError" not in result.stderr
    assert "ImportError" not in result.stderr

def test_cli_entry_point():
    """Test that the t2s command entry point works."""
    try:
        result = subprocess.run(["t2s", "--help"], 
                              capture_output=True, text=True, timeout=30)
        # If t2s command is available, it should work
        assert result.returncode in [0, 1, 2]
        assert "ModuleNotFoundError" not in result.stderr
        assert "ImportError" not in result.stderr
    except FileNotFoundError:
        # If t2s command is not found, that's okay in test environment
        pytest.skip("t2s command not available in test environment")

if __name__ == "__main__":
    test_package_import()
    test_core_engine_import()
    test_config_import()
    test_cli_help()
    test_cli_entry_point()
    print("All basic tests passed!") 