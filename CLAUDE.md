# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T2S is a privacy-first Text-to-SQL converter that runs AI models locally to transform natural language queries into SQL statements. It supports multiple databases (SQLite, PostgreSQL, MySQL) and AI models (Gemma, Llama, SmolVLM, Defog SQLCoder).

## Essential Commands

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_basic.py -v
```

### Code Quality
```bash
# Format code (MUST run before committing)
black .

# Check code style
flake8 .

# Type checking
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Building and Publishing
```bash
# Build distribution
python -m build

# Upload to PyPI (requires credentials)
twine upload dist/*
```

## Architecture Overview

### Core Components
- **t2s/core/engine.py**: Main orchestration engine handling the text-to-SQL conversion pipeline
- **t2s/models/**: AI model management, including downloading, loading, and inference
- **t2s/database/db_manager.py**: Database connection management using SQLAlchemy
- **t2s/utils/schema_analyzer.py**: Analyzes database schemas to provide context to AI models
- **t2s/cli.py**: Click-based CLI entry point with commands for query, config, models, and databases

### Key Design Patterns
1. **Async Architecture**: Uses asyncio throughout for non-blocking operations
2. **Configuration**: Pydantic models in core/config.py, stored as JSON in user's home directory
3. **Model Management**: Dynamic downloading from HuggingFace with memory-aware recommendations
4. **Error Handling**: Comprehensive try-catch blocks with user-friendly error messages
5. **Platform Detection**: Special handling for Windows, macOS, and VM environments

### Critical Implementation Details
- **Device Management**: Careful handling of CUDA/CPU device selection, especially for Windows and VMs
- **Model Loading**: Uses transformers library with specific optimizations for each model type
- **Query Validation**: SQL queries are validated using sqlparse before execution
- **Schema Context**: Database schemas are analyzed and included in prompts for better accuracy

## Development Guidelines

### Adding New Models
1. Update `SUPPORTED_MODELS` in `t2s/core/config.py`
2. Add model-specific loading logic in `t2s/models/model_manager.py`
3. Update memory requirements and recommendations

### Working with Databases
- Database connections use SQLAlchemy for abstraction
- Connection strings are stored securely in config
- Auto-discovery works for SQLite files in the current directory

### UI Development
- Uses Rich library for terminal UI
- Progress bars for long operations
- ASCII art and syntax highlighting for better UX

### Platform-Specific Considerations
- **Windows**: Check CUDA availability, handle cache paths properly
- **macOS**: May need sentencepiece/protobuf from Homebrew
- **Linux**: Standard paths, fewer edge cases
- **VMs**: Optimized tensor handling, CPU-only operation

## Common Issues and Solutions

1. **Tensor Shape Mismatch**: Clear model cache and disable use_cache during generation
2. **CUDA Errors on CPU Systems**: Ensure proper device detection and fallback to CPU
3. **Memory Issues**: Recommend smaller models based on available RAM
4. **Import Errors**: Check virtual environment activation and dependency installation