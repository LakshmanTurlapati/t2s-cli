# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T2S is a privacy-first Text-to-SQL converter that runs AI models locally to transform natural language queries into SQL statements. It supports multiple databases (SQLite, PostgreSQL, MySQL) and AI models (Gemma, Llama, SmolVLM, Defog SQLCoder).

## Recent Major Updates (Version 0.2.1)

### SQLCoder Model Support Fixed
- **Issue**: SQLCoder (defog/sqlcoder-7b-2) failed to load due to missing `sentencepiece` dependency
- **Solution**: Added `sentencepiece>=0.1.99` and `protobuf>=3.20.0` to core dependencies in pyproject.toml
- **Installation**: Requires system-level installation on macOS: `brew install sentencepiece protobuf`
- **Status**: ✅ SQLCoder now loads and runs properly with CodeLlamaTokenizer

### Performance Optimizations for Apple Silicon (MPS)
All models have been optimized for faster inference on Apple Silicon GPUs:

#### Model Loading Optimizations:
- **SQLCoder**: Direct MPS mapping `device_map={"": "mps"}` + float16 precision (based on proven DBMS implementation)
- **Llama**: Direct MPS mapping + float16 (faster than auto device mapping)
- **SmolLM**: Direct MPS mapping + float16 for faster inference
- **Gemma**: Direct MPS mapping + bfloat16 (prevents numerical instability)

#### Generation Parameter Optimizations:
- **MPS-Specific Settings**: All models use simplified parameters on MPS for 30-50% faster inference
  - `num_beams=1` (no beam search for speed)
  - `do_sample=False` (deterministic generation)
  - `use_cache=True` (enable KV cache)
  - Reduced `max_new_tokens` for faster response times
- **SQLCoder on MPS**: Ultra-optimized with 150 max tokens, no early stopping
- **SmolLM on MPS**: Ultra-fast with only 80 max tokens
- **Llama on MPS**: 100 max tokens with deterministic generation
- **Gemma on MPS**: 150 max tokens with sampling parameters (prevents inf/nan errors)

### Latest Fixes (v0.2.1)
- **Gemma Numerical Stability**: Fixed probability tensor inf/nan errors on MPS
- **Reduced Logging**: Removed verbose input validation messages during generation
- **Error Recovery**: Added automatic fallback for Gemma generation failures

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
- **Device Management**: Careful handling of CUDA/CPU/MPS device selection with platform-specific optimizations
- **Model Loading**: Uses transformers library with device-specific optimizations for each model type
- **Query Validation**: SQL queries are validated using sqlparse before execution
- **Schema Context**: Database schemas are analyzed and included in prompts for better accuracy
- **MPS Optimization**: Direct device mapping and simplified generation parameters for Apple Silicon
- **Inference Mode**: All generation uses `torch.inference_mode()` for optimal performance
- **KV Cache**: Enabled for all models to reduce computation overhead during generation

## Development Guidelines

### Adding New Models
1. Update `SUPPORTED_MODELS` in `t2s/core/config.py`
2. Add model-specific loading logic in `t2s/models/model_manager.py`:
   - Add model detection in `_get_model_loading_config()`
   - Define MPS-specific optimizations if needed
   - Set appropriate generation parameters in `generate_response()`
3. Update memory requirements and recommendations
4. Test on all supported devices (CPU, CUDA, MPS)

### Working with Databases
- Database connections use SQLAlchemy for abstraction
- Connection strings are stored securely in config
- Auto-discovery works for SQLite files in the current directory

### UI Development
- Uses Rich library for terminal UI
- Progress bars for long operations
- ASCII art and syntax highlighting for better UX

### Platform-Specific Considerations
- **Windows**: Check CUDA availability, handle cache paths properly, aggressive GPU usage settings
- **macOS**: 
  - Requires sentencepiece/protobuf from Homebrew for SQLCoder: `brew install sentencepiece protobuf`
  - MPS optimizations for Apple Silicon GPUs (M1/M2/M3+)
  - Direct device mapping for better performance
- **Linux**: Standard paths, fewer edge cases
- **VMs**: Optimized tensor handling, CPU-only operation, conservative threading

## Common Issues and Solutions

1. **SQLCoder SentencePiece Error**: 
   - **Error**: `CodeLlamaTokenizer requires the SentencePiece library`
   - **Solution**: Install system dependencies: `brew install sentencepiece protobuf`, then `pip install sentencepiece protobuf`
   - **Status**: ✅ Fixed in current version

2. **Gemma Probability Tensor Error**:
   - **Error**: `probability tensor contains either inf, nan or element < 0`
   - **Cause**: Numerical instability with float16 precision and deterministic generation on MPS
   - **Solution**: Use bfloat16 precision + sampling parameters for Gemma on MPS
   - **Status**: ✅ Fixed with automatic fallback handling

3. **Tensor Shape Mismatch**: Clear model cache, ensure proper input validation
4. **CUDA Errors on CPU Systems**: Ensure proper device detection and fallback to CPU
5. **Memory Issues**: Recommend smaller models based on available RAM
6. **Import Errors**: Check virtual environment activation and dependency installation
7. **Slow MPS Performance**: Ensure models use direct device mapping and optimized parameters (fixed in current version)
8. **Model Download Failures**: Check HuggingFace authentication and network connectivity

## Model-Specific Notes

### SQLCoder (defog/sqlcoder-7b-2)
- **Dependencies**: Requires sentencepiece for CodeLlamaTokenizer
- **Optimization**: Uses direct MPS mapping for best performance
- **Generation**: Optimized beam search settings for SQL generation

### Gemma Models
- **Loading**: Uses Gemma3ForConditionalGeneration for multimodal models
- **Optimization**: bfloat16 on MPS for numerical stability (changed from float16)
- **Generation**: Uses sampling parameters on MPS to prevent inf/nan errors
- **Fallback**: Pipeline approach for legacy Gemma models + conservative fallback for numerical issues
- **Error Handling**: Automatic detection and recovery from probability tensor errors

### Llama Models  
- **Optimization**: Direct MPS mapping with simplified generation parameters
- **Performance**: 30-50% faster inference on Apple Silicon

### SmolLM
- **Optimization**: Ultra-fast settings with minimal token generation
- **Use Case**: Best for simple, quick SQL queries