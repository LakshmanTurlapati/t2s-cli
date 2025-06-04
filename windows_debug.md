# Windows Debug Guide for T2S CLI

## Installing T2S CLI with Windows Fixes

### Install from Windows Branch (Recommended for Windows Users)
```bash
# Install directly from the Windows branch with all fixes
pip install git+https://github.com/lakshmanturlapati/T2S.git@Windows

# Or clone and install locally:
git clone -b Windows https://github.com/lakshmanturlapati/T2S.git
cd T2S
pip install -e .
```

### Verify Windows-Optimized Installation
```bash
# Test the installation
t2s --version

# Check if Windows fixes are active
python -c "import platform; print(f'Platform: {platform.system()}')"
```

## Gemma 3 (4B) Vision Model Installation Error on Windows

### Problem
Error when downloading Gemma 3 (4B): `'vision_tower.vision_model.embeddings.patch_embedding.weight'`

This error occurs because **Gemma 3 4B IS a multimodal vision model** that includes vision components. The error indicates issues with loading the vision tower components properly on Windows.

### Root Cause
- Gemma 3 4B is a **vision-language model** (not text-only)
- Windows-specific issues with multimodal model component loading
- Pipeline approach may resolve vision tower loading conflicts
- Memory alignment issues with vision embeddings on Windows

### Solutions (Try in Order)

#### Solution 1: Use Pipeline Approach (RECOMMENDED)
Based on successful Mac fixes, use the pipeline approach to bypass vision loading issues:

```python
import torch
from transformers import pipeline

# Use pipeline instead of direct model loading
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# Test with text-only first
result = pipe("What is machine learning?")
print(result)
```

#### Solution 2: Force CPU Mode with Pipeline
If GPU fails, force CPU mode:

```python
import torch
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it", 
    device="cpu",
    torch_dtype=torch.float32
)
```

#### Solution 3: Environment Variables for Windows
Set these before running:

```cmd
set TRANSFORMERS_CACHE=%USERPROFILE%\.cache\transformers
set HF_HOME=%USERPROFILE%\.cache\huggingface
set PYTORCH_TRANSFORMERS_CACHE=%USERPROFILE%\.cache\transformers
set TORCH_USE_CUDA_DSA=1
```

#### Solution 4: Update Dependencies for Windows
```bash
# Uninstall and reinstall with Windows-optimized versions
pip uninstall torch transformers accelerate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.49.0 accelerate>=0.20.0

# Install vision dependencies
pip install Pillow>=8.0.0 opencv-python>=4.5.0
```

#### Solution 5: Alternative Installation Method
Use Ollama for simpler installation:

```bash
# Install Ollama for Windows
# Download from https://ollama.ai
ollama pull gemma3:4b
ollama run gemma3:4b
```

#### Solution 6: Use AutoProcessor Approach
```python
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    # Fallback to CPU
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
```

#### Solution 7: Clear All Caches and Retry
```cmd
# Clear all model caches
rd /s /q "%USERPROFILE%\.cache\huggingface"
rd /s /q "%USERPROFILE%\.cache\transformers"
rd /s /q "%USERPROFILE%\.cache\torch"

# Clear Python cache
rd /s /q "__pycache__"

# Restart and try again
```

### Windows-Specific Fixes

#### Memory Alignment Fix
Add this to your Python script:
```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Enable device-side assertions
```

#### Visual Studio Runtime
Install Visual C++ Redistributable:
- Download from Microsoft: `vc_redist.x64.exe`
- Install and restart

#### Windows Defender Exclusions
Add these folders to Windows Defender exclusions:
- `%USERPROFILE%\.cache\huggingface`
- Your Python environment folder
- T2S installation directory

### Alternative Models for Windows
If Gemma 3 4B continues to fail:

1. **Gemma 2 (2B-IT)** - Text-only, more stable
   ```python
   pipeline("text-generation", model="google/gemma-2-2b-it")
   ```

2. **SmolVLM (500M)** - Smaller vision model
   ```python
   pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-256M-Instruct")
   ```

3. **Qwen2-VL (2B)** - Alternative vision model
   ```python
   pipeline("image-text-to-text", model="Qwen/Qwen2-VL-2B-Instruct")
   ```

### Testing Multimodal Functionality
Once installed, test with:

```python
from PIL import Image
import requests

# Test image processing
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

result = pipe("What do you see in this image?", image=image)
print(result)
```

### Model Size Requirements for Windows

| Model | RAM Required | VRAM Required | Speed | Accuracy | Windows Compatibility |
|-------|-------------|---------------|-------|----------|---------------------|
| Gemma 3 (1B) | 4GB | 2GB | Fast | 70% | ⭐⭐⭐⭐⭐ |
| Gemma 3 (4B) | 8GB | 6GB | Medium | 85% | ⭐⭐⭐⭐ |
| SmolVLM (500M) | 2GB | 1GB | Very Fast | 65% | ⭐⭐⭐⭐⭐ |
| Qwen2-VL (2B) | 6GB | 4GB | Fast | 80% | ⭐⭐⭐⭐ |

### Debugging Steps
1. **Check GPU compatibility**: Run `nvidia-smi` if using CUDA
2. **Verify Python version**: Should be 3.9-3.11
3. **Check available memory**: Task Manager → Performance
4. **Test with smaller model first**: Try Gemma 3 1B
5. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Getting Help
If issues persist:
1. **Check Windows Event Viewer** for system errors
2. **Run Windows Memory Diagnostic** 
3. **Update GPU drivers** (NVIDIA/AMD)
4. **Disable Windows Fast Startup** (may interfere with CUDA)
5. **Run from Windows Terminal** (not Command Prompt)

### Success Verification
You'll know it's working when you see:
```
Loading checkpoint shards: 100%
Model loaded successfully with vision capabilities
Pipeline ready for image-text tasks
```

---
*Last updated: June 2025*
*Includes Gemma 3 multimodal vision model fixes*
