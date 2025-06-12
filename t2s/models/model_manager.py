"""Model management for T2S - handles downloading and running AI models."""

import os
import re
import asyncio
import platform
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Suppress transformers progress bars for cleaner UI
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Windows-specific environment variables for better model loading
if platform.system() == "Windows":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages on Windows
    os.environ["TORCH_USE_CUDA_DSA"] = "1"    # Enable device-side assertions for debugging
    
    # Set Windows-specific cache paths
    cache_dir = os.path.expanduser("~\\AppData\\Local\\t2s\\cache")
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = cache_dir

import torch

# Additional progress bar suppression
import warnings
warnings.filterwarnings("ignore")

# Suppress tqdm progress bars globally
try:
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except ImportError:
    pass

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoProcessor,  # For multimodal models like Gemma 3
    AutoModelForVision2Seq,  # For vision-language models
    Gemma3ForConditionalGeneration,  # For Gemma 3 multimodal models
    pipeline,
    BitsAndBytesConfig,
    logging as transformers_logging
)

# Set transformers logging to error level to suppress progress bars
transformers_logging.set_verbosity_error()

# Additional suppression for huggingface_hub
try:
    from huggingface_hub import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

from huggingface_hub import login, logout, whoami, HfApi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
import requests

from ..core.config import Config


class ModelManager:
    """Manages AI models for T2S."""
    
    def __init__(self, config: Config):
        """Initialize the model manager."""
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.current_model = None
        self.current_tokenizer = None
        self.current_pipeline = None
        self.current_processor = None
        self.hf_api = HfApi()
        
        # Detect virtualized environment early
        self.is_virtualized = self._is_virtualized_environment()
        
        # Show platform information
        os_name = platform.system()
        os_version = platform.release()
        if self.is_virtualized:
            platform_info = f"{os_name} {os_version} (Virtual Machine)"
        else:
            platform_info = f"{os_name} {os_version}"
        self.console.print(f"[blue]Platform: {platform_info}[/blue]")
        
        # Setup device with detailed Windows GPU detection
        self.device = self._get_optimal_device()
        self._report_device_info()
    
    def _get_optimal_device(self) -> str:
        """Determine the optimal device for model inference with detailed Windows GPU detection."""
        
        # Detailed CUDA detection for Windows
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Force CUDA device selection
                device = "cuda"
                
                if self._is_windows_system():
                    # Extra validation for Windows NVIDIA GPU
                    try:
                        # Test CUDA functionality
                        test_tensor = torch.tensor([1.0]).cuda()
                        test_result = test_tensor.cpu()
                        return device
                        
                    except Exception as e:
                        self.console.print(f"[red]CUDA test failed on Windows: {e}[/red]")
                        self.console.print(f"[yellow]Falling back to CPU mode[/yellow]")
                        return "cpu"
                else:
                    return device
                    
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS
            return "mps"
        else:
            if self._is_windows_system():
                # Check if PyTorch was built with CUDA
                cuda_version = torch.version.cuda
                if cuda_version is None:
                    # Check if NVIDIA GPU is available via nvidia-ml-py
                    if self._has_nvidia_gpu():
                        self.console.print(f"[yellow]Detected NVIDIA GPU but PyTorch has no CUDA support[/yellow]")
                        if self._auto_install_cuda_pytorch():
                            self.console.print(f"[green]✓ CUDA PyTorch installed! Please restart T2S to use GPU[/green]")
                            return "cpu"  # Still use CPU for this session
                        else:
                            self.console.print(f"[red]❌ PyTorch was installed without CUDA support![/red]")
                            self.console.print(f"[yellow]To fix: pip uninstall torch torchvision torchaudio[/yellow]")
                            self.console.print(f"[yellow]Then: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/yellow]")
                    else:
                        self.console.print(f"[red]❌ PyTorch was installed without CUDA support![/red]")
                        self.console.print(f"[yellow]To fix: pip uninstall torch torchvision torchaudio[/yellow]")
                        self.console.print(f"[yellow]Then: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/yellow]")
                else:
                    self.console.print(f"[yellow]PyTorch has CUDA {cuda_version} support but no GPU detected[/yellow]")
                    self.console.print(f"[yellow]Check NVIDIA drivers: nvidia-smi[/yellow]")
            return "cpu"
    
    def _has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available on the system."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and 'NVIDIA' in result.stdout
        except Exception:
            return False
    
    def _auto_install_cuda_pytorch(self) -> bool:
        """Automatically install CUDA-enabled PyTorch on Windows."""
        try:
            import subprocess
            import sys
            
            self.console.print(f"[blue]Attempting to automatically install CUDA PyTorch...[/blue]")
            
            # Uninstall CPU-only PyTorch
            self.console.print(f"[yellow]Uninstalling CPU-only PyTorch...[/yellow]")
            subprocess.run([
                sys.executable, '-m', 'pip', 'uninstall', 
                'torch', 'torchvision', 'torchaudio', '-y'
            ], check=True, capture_output=True)
            
            # Install CUDA PyTorch
            self.console.print(f"[yellow]Installing CUDA PyTorch (this may take a few minutes)...[/yellow]")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu121'
            ], check=True, capture_output=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Auto-installation failed: {e}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Auto-installation error: {e}[/red]")
            return False
    
    def _report_device_info(self):
        """Report device information to the user."""        
        if self.device == "cuda":
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                self.console.print(f"[blue]Device: {self.device.upper()} - {gpu_name} ({gpu_memory:.1f} GB)[/blue]")
            else:
                self.console.print(f"[blue]Device: {self.device.upper()}[/blue]")
                
        elif self.device == "mps":
            self.console.print(f"[blue]Device: {self.device.upper()} (Apple Silicon)[/blue]")
            
        elif self.device == "cpu":
            # Only show CPU warnings when there might be GPU available
            if self._is_windows_system():
                cuda_version = torch.version.cuda
                if cuda_version is None and self._has_nvidia_gpu():
                    # GPU available but PyTorch doesn't have CUDA - this was already handled in _get_optimal_device
                    pass
            self.console.print(f"[blue]Device: {self.device.upper()}[/blue]")
    
    def _is_gemma3_multimodal(self, model_config) -> bool:
        """Check if this is a Gemma 3 multimodal model."""
        model_id = model_config.hf_model_id.lower()
        return "gemma-3" in model_id and any(size in model_id for size in ["4b", "12b", "27b"])
    
    def _is_windows_system(self) -> bool:
        """Check if running on Windows."""
        return platform.system() == "Windows"
    
    def _is_virtualized_environment(self) -> bool:
        """Detect if running in a virtualized environment (VM)."""
        try:
            import subprocess
            import os
            
            # Check for common VM indicators
            vm_detected = False
            vm_type = ""
            
            # Check system info for VM signatures
            try:
                if platform.system() == "Darwin":  # macOS
                    # Check for Parallels Desktop
                    result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                          capture_output=True, text=True, timeout=5)
                    if 'Parallels' in result.stdout or 'Virtual' in result.stdout:
                        vm_detected = True
                        vm_type = "Parallels Desktop"
                    
                    # Check for other VM signatures
                    if not vm_detected:
                        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                              capture_output=True, text=True, timeout=5)
                        if any(vm_name in result.stdout.lower() for vm_name in ['vmware', 'virtualbox', 'qemu']):
                            vm_detected = True
                            vm_type = "Virtual Machine"
                        
                elif platform.system() == "Windows":
                    # Check for Windows VM indicators
                    result = subprocess.run(['wmic', 'computersystem', 'get', 'model'], 
                                          capture_output=True, text=True, timeout=5)
                    if any(vm_name in result.stdout.lower() for vm_name in ['vmware', 'virtualbox', 'parallels', 'virtual']):
                        vm_detected = True
                        vm_type = "Virtual Machine"
                        
                elif platform.system() == "Linux":
                    # Check for Linux VM indicators
                    if os.path.exists('/proc/cpuinfo'):
                        with open('/proc/cpuinfo', 'r') as f:
                            cpuinfo = f.read().lower()
                            if any(vm_name in cpuinfo for vm_name in ['vmware', 'virtualbox', 'qemu', 'kvm']):
                                vm_detected = True
                                vm_type = "Virtual Machine"
                
            except Exception:
                pass
            
            # Additional checks
            try:
                # Check environment variables
                vm_env_vars = {'PARALLELS_TOOLS_VERSION': 'Parallels Desktop', 
                              'VMWARE_VERSION': 'VMware', 
                              'VBOX_VERSION': 'VirtualBox'}
                for var, name in vm_env_vars.items():
                    if os.getenv(var):
                        vm_detected = True
                        vm_type = name
                        break
                        
            except Exception:
                pass
            
            if vm_detected:
                return True
                
            return False
            
        except Exception:
            return False
    
    def _setup_windows_env_for_gemma(self):
        """Setup Windows-specific environment variables for Gemma models."""
        if not self._is_windows_system():
            return
        
        # Windows-specific fixes for vision model loading
        env_vars = {
            "TORCH_USE_CUDA_DSA": "1",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "HF_HUB_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
    
    async def initialize(self) -> None:
        """Initialize the model manager and load the selected model."""
        selected_model = self.config.config.selected_model
        if not selected_model:
            self.console.print("[yellow]No model selected. Use configuration to select a model.[/yellow]")
            return
        
        if not self.config.is_model_downloaded(selected_model):
            self.console.print(f"[yellow]Model {selected_model} not downloaded. Use configuration to download it.[/yellow]")
            return
        
        await self.load_model(selected_model)
    
    async def download_model(self, model_id: str, progress_callback: Optional[callable] = None) -> bool:
        """Download and cache a model from HuggingFace."""
        if model_id not in self.config.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_id} not supported")
        
        model_config = self.config.SUPPORTED_MODELS[model_id]
        
        # Setup Windows environment if needed
        self._setup_windows_env_for_gemma()
        
        # Configure HTTP timeouts for large model downloads
        import os
        
        # Set environment variables for longer timeouts
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
        os.environ["REQUESTS_TIMEOUT"] = "300"
        
        # Configure requests with longer timeout
        try:
            import requests
            # Monkey patch requests to use longer timeout
            original_get = requests.get
            original_post = requests.post
            
            def patched_get(*args, **kwargs):
                kwargs.setdefault('timeout', 300)
                return original_get(*args, **kwargs)
            
            def patched_post(*args, **kwargs):
                kwargs.setdefault('timeout', 300)
                return original_post(*args, **kwargs)
            
            requests.get = patched_get
            requests.post = patched_post
        except Exception:
            pass  # Continue without timeout patching if it fails
        
        # Check authentication for gated models
        if self.config.config.huggingface_token:
            try:
                login(token=self.config.config.huggingface_token)
                self.console.print("[dim]Using HuggingFace authentication...[/dim]")
            except Exception as e:
                self.console.print(f"[red]HuggingFace authentication failed: {e}[/red]")
                return False
        
        model_path = self.config.get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                # Check if this is a Gemma 3 multimodal model
                is_gemma3_multimodal = self._is_gemma3_multimodal(model_config)
                
                # Check if this is a legacy Gemma model (non-multimodal)
                is_legacy_gemma = "gemma" in model_config.hf_model_id.lower() and not is_gemma3_multimodal
                
                # Check if this is a SmolLM model (text-only, uses standard loading)
                is_smollm_model = "smollm" in model_config.hf_model_id.lower()
                
                # Configure model loading based on device and model size
                model_kwargs = self._get_model_loading_config(model_config)
                
                if is_gemma3_multimodal:
                    # Use direct download approach for Gemma 3 multimodal models
                    self.console.print(f"[blue]Downloading Gemma 3 multimodal model for {self.device.upper()}...[/blue]")
                    
                    # Windows + CUDA: Use aggressive GPU settings
                    if self._is_windows_system() and self.device == "cuda":
                        self.console.print(f"[green]Using Windows CUDA acceleration[/green]")
                    
                    task1 = progress.add_task(f"Downloading {model_config.name} components...", total=100)
                    
                    try:
                        # First try to download with AutoProcessor (for true multimodal models)
                        progress.update(task1, completed=25)
                        
                        processor = None
                        tokenizer = None
                        
                        try:
                            processor = AutoProcessor.from_pretrained(
                                model_config.hf_model_id,
                                cache_dir=str(model_path),
                                local_files_only=False,
                                resume_download=True
                            )
                            self.console.print(f"[green]✓ Using AutoProcessor for {model_config.name}[/green]")
                        except Exception as processor_error:
                            self.console.print(f"[yellow]AutoProcessor failed, trying AutoTokenizer: {processor_error}[/yellow]")
                            # Fallback to AutoTokenizer for models that don't use AutoProcessor
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_config.hf_model_id,
                                cache_dir=str(model_path),
                                local_files_only=False,
                                resume_download=True
                            )
                            self.console.print(f"[green]✓ Using AutoTokenizer for {model_config.name}[/green]")
                        
                        progress.update(task1, completed=50)
                        
                        # Download model using optimized configuration
                        model_kwargs_for_download = self._get_model_loading_config(model_config)
                        model_kwargs_for_download.update({
                            "cache_dir": str(model_path),
                            "local_files_only": False,
                            "resume_download": True
                        })
                        
                        # Use Gemma3ForConditionalGeneration for Gemma 3 models
                        model = Gemma3ForConditionalGeneration.from_pretrained(
                            model_config.hf_model_id,
                            **model_kwargs_for_download
                        )
                        
                        progress.update(task1, completed=75)
                        
                        # Save components locally
                        if processor:
                            processor.save_pretrained(str(model_path))
                        if tokenizer:
                            tokenizer.save_pretrained(str(model_path))
                        model.save_pretrained(str(model_path))
                        
                        progress.update(task1, completed=100)
                        
                        self.console.print(f"[green]✓ Successfully downloaded Gemma 3 model for {self.device.upper()}[/green]")
                        
                        # Clean up to free memory
                        if processor:
                            del processor
                        if tokenizer:
                            del tokenizer
                        del model
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return True
                        
                    except Exception as e:
                        # Windows-specific error handling with fallback
                        if self._is_windows_system() and ("vision_tower" in str(e) or "patch_embedding" in str(e) or "out of memory" in str(e).lower()):
                            self.console.print(f"[red]Windows GPU error: {e}[/red]")
                            
                            # Try with reduced precision first
                            if self.device == "cuda" and "out of memory" not in str(e).lower():
                                self.console.print("[yellow]Trying Windows GPU with reduced precision...[/yellow]")
                                
                                try:
                                    # Try processor first, fallback to tokenizer
                                    processor = None
                                    tokenizer = None
                                    
                                    try:
                                        processor = AutoProcessor.from_pretrained(
                                            model_config.hf_model_id,
                                            cache_dir=str(model_path),
                                            local_files_only=False,
                                            resume_download=True
                                        )
                                    except Exception:
                                        tokenizer = AutoTokenizer.from_pretrained(
                                            model_config.hf_model_id,
                                            cache_dir=str(model_path),
                                            local_files_only=False,
                                            resume_download=True
                                        )
                                        
                                    # Use float32 and smaller batch size for stability
                                    reduced_kwargs = {
                                        "torch_dtype": torch.float32,
                                        "attn_implementation": "eager",
                                        "device_map": "auto",  # Still try GPU but with float32
                                        "cache_dir": str(model_path),
                                        "local_files_only": False,
                                        "resume_download": True
                                    }
                                    
                                    model = Gemma3ForConditionalGeneration.from_pretrained(
                                        model_config.hf_model_id,
                                        **reduced_kwargs
                                    )
                                    
                                    # Save components
                                    if processor:
                                        processor.save_pretrained(str(model_path))
                                    if tokenizer:
                                        tokenizer.save_pretrained(str(model_path))
                                    model.save_pretrained(str(model_path))
                                    
                                    progress.update(task1, completed=100)
                                    self.console.print(f"[green]✓ Windows GPU fallback successful (float32)[/green]")
                                    
                                    # Clean up
                                    if processor:
                                        del processor
                                    if tokenizer:
                                        del tokenizer
                                    del model
                                    
                                    return True
                                    
                                except Exception as reduced_error:
                                    self.console.print(f"[red]GPU fallback also failed: {reduced_error}[/red]")
                                
                            # Final CPU fallback
                            self.console.print("[yellow]Trying final CPU fallback...[/yellow]")
                            try:
                                # Try CPU-only approach as last resort
                                processor = None
                                tokenizer = None
                                
                                try:
                                    processor = AutoProcessor.from_pretrained(
                                        model_config.hf_model_id,
                                        cache_dir=str(model_path),
                                        local_files_only=False,
                                        resume_download=True
                                    )
                                except Exception:
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        model_config.hf_model_id,
                                        cache_dir=str(model_path),
                                        local_files_only=False,
                                        resume_download=True
                                    )
                                    
                                    cpu_model_kwargs = {
                                        "torch_dtype": torch.float32,
                                        "attn_implementation": "eager",
                                        "cache_dir": str(model_path),
                                        "local_files_only": False,
                                        "resume_download": True
                                        # No device_map for CPU
                                    }
                                    
                                    model = Gemma3ForConditionalGeneration.from_pretrained(
                                        model_config.hf_model_id,
                                        **cpu_model_kwargs
                                    )
                                    
                                    # Save components
                                    if processor:
                                        processor.save_pretrained(str(model_path))
                                    if tokenizer:
                                        tokenizer.save_pretrained(str(model_path))
                                    model.save_pretrained(str(model_path))
                                    
                                    progress.update(task1, completed=100)
                                    self.console.print(f"[yellow]✓ CPU fallback successful (will be slower)[/yellow]")
                                    
                                    # Clean up
                                    if processor:
                                        del processor
                                    if tokenizer:
                                        del tokenizer
                                    del model
                                    
                                    return True
                                    
                                except Exception as fallback_error:
                                    self.logger.error(f"All fallbacks failed: {fallback_error}")
                                    raise e
                            except Exception as cpu_fallback_error:
                                self.logger.error(f"CPU fallback failed: {cpu_fallback_error}")
                                raise e
                        else:
                            raise e
                
                elif is_legacy_gemma:
                    # Use text-generation pipeline for legacy Gemma models (non-multimodal)
                    self.console.print(f"[blue]Using pipeline-first approach for legacy Gemma model (bypassing SentencePiece issues)...[/blue]")
                    
                    task1 = progress.add_task(f"Downloading {model_config.name} via pipeline...", total=100)
                    
                    # Create pipeline directly - this worked in our test!
                    test_pipeline = pipeline(
                        "text-generation",  # Legacy Gemma models use text-generation
                        model=model_config.hf_model_id,  # Direct from hub like our test
                        torch_dtype=model_kwargs.get("torch_dtype", torch.bfloat16),
                        device_map=model_kwargs.get("device_map", "auto"),
                        model_kwargs={
                            "attn_implementation": model_kwargs.get("attn_implementation", "eager")
                        },
                        cache_dir=str(model_path)
                    )
                    
                    progress.update(task1, completed=50)
                    
                    # Extract and save the components  
                    tokenizer = test_pipeline.tokenizer
                    model = test_pipeline.model
                    
                    progress.update(task1, completed=75)
                    
                    # Save locally
                    tokenizer.save_pretrained(str(model_path))
                    model.save_pretrained(str(model_path))
                    
                    progress.update(task1, completed=100)
                    
                    self.console.print(f"[green]✓ Successfully downloaded legacy Gemma model using pipeline approach[/green]")
                    
                    # Clean up pipeline to free memory
                    del test_pipeline
                    del tokenizer
                    del model
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.console.print(f"[green]Successfully downloaded {model_config.name}![/green]")
                    
                    if progress_callback:
                        progress_callback(model_id, "completed")
                    
                    return True
                
                # SmolLM uses standard text model downloading (handled in else block below)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.console.print(f"[green]Successfully downloaded {model_config.name}![/green]")
                    
                    if progress_callback:
                        progress_callback(model_id, "completed")
                    
                    return True
                
                else:
                    # Original approach for non-Gemma models
                    # Download tokenizer
                    task1 = progress.add_task(f"Downloading {model_config.name} tokenizer...", total=100)
                    
                    # Simulate progress for tokenizer (small files)
                    for i in range(0, 101, 20):
                        progress.update(task1, completed=i)
                        await asyncio.sleep(0.1)  # Small delay to show progress
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_config.hf_model_id,
                        cache_dir=str(model_path),
                        # Increase timeout for large downloads
                        local_files_only=False,
                        trust_remote_code=False,
                        resume_download=True,
                        # Configure HTTP timeout settings
                        use_fast=True if hasattr(AutoTokenizer, 'use_fast') else None
                    )
                    progress.update(task1, completed=100)
                    
                    # Download model with more realistic progress
                    task2 = progress.add_task(f"Downloading {model_config.name} model...", total=100)
                    
                    # Simulate realistic download progress
                    async def simulate_download_progress():
                        # Simulate download phases with realistic timing
                        phases = [
                            (5, 0.3),   # Initial connection
                            (15, 0.5),  # Starting download
                            (35, 0.8),  # Main download chunk 1
                            (55, 1.0),  # Main download chunk 2
                            (75, 0.7),  # Main download chunk 3
                            (90, 0.5),  # Finalizing
                            (95, 0.3),  # Almost done
                        ]
                        
                        for target_progress, delay in phases:
                            progress.update(task2, completed=target_progress)
                            await asyncio.sleep(delay)
                    
                    # Start progress simulation
                    progress_task = asyncio.create_task(simulate_download_progress())
                    
                    # Load model - try CausalLM first, then Seq2Seq as fallback
                    model = None
                    loading_error = None
                    
                    # Check if this is a GPT-based model (only supports CausalLM)
                    is_gpt_model = any(name in model_config.hf_model_id.lower() for name in ['gpt', 'distilgpt'])
                    
                    try:
                        self.console.print(f"[blue]Attempting to load {model_config.name} as CausalLM...[/blue]")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_config.hf_model_id,
                            cache_dir=str(model_path),
                            # Increase timeout and enable resumable downloads
                            local_files_only=False,
                            trust_remote_code=False,
                            resume_download=True,
                            **model_kwargs
                        )
                        self.console.print(f"[green]✓ Successfully loaded as CausalLM[/green]")
                    except Exception as e:
                        loading_error = str(e)
                        self.logger.info(f"CausalLM failed for {model_config.hf_model_id}: {e}")
                        
                        # Only try Seq2Seq if it's not a GPT model
                        if not is_gpt_model:
                            self.console.print(f"[yellow]CausalLM loading failed, trying Seq2Seq...[/yellow]")
                            
                            try:
                                model = AutoModelForSeq2SeqLM.from_pretrained(
                                    model_config.hf_model_id,
                                    cache_dir=str(model_path),
                                    # Increase timeout and enable resumable downloads
                                    local_files_only=False,
                                    trust_remote_code=False,
                                    resume_download=True,
                                    **model_kwargs
                                )
                                self.console.print(f"[green]✓ Successfully loaded as Seq2Seq[/green]")
                            except Exception as e2:
                                self.logger.error(f"Both model types failed for {model_config.hf_model_id}")
                                self.logger.error(f"CausalLM error: {loading_error}")
                                self.logger.error(f"Seq2Seq error: {e2}")
                                
                                # Provide helpful error message
                                if "Unrecognized configuration class" in str(e2):
                                    raise RuntimeError(
                                        f"Model {model_config.hf_model_id} configuration not supported. "
                                        f"This model may require a different transformers version or "
                                        f"may not be compatible with the current setup."
                                    )
                                else:
                                    raise RuntimeError(f"Failed to load model with both CausalLM and Seq2Seq: {e2}")
                        else:
                            # For GPT models, don't try Seq2Seq
                            self.logger.error(f"CausalLM failed for GPT model {model_config.hf_model_id}: {loading_error}")
                            
                            # Check for specific issues and provide better error messages
                            if "accelerate" in loading_error:
                                raise RuntimeError(f"Missing accelerate package. Please install with: pip install accelerate")
                            elif "device_map" in loading_error:
                                raise RuntimeError(f"Device mapping issue. Try without device_map or install accelerate.")
                            else:
                                raise RuntimeError(f"Failed to load GPT model: {loading_error}")
                    
                    if model is None:
                        raise RuntimeError("Model loading failed - no model object created")
                    
                    progress.update(task2, completed=100)
                    
                    # Stop progress monitoring
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    
                    # Save locally
                    tokenizer.save_pretrained(str(model_path))
                    model.save_pretrained(str(model_path))
                    
                    self.console.print(f"[green]Successfully downloaded {model_config.name}![/green]")
                    
                    if progress_callback:
                        progress_callback(model_id, "completed")
                    
                    return True
                
        except Exception as e:
            self.logger.error(f"Error downloading model {model_id}: {e}")
            self.console.print(f"[red]Error downloading model: {e}[/red]")
            
            # Clean up partial download
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path, ignore_errors=True)
            
            if progress_callback:
                progress_callback(model_id, "error", str(e))
            
            return False
    
    def _get_model_loading_config(self, model_config) -> Dict[str, Any]:
        """Get model loading configuration based on device and model size with aggressive Windows GPU usage."""
        config = {}
        
        # Check if this is a Gemma 3 multimodal model
        is_gemma3_multimodal = self._is_gemma3_multimodal(model_config)
        
        # Check if this is a legacy Gemma model (non-multimodal)
        is_legacy_gemma = "gemma" in model_config.hf_model_id.lower() and not is_gemma3_multimodal
        
        # Check if this is a SmolLM model (text-only, uses standard loading)
        is_smollm_model = "smollm" in model_config.hf_model_id.lower()
        
        # Check if this is SQLCoder model
        is_sqlcoder = "sqlcoder" in model_config.hf_model_id.lower()
        
        # VM-specific optimizations
        if self.is_virtualized:
            # Force more conservative settings for VMs
            config["torch_dtype"] = torch.float32  # More stable in VMs
            config["low_cpu_mem_usage"] = True
            config["device_map"] = None  # Avoid device mapping issues in VMs
            config["use_cache"] = False  # Disable caching to prevent state issues
            
            # VM-specific memory and processing settings
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = False  # Disable MKL-DNN for VM compatibility
            
            # Set conservative thread counts for VMs
            torch.set_num_threads(2)  # Conservative threading in VMs
            
            return config
        
        # SQLCoder-specific optimizations for MPS (based on DBMS implementation)
        if is_sqlcoder and self.device == "mps":
            config["torch_dtype"] = torch.float16  # Use float16 for MPS as proven in DBMS
            config["device_map"] = {"": self.device}  # Direct mapping to MPS
            config["trust_remote_code"] = True
            self.console.print(f"[green]Using optimized MPS settings for SQLCoder[/green]")
            return config
        
        # Optimized settings for specific models on MPS
        if self.device == "mps":
            # Llama optimization for MPS
            if "llama" in model_config.hf_model_id.lower():
                config["torch_dtype"] = torch.float16  # float16 is faster than bfloat16 on MPS
                config["device_map"] = {"": self.device}  # Direct mapping like SQLCoder
                config["trust_remote_code"] = True
                self.console.print(f"[green]Using optimized MPS settings for Llama[/green]")
                return config
            
            # SmolLM optimization for MPS
            elif is_smollm_model:
                config["torch_dtype"] = torch.float16
                config["device_map"] = {"": self.device}  # Direct mapping for faster inference
                config["low_cpu_mem_usage"] = True
                self.console.print(f"[green]Using optimized MPS settings for SmolLM[/green]")
                return config
        
        # Windows-specific optimizations - AGGRESSIVE GPU USAGE
        if self._is_windows_system() and self.device == "cuda":
            if is_gemma3_multimodal:
                # Use CUDA aggressively for Gemma 3 on Windows
                config["torch_dtype"] = torch.float16  # Use float16 for better GPU performance
                config["device_map"] = "auto"  # Enable device_map for better GPU utilization
                config["attn_implementation"] = "eager"  # Most stable attention
                config["low_cpu_mem_usage"] = True  # Keep more on GPU
                self.console.print(f"[green]Windows + CUDA: Using aggressive GPU settings for Gemma 3[/green]")
                return config
            else:
                # For other models on Windows + CUDA
                config["torch_dtype"] = torch.float16
                config["device_map"] = "auto"
                config["low_cpu_mem_usage"] = True
                self.console.print(f"[green]Windows + CUDA: Using aggressive GPU settings[/green]")
        elif self._is_windows_system() and self.device == "cpu":
            # CPU fallback for Windows
            if is_gemma3_multimodal:
                config["torch_dtype"] = torch.float32  # More stable on CPU
                config["device_map"] = None  # No device mapping for CPU
                config["attn_implementation"] = "eager"
                self.console.print(f"[yellow]⚠️  Windows + CPU: Using CPU-optimized settings[/yellow]")
                return config
        
        # Use quantization for large models or limited memory - but NOT for MPS or SQLCoder
        if model_config.size.value == "large" and self.device == "cuda" and not is_sqlcoder:
            # Use 4-bit quantization for CUDA to fit large models
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.console.print(f"[blue]Using 4-bit quantization for large model on GPU[/blue]")
        else:
            # Use appropriate precision for device
            if self.device == "mps":
                # For Apple Silicon - use bfloat16 for Gemma to prevent numerical instability
                if is_legacy_gemma or is_gemma3_multimodal:
                    config["torch_dtype"] = torch.bfloat16  # More stable for Gemma on MPS
                    config["device_map"] = {"": self.device}  # Direct mapping for Gemma
                    config["trust_remote_code"] = True
                    self.console.print(f"[green]Using stable MPS settings for Gemma (bfloat16)[/green]")
                    return config  # Return early to avoid device_map override
                else:
                    # For other models, use float16
                    config["torch_dtype"] = torch.float16
                    
            elif self.device == "cpu":
                config["torch_dtype"] = torch.float32
            else:  # CUDA
                # For CUDA, use float16 for better performance
                config["torch_dtype"] = torch.float16
        
        # Set device map for auto distribution - ALWAYS use for GPU (except SQLCoder and Gemma on MPS)
        if self.device != "cpu" and not (is_sqlcoder and self.device == "mps") and not ((is_legacy_gemma or is_gemma3_multimodal) and self.device == "mps"):
            # Use device_map for CUDA/MPS for better performance
            config["device_map"] = "auto"
            config["low_cpu_mem_usage"] = True  # Keep more on GPU
        
        # Add attention implementation for Gemma models to avoid conflicts
        if is_legacy_gemma or is_gemma3_multimodal:
            config["attn_implementation"] = "eager"
        
        return config
    
    async def _requires_huggingface_auth(self, model_id: str) -> bool:
        """Check if a model requires HuggingFace authentication."""
        try:
            # Try to access model info without authentication
            response = requests.get(f"https://huggingface.co/api/models/{model_id}")
            return response.status_code == 401
        except Exception:
            return False
    
    async def load_model(self, model_id: str) -> bool:
        """Load a model for inference."""
        if model_id not in self.config.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_id} is not supported")
        
        if not self.config.is_model_downloaded(model_id):
            self.console.print(f"[red]Model {model_id} is not downloaded[/red]")
            return False
        
        # Check for corruption and clean up if needed
        was_corrupted = await self.cleanup_corrupted_model(model_id)
        if was_corrupted:
            self.console.print(f"[red]Model {model_id} was corrupted and has been cleaned up. Please re-download.[/red]")
            return False
        
        model_path = self.config.get_model_path(model_id)
        model_config = self.config.SUPPORTED_MODELS[model_id]
        
        # Temporarily suppress all progress bars and verbose output
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        
        # Store original environment variables
        original_env = {}
        suppress_env_vars = {
            "TRANSFORMERS_VERBOSITY": "error",
            "HF_HUB_VERBOSITY": "error", 
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "TQDM_DISABLE": "1",  # Disable tqdm progress bars
            "HF_HUB_DISABLE_TELEMETRY": "1",  # Disable telemetry
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",  # Disable token warnings
            "TRANSFORMERS_CACHE": str(self.config.get_models_dir()),  # Set cache dir
        }
        
        for key, value in suppress_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Show loading animation to user BEFORE suppressing output
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Loading {task.description}[/cyan]"),
                console=self.console,
                transient=False  # Keep visible until completion
            ) as progress:
                task = progress.add_task(f"{model_config.name}", total=None)
                
                # Small delay to ensure spinner is visible
                await asyncio.sleep(0.1)
                
                # Redirect stderr to suppress progress bars, but keep stdout for our spinner
                stderr_buffer = StringIO()
                
                with redirect_stderr(stderr_buffer):
                    # Get model loading configuration
                    model_kwargs = self._get_model_loading_config(model_config)
                    
                    # Setup Windows environment if needed
                    self._setup_windows_env_for_gemma()
                    
                    # Check if this is a Gemma 3 multimodal model
                    is_gemma3_multimodal = self._is_gemma3_multimodal(model_config)
                    
                    # Check if this is a legacy Gemma model (non-multimodal)
                    is_legacy_gemma = "gemma" in model_config.hf_model_id.lower() and not is_gemma3_multimodal
                    
                    # Check if this is a SmolLM model (text-only, uses standard loading)
                    is_smollm_model = "smollm" in model_config.hf_model_id.lower()
                    
                    if is_gemma3_multimodal:
                        # Use direct approach for Gemma 3 models with processor/tokenizer fallback
                        try:
                            self.console.print(f"[blue]Loading {model_config.name} as Gemma 3 model on {self.device.upper()}...[/blue]")
                            
                            # Windows + CUDA: Report GPU usage
                            if self._is_windows_system() and self.device == "cuda":
                                self.console.print(f"[green]Using Windows CUDA acceleration[/green]")
                            
                            # Try to load processor first, fallback to tokenizer
                            processor = None
                            tokenizer = None
                            
                            try:
                                self.current_processor = AutoProcessor.from_pretrained(str(model_path))
                                processor = self.current_processor
                                self.console.print(f"[green]✓ Loaded AutoProcessor for {model_config.name}[/green]")
                            except Exception as processor_error:
                                self.console.print(f"[yellow]AutoProcessor failed, using AutoTokenizer: {processor_error}[/yellow]")
                                self.current_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                                tokenizer = self.current_tokenizer
                                self.current_processor = None
                                self.console.print(f"[green]✓ Loaded AutoTokenizer for {model_config.name}[/green]")
                            
                            # Load model with optimized settings
                            model_load_kwargs = self._get_model_loading_config(model_config)
                            
                            self.current_model = Gemma3ForConditionalGeneration.from_pretrained(
                                str(model_path),
                                **model_load_kwargs
                            )
                            
                            # For consistency, we don't use pipeline for Gemma 3
                            self.current_pipeline = None
                            
                            # Ensure tokenizer has necessary special tokens if using tokenizer
                            if self.current_tokenizer and self.current_tokenizer.pad_token is None:
                                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
                            
                            # Report successful GPU loading
                            if self.device == "cuda":
                                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                                self.console.print(f"[green]✓ Model loaded on GPU (using {gpu_memory_used:.1f} GB VRAM)[/green]")
                            
                            self.console.print(f"[green]✓ Successfully loaded {model_config.name} on {self.device.upper()}[/green]")
                            
                        except Exception as e:
                            # Windows-specific error handling with aggressive retry
                            if self._is_windows_system() and ("vision_tower" in str(e) or "patch_embedding" in str(e) or "out of memory" in str(e).lower()):
                                self.console.print(f"[red]Windows GPU error: {e}[/red]")
                                
                                # Try with reduced precision first if not OOM
                                if self.device == "cuda" and "out of memory" not in str(e).lower():
                                    self.console.print("[yellow]Trying reduced precision on GPU...[/yellow]")
                                    
                                    try:
                                        # Reload with float32 but keep on GPU
                                        processor = None
                                        tokenizer = None
                                        
                                        try:
                                            self.current_processor = AutoProcessor.from_pretrained(str(model_path))
                                            processor = self.current_processor
                                        except Exception:
                                            self.current_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                                            tokenizer = self.current_tokenizer
                                            self.current_processor = None
                                            
                                            # Ensure tokenizer has necessary special tokens
                                            if self.current_tokenizer.pad_token is None:
                                                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
                                        
                                        # Use float32 for stability but keep device_map for GPU
                                        reduced_kwargs = {
                                            "torch_dtype": torch.float32,
                                            "attn_implementation": "eager",
                                            "device_map": "auto",  # Still use GPU
                                            "low_cpu_mem_usage": True
                                        }
                                        
                                        self.current_model = Gemma3ForConditionalGeneration.from_pretrained(
                                            str(model_path),
                                            **reduced_kwargs
                                        )
                                        
                                        self.current_pipeline = None
                                        
                                        # Report successful GPU loading with reduced precision
                                        gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                                        self.console.print(f"[green]✓ Loaded on GPU with float32 (using {gpu_memory_used:.1f} GB VRAM)[/green]")
                                        
                                    except Exception as reduced_error:
                                        self.console.print(f"[red]Reduced precision failed: {reduced_error}[/red]")
                                        raise RuntimeError(f"Failed to load Gemma 3 on Windows GPU: {e}")
                                else:
                                    # OOM or other critical error - don't fallback to CPU, inform user
                                    if "out of memory" in str(e).lower():
                                        self.console.print(f"[red]GPU out of memory! Model too large for available VRAM.[/red]")
                                        raise RuntimeError(f"GPU out of memory. Try a smaller model or upgrade GPU: {e}")
                                    else:
                                        raise RuntimeError(f"Failed to load Gemma 3 on Windows GPU: {e}")
                            else:
                                self.logger.error(f"Gemma 3 loading failed: {e}")
                                raise
                    
                    elif is_legacy_gemma:
                        # Use pipeline-first approach for legacy Gemma models (fixes device mapping issues)
                        try:
                            self.console.print(f"[blue]Loading {model_config.name} using pipeline approach...[/blue]")
                            
                            # Create pipeline directly - let accelerate handle everything
                            self.current_pipeline = pipeline(
                                "text-generation",
                                model=str(model_path),
                                torch_dtype=model_kwargs.get("torch_dtype", torch.bfloat16),
                                device_map=model_kwargs.get("device_map", "auto"),
                                model_kwargs={
                                    "attn_implementation": model_kwargs.get("attn_implementation", "eager")
                                }
                            )
                            
                            # Extract tokenizer and model from pipeline
                            self.current_tokenizer = self.current_pipeline.tokenizer
                            self.current_model = self.current_pipeline.model
                            
                            # Ensure tokenizer has necessary special tokens
                            if self.current_tokenizer.pad_token is None:
                                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
                            
                            self.console.print(f"[green]✓ Successfully loaded {model_config.name} using pipeline approach[/green]")
                            
                        except Exception as e:
                            # Check for specific device mapping errors and provide better messages
                            if "device" in str(e).lower() and "accelerate" in str(e).lower():
                                raise RuntimeError(
                                    f"Device mapping conflict for Gemma model. "
                                    f"This is usually fixed by the pipeline approach, but failed: {e}"
                                )
                            else:
                                self.logger.error(f"Pipeline approach failed for Gemma model: {e}")
                                # Fall back to manual loading for Gemma
                                raise
                    
                    # SmolLM uses standard text model loading (handled in else block below)
                    
                    else:
                        # Use traditional approach for non-Gemma models
                        # Load tokenizer first
                        self.current_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                        
                        # Ensure tokenizer has necessary special tokens
                        if self.current_tokenizer.pad_token is None:
                            self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
                        
                        # Load model - try CausalLM first, then Seq2Seq as fallback
                        model = None
                        loading_error = None
                        
                        # Check if this is a GPT-based model (only supports CausalLM)
                        is_gpt_model = any(name in model_config.hf_model_id.lower() for name in ['gpt', 'distilgpt'])
                        
                        try:
                            self.console.print(f"[blue]Attempting to load {model_config.name} as CausalLM...[/blue]")
                            model = AutoModelForCausalLM.from_pretrained(
                                str(model_path),
                                **model_kwargs
                            )
                            self.console.print(f"[green]✓ Successfully loaded as CausalLM[/green]")
                        except Exception as e:
                            loading_error = str(e)
                            self.logger.info(f"CausalLM failed for {model_path}: {e}")
                            
                            # Only try Seq2Seq if it's not a GPT model
                            if not is_gpt_model:
                                self.console.print(f"[yellow]CausalLM loading failed, trying Seq2Seq...[/yellow]")
                                
                                try:
                                    model = AutoModelForSeq2SeqLM.from_pretrained(
                                        str(model_path),
                                        **model_kwargs
                                    )
                                    self.console.print(f"[green]✓ Successfully loaded as Seq2Seq[/green]")
                                except Exception as e2:
                                    self.logger.error(f"Both model types failed for {model_path}")
                                    self.logger.error(f"CausalLM error: {loading_error}")
                                    self.logger.error(f"Seq2Seq error: {e2}")
                                    
                                    # Provide helpful error message
                                    if "Unrecognized configuration class" in str(e2):
                                        raise RuntimeError(
                                            f"Model {model_path} configuration not supported. "
                                            f"This model may require a different transformers version or "
                                            f"may not be compatible with the current setup."
                                        )
                                    else:
                                        raise RuntimeError(f"Failed to load model with both CausalLM and Seq2Seq: {e2}")
                            else:
                                # For GPT models, don't try Seq2Seq
                                self.logger.error(f"CausalLM failed for GPT model {model_config.hf_model_id}: {loading_error}")
                                
                                # Check for specific issues and provide better error messages
                                if "accelerate" in loading_error:
                                    raise RuntimeError(f"Missing accelerate package. Please install with: pip install accelerate")
                                elif "device_map" in loading_error:
                                    raise RuntimeError(f"Device mapping issue. Try without device_map or install accelerate.")
                                else:
                                    raise RuntimeError(f"Failed to load GPT model: {loading_error}")
                        
                        if model is None:
                            raise RuntimeError("Model loading failed - no model object created")
                        
                        # Assign the successfully loaded model
                        self.current_model = model
                        
                        # Create pipeline for non-Gemma models
                        # Note: Don't manually specify device for pipeline when using device_map
                        pipeline_device = None
                        if "device_map" not in model_kwargs:
                            # Only specify device if we're not using device_map
                            if self.device == "cuda":
                                pipeline_device = 0
                            elif self.device == "cpu":
                                pipeline_device = -1
                            # For MPS, leave as None (pipeline will handle it)
                        
                        self.current_pipeline = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=self.current_tokenizer,
                            device=pipeline_device
                        )
                    
                    progress.update(task, completed=100)
                
            self.console.print(f"[green]Successfully loaded {model_config.name}![/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            self.console.print(f"[red]Error loading model: {e}[/red]")
            return False
        finally:
            # Restore original environment variables
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def _validate_input_tensors(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix tensor indices and shapes to prevent device-side asserts and shape mismatches."""
        validated_inputs = {}
        
        # Check if running in VM for enhanced validation
        is_vm = self.is_virtualized
        
        # First pass: collect all tensors and validate basic properties
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # VM-specific tensor fixes
                if is_vm:
                    # Force CPU-contiguous memory layout for VM compatibility
                    if value.device.type != 'cpu':
                        # Temporarily move to CPU for validation, then back
                        cpu_value = value.cpu()
                        if not cpu_value.is_contiguous():
                            cpu_value = cpu_value.contiguous()
                        value = cpu_value.to(value.device)
                    else:
                        if not value.is_contiguous():
                            value = value.contiguous()
                    
                    # VM: Force explicit clone to prevent memory address issues
                    value = value.clone()
                
                # Fix potential index out of bounds issues
                if key in ['input_ids', 'attention_mask']:
                    # Ensure indices are within valid range for vocabulary
                    if key == 'input_ids' and hasattr(self.current_tokenizer, 'vocab_size'):
                        # Clamp input_ids to valid vocabulary range
                        vocab_size = self.current_tokenizer.vocab_size
                        value = torch.clamp(value, 0, vocab_size - 1)
                    elif key == 'input_ids' and hasattr(self.current_processor, 'tokenizer') and hasattr(self.current_processor.tokenizer, 'vocab_size'):
                        # For processor-based models
                        vocab_size = self.current_processor.tokenizer.vocab_size
                        value = torch.clamp(value, 0, vocab_size - 1)
                
                # Ensure tensor is contiguous (fixes memory layout issues)
                if not value.is_contiguous():
                    value = value.contiguous()
                
                validated_inputs[key] = value
            else:
                validated_inputs[key] = value
        
        # Second pass: Fix shape mismatches between input_ids and attention_mask
        if 'input_ids' in validated_inputs and 'attention_mask' in validated_inputs:
            input_ids = validated_inputs['input_ids']
            attention_mask = validated_inputs['attention_mask']
            
            # Check if shapes match
            if input_ids.shape != attention_mask.shape:
                self.console.print(f"[yellow]Fixing tensor shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}[/yellow]")
                
                # VM-specific: Use more conservative approach
                if is_vm:
                    # In VMs, be more aggressive about ensuring exact matches
                    input_ids_len = input_ids.shape[-1] if len(input_ids.shape) > 0 else 0
                    attention_mask_len = attention_mask.shape[-1] if len(attention_mask.shape) > 0 else 0
                    
                    # Always truncate to the shorter length in VMs for stability
                    target_length = min(input_ids_len, attention_mask_len)
                    
                    if len(input_ids.shape) == 2:  # [batch_size, seq_len]
                        validated_inputs['input_ids'] = input_ids[:, :target_length].contiguous().clone()
                        validated_inputs['attention_mask'] = attention_mask[:, :target_length].contiguous().clone()
                    elif len(input_ids.shape) == 1:  # [seq_len]
                        validated_inputs['input_ids'] = input_ids[:target_length].contiguous().clone()
                        validated_inputs['attention_mask'] = attention_mask[:target_length].contiguous().clone()
                    
                    self.console.print(f"[green]✓ VM-optimized: Truncated to {target_length} tokens for stability[/green]")
                    
                else:
                    # Original logic for non-VM environments
                    # Get the target length (use the longer of the two)
                    input_ids_len = input_ids.shape[-1] if len(input_ids.shape) > 0 else 0
                    attention_mask_len = attention_mask.shape[-1] if len(attention_mask.shape) > 0 else 0
                    
                    if input_ids_len > attention_mask_len:
                        # Extend attention_mask to match input_ids
                        target_length = input_ids_len
                        if len(attention_mask.shape) == 2:  # [batch_size, seq_len]
                            batch_size = attention_mask.shape[0]
                            padding_length = target_length - attention_mask_len
                            padding = torch.ones(batch_size, padding_length, dtype=attention_mask.dtype, device=attention_mask.device)
                            validated_inputs['attention_mask'] = torch.cat([attention_mask, padding], dim=-1)
                        elif len(attention_mask.shape) == 1:  # [seq_len]
                            padding_length = target_length - attention_mask_len
                            padding = torch.ones(padding_length, dtype=attention_mask.dtype, device=attention_mask.device)
                            validated_inputs['attention_mask'] = torch.cat([attention_mask, padding], dim=-1)
                        
                    elif attention_mask_len > input_ids_len:
                        # Truncate attention_mask to match input_ids
                        target_length = input_ids_len
                        if len(attention_mask.shape) == 2:  # [batch_size, seq_len]
                            validated_inputs['attention_mask'] = attention_mask[:, :target_length]
                        elif len(attention_mask.shape) == 1:  # [seq_len]
                            validated_inputs['attention_mask'] = attention_mask[:target_length]
                    
                    self.console.print(f"[green]✓ Fixed tensor shapes: input_ids {validated_inputs['input_ids'].shape} = attention_mask {validated_inputs['attention_mask'].shape}[/green]")
        
        # Third pass: Ensure reasonable sequence lengths to prevent memory issues
        if 'input_ids' in validated_inputs:
            input_ids = validated_inputs['input_ids']
            # VM-specific: Use smaller max length for stability
            max_length = 2048 if is_vm else 4096  # More conservative in VMs
            
            if input_ids.shape[-1] > max_length:
                self.console.print(f"[yellow]Truncating input sequence from {input_ids.shape[-1]} to {max_length} tokens (VM-optimized: {is_vm})[/yellow]")
                
                if len(input_ids.shape) == 2:  # [batch_size, seq_len]
                    validated_inputs['input_ids'] = input_ids[:, :max_length]
                elif len(input_ids.shape) == 1:  # [seq_len]
                    validated_inputs['input_ids'] = input_ids[:max_length]
                
                # Also truncate attention_mask if it exists
                if 'attention_mask' in validated_inputs:
                    attention_mask = validated_inputs['attention_mask']
                    if len(attention_mask.shape) == 2:  # [batch_size, seq_len]
                        validated_inputs['attention_mask'] = attention_mask[:, :max_length]
                    elif len(attention_mask.shape) == 1:  # [seq_len]
                        validated_inputs['attention_mask'] = attention_mask[:max_length]
        
        # VM-specific: Final validation pass
        if is_vm:
            # Ensure all tensors are properly cloned and contiguous
            for key, value in validated_inputs.items():
                if isinstance(value, torch.Tensor):
                    validated_inputs[key] = value.contiguous().clone()
        
        return validated_inputs

    def _safe_to_device(self, tensor_dict: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Safely move tensors to device with proper error handling."""
        result = {}
        
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                try:
                    # Move tensor to device with proper dtype conversion
                    if device.type == 'cuda':
                        # Use non_blocking for better performance
                        result[key] = value.to(device, non_blocking=True)
                    else:
                        result[key] = value.to(device)
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        # Fallback to CPU for this tensor
                        self.console.print(f"[yellow]Moving {key} to CPU due to GPU error: {e}[/yellow]")
                        result[key] = value.cpu()
                    else:
                        raise e
            else:
                result[key] = value
        
        return result

    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL query using the loaded model."""
        
        # SmolLM uses standard pipeline generation (handled below)
        
        # Check if we have a Gemma 3 multimodal model
        current_model_id = self.config.config.selected_model
        if current_model_id and self._is_gemma3_multimodal(self.config.SUPPORTED_MODELS.get(current_model_id)):
            return await self._generate_sql_gemma3_multimodal(system_prompt, user_prompt)
        
        if not self.current_pipeline:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Replace the user_question placeholder in the prompt
        full_prompt = system_prompt.replace("{user_question}", user_prompt)
        
        # Pre-validate input with tokenizer to prevent shape mismatches
        if self.current_tokenizer:
            try:
                # Tokenize to check for issues and ensure proper format
                test_inputs = self.current_tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096  # Conservative limit
                )
                
                # Validate the tokenized inputs
                test_inputs = self._validate_input_tensors(test_inputs)
                
                # Check for reasonable input length
                if 'input_ids' in test_inputs and test_inputs['input_ids'].shape[-1] < 10:
                    self.console.print(f"[yellow]Warning: Very short input detected ({test_inputs['input_ids'].shape[-1]} tokens)[/yellow]")
                elif 'input_ids' in test_inputs and test_inputs['input_ids'].shape[-1] > 3000:
                    self.console.print(f"[yellow]Warning: Very long input detected ({test_inputs['input_ids'].shape[-1]} tokens), may cause issues[/yellow]")
                
                # Input validation completed successfully
                
            except Exception as tokenizer_error:
                self.console.print(f"[red]Input validation failed: {tokenizer_error}[/red]")
                # Try to clean the prompt
                if len(full_prompt) > 10000:  # Very long prompt
                    full_prompt = full_prompt[:5000] + "..."
                    self.console.print(f"[yellow]Truncated very long prompt to prevent tokenizer issues[/yellow]")
        
        try:
            # Set generation parameters based on model type
            if current_model_id and "sqlcoder" in current_model_id.lower():
                # SQLCoder-specific parameters optimized for MPS (based on DBMS implementation)
                using_mps = "mps" in str(self.current_model.device) if hasattr(self.current_model, 'device') else False
                
                if using_mps:
                    # Simplified parameters for MPS to prevent errors
                    generation_params = {
                        "max_new_tokens": 150,
                        "num_beams": 1,  # Simple beam search for MPS
                        "do_sample": False,
                        "early_stopping": False,  # Disable for MPS
                        "pad_token_id": self.current_tokenizer.pad_token_id,
                        "return_full_text": False
                    }
                else:
                    # Full parameters for CUDA/CPU
                    generation_params = {
                        "max_new_tokens": 150,
                        "do_sample": False,
                        "num_beams": 2,
                        "repetition_penalty": 1.1,
                        "length_penalty": 1.0,
                        "early_stopping": True,
                        "pad_token_id": self.current_tokenizer.pad_token_id,
                        "eos_token_id": self.current_tokenizer.eos_token_id,
                        "return_full_text": False
                    }
            elif current_model_id and any(model_name in current_model_id.lower() for model_name in ["llama", "phi", "mistral", "qwen"]):
                # Llama/Phi/Mistral/Qwen-specific parameters - optimized for speed and accuracy
                using_mps = "mps" in str(self.current_model.device) if hasattr(self.current_model, 'device') else False
                
                if using_mps:
                    # Optimized for MPS - simpler parameters for faster inference
                    generation_params = {
                        "max_new_tokens": 100,  # Reduced for faster generation
                        "do_sample": False,     # Deterministic for speed
                        "num_beams": 1,        # No beam search for MPS
                        "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                        "return_full_text": False,
                        "use_cache": True      # Enable KV cache for speed
                    }
                else:
                    # Full parameters for CUDA/CPU
                    generation_params = {
                        "max_new_tokens": 150,  # Conservative for focused SQL generation
                        "do_sample": True,      # Enable sampling but controlled
                        "temperature": 0.3,     # Lower temperature for more deterministic output
                        "top_p": 0.8,          # More focused nucleus sampling
                        "repetition_penalty": 1.2,  # Prevent repetitive output
                        "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                        "eos_token_id": self.current_tokenizer.eos_token_id,
                        "return_full_text": False,
                        "use_cache": True      # Enable KV cache for speed
                    }
            elif current_model_id and "smollm" in current_model_id.lower():
                # SmolLM-specific parameters - optimized for fast inference
                using_mps = "mps" in str(self.current_model.device) if hasattr(self.current_model, 'device') else False
                
                if using_mps:
                    # Ultra-fast settings for SmolLM on MPS
                    generation_params = {
                        "max_new_tokens": 80,   # Reduced for speed
                        "do_sample": False,     # Deterministic
                        "num_beams": 1,        # No beam search for speed
                        "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                        "return_full_text": False,
                        "use_cache": True      # Enable KV cache
                    }
                else:
                    # Standard parameters for CPU/CUDA
                    generation_params = {
                        "max_new_tokens": 100,  # Smaller for focused output
                        "do_sample": False,     # Deterministic for better results
                        "num_beams": 2,        # Reduced beam search for speed
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                        "eos_token_id": self.current_tokenizer.eos_token_id,
                        "return_full_text": False,
                        "use_cache": True      # Enable KV cache
                    }
            else:
                # General model parameters - optimized for models like Gemma
                using_mps = "mps" in str(self.current_model.device) if hasattr(self.current_model, 'device') else False
                
                if using_mps:
                    # Optimized for MPS (Gemma and others) - use sampling to prevent numerical issues
                    is_gemma = current_model_id and "gemma" in current_model_id.lower()
                    
                    if is_gemma:
                        # Special parameters for Gemma on MPS to prevent inf/nan
                        generation_params = {
                            "max_new_tokens": 150,
                            "do_sample": True,         # Use sampling for Gemma stability
                            "temperature": 0.8,       # Moderate temperature for stability
                            "top_p": 0.9,            # Nucleus sampling
                            "top_k": 50,             # Top-k sampling for stability
                            "repetition_penalty": 1.1,
                            "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                            "eos_token_id": self.current_tokenizer.eos_token_id,
                            "return_full_text": False,
                            "use_cache": True
                        }
                    else:
                        # For other models on MPS
                        generation_params = {
                            "max_new_tokens": 150,  # Reduced for faster inference
                            "do_sample": False,     # Deterministic for speed on MPS
                            "num_beams": 1,        # No beam search
                            "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                            "return_full_text": False,
                            "use_cache": True      # Enable KV cache
                        }
                else:
                    # Standard parameters for CPU/CUDA
                    generation_params = {
                        "max_new_tokens": 200,  # Increased for better SQL generation
                        "do_sample": True,      # Enable sampling for better diversity
                        "temperature": 0.7,     # Higher temperature for better generation
                        "top_p": 0.9,          # Nucleus sampling
                        "repetition_penalty": 1.1,  # Lower repetition penalty
                        "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                        "eos_token_id": self.current_tokenizer.eos_token_id,
                        "return_full_text": False,
                        "use_cache": True      # Enable KV cache
                    }
            
            # Generate response with model-specific parameters
            with torch.inference_mode():
                try:
                    # Set CUDA_LAUNCH_BLOCKING for better error reporting
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                    
                    # Pre-tokenize with our validation to ensure consistency
                    if hasattr(self.current_pipeline, 'tokenizer'):
                        try:
                            # Force consistent tokenization before pipeline
                            pipeline_inputs = self.current_pipeline.tokenizer(
                                full_prompt,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=4096
                            )
                            
                            # Apply our validation and fixing
                            pipeline_inputs = self._validate_input_tensors(pipeline_inputs)
                            
                            # Ensure tensors are on the right device - fix for CPU/GPU compatibility
                            try:
                                # Check if pipeline is on GPU (device is numeric) or CPU (device is -1 or string)
                                if isinstance(self.current_pipeline.device, int) and self.current_pipeline.device >= 0:
                                    # GPU device
                                    target_device = torch.device(f"cuda:{self.current_pipeline.device}")
                                    pipeline_inputs = self._safe_to_device(pipeline_inputs, target_device)
                                elif hasattr(self.current_pipeline.model, 'device'):
                                    # Use model's device directly
                                    target_device = self.current_pipeline.model.device
                                    pipeline_inputs = self._safe_to_device(pipeline_inputs, target_device)
                                # else: leave tensors on CPU (default)
                                    
                            except Exception as device_error:
                                self.console.print(f"[yellow]Device handling warning: {device_error}, using CPU[/yellow]")
                                # Continue with CPU tensors
                            
                            # Use the model directly with our validated inputs instead of pipeline
                            with torch.no_grad():
                                # Clear any cached states that might cause shape mismatches
                                if hasattr(self.current_pipeline.model, 'generation_config'):
                                    # Reset generation config to clear any cached sequences
                                    self.current_pipeline.model.generation_config.past_key_values = None
                                    
                                # Ensure no past_key_values are passed (common cause of shape mismatch)
                                generation_params_clean = generation_params.copy()
                                generation_params_clean['past_key_values'] = None
                                # Keep use_cache from generation_params if specified, otherwise default to True for speed
                                if 'use_cache' not in generation_params_clean:
                                    generation_params_clean['use_cache'] = True
                                
                                # Clear any existing model cache
                                if hasattr(self.current_pipeline.model, '_cache'):
                                    self.current_pipeline.model._cache = None
                                    
                                
                                generated_ids = self.current_pipeline.model.generate(
                                    **pipeline_inputs,
                                    **generation_params_clean
                                )
                                
                                # Decode using the tokenizer
                                generated_text = self.current_pipeline.tokenizer.decode(
                                    generated_ids[0],
                                    skip_special_tokens=True
                                )
                                
                                # Remove the original prompt
                                if full_prompt in generated_text:
                                    generated_text = generated_text.replace(full_prompt, "").strip()
                                
                                # Create response in pipeline format
                                response = [{"generated_text": generated_text}]
                        
                        except Exception as direct_error:
                            # Fallback to original pipeline approach
                            response = self.current_pipeline(
                                full_prompt,
                                **generation_params
                            )
                    else:
                        # No tokenizer available, use pipeline directly
                        response = self.current_pipeline(
                            full_prompt,
                            **generation_params
                        )
                    
                except RuntimeError as e:
                    if "probability tensor contains either `inf`, `nan`" in str(e):
                        # Handle Gemma numerical instability
                        self.console.print(f"[yellow]Gemma numerical instability detected, trying fallback generation...[/yellow]")
                        
                        try:
                            # Fallback with more conservative parameters
                            fallback_params = {
                                "max_new_tokens": 100,
                                "do_sample": True,
                                "temperature": 1.0,  # Higher temperature for stability
                                "top_p": 0.95,
                                "repetition_penalty": 1.05,  # Lower penalty
                                "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                                "eos_token_id": self.current_tokenizer.eos_token_id,
                                "use_cache": False,  # Disable cache for stability
                                "return_full_text": False
                            }
                            
                            response = self.current_pipeline(
                                full_prompt,
                                **fallback_params
                            )
                        except Exception as fallback_error:
                            raise RuntimeError(f"Gemma generation failed even with fallback: {fallback_error}")
                            
                    elif "device-side assert" in str(e) or "CUDA error" in str(e) or "index out of bounds" in str(e):
                        # Comprehensive GPU fallback handling
                        self.console.print(f"[yellow]GPU generation failed with device-side assert, trying CPU fallback...[/yellow]")
                        
                        try:
                            # Temporarily move pipeline to CPU
                            original_device = self.current_pipeline.model.device
                            
                            # Move model to CPU
                            self.current_pipeline.model = self.current_pipeline.model.cpu()
                            self.current_pipeline.device = -1  # CPU device for pipeline
                            
                            # Clear CUDA cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            # Try generation on CPU
                            response = self.current_pipeline(
                                full_prompt,
                                **generation_params
                            )
                            
                            # Move back to GPU after successful generation
                            self.current_pipeline.model = self.current_pipeline.model.to(original_device)
                            self.current_pipeline.device = 0 if original_device.type == 'cuda' else -1
                            
                        except Exception as cpu_error:
                            self.console.print(f"[red]CPU fallback also failed: {cpu_error}[/red]")
                            raise RuntimeError(f"Both GPU and CPU generation failed. GPU error: {e}, CPU error: {cpu_error}")
                    else:
                        raise e
            
            generated_text = response[0]["generated_text"].strip()
            
            # CRITICAL: Reset model state after generation to prevent cache conflicts
            try:
                self.current_pipeline.model.eval()  # Reset model to evaluation mode
                
                # Clear any residual cache/state
                if hasattr(self.current_pipeline.model, '_cache'):
                    self.current_pipeline.model._cache = None
                if hasattr(self.current_pipeline.model, 'past_key_values'):
                    self.current_pipeline.model.past_key_values = None
                    
                # Force garbage collection to clear any lingering tensors
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as reset_error:
                self.console.print(f"[yellow]Warning: Could not reset model state: {reset_error}[/yellow]")
            
            # Extract SQL from the response
            sql_query = self._clean_generated_sql(generated_text)
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Error generating SQL: {e}")
            # Enhanced error handling for common GPU issues
            if "device-side assert" in str(e) or "CUDA error" in str(e) or "index out of bounds" in str(e):
                error_msg = (f"GPU generation failed with device-side assert error. "
                           f"This is often caused by index out of bounds or memory issues. "
                           f"The model has been moved to CPU as a fallback. Try using a different model or restart T2S. "
                           f"Original error: {e}")
                raise RuntimeError(error_msg)
            elif "expanded size" in str(e) and "must match" in str(e):
                error_msg = (f"Tensor shape mismatch error. This happens when input_ids and attention_mask have different lengths. "
                           f"Try using a different prompt or restart T2S. If this persists, the model may have tokenizer compatibility issues. "
                           f"Original error: {e}")
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError(f"Error generating SQL: {e}")
    
    async def _generate_sql_gemma3_multimodal(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL using Gemma 3 model with text-only input (supports both processor and tokenizer)."""
        try:
            # Replace the user_question placeholder in the prompt
            full_prompt = system_prompt.replace("{user_question}", user_prompt)
            
            if self.current_processor:
                # Use processor approach (for true multimodal Gemma 3 models)
                try:
                    inputs = self.current_processor(
                        text=full_prompt,
                        images=None,  # Text-only input
                        return_tensors="pt"
                    )
                    
                    # Validate and fix input tensors to prevent device-side asserts
                    inputs = self._validate_input_tensors(inputs)
                    
                    # Move inputs to the model's device safely
                    inputs = self._safe_to_device(inputs, self.current_model.device)
                    # Filter out None values
                    inputs = {k: v for k, v in inputs.items() if v is not None}
                    
                except Exception as input_error:
                    self.console.print(f"[red]Input processing failed: {input_error}[/red]")
                    raise RuntimeError(f"Failed to process input for Gemma 3: {input_error}")
                
                # Generate with appropriate parameters for Gemma 3
                with torch.inference_mode():
                    try:
                        # Set CUDA_LAUNCH_BLOCKING for better error reporting
                        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                        
                        generated_ids = self.current_model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            repetition_penalty=1.1,
                            pad_token_id=self.current_processor.tokenizer.pad_token_id,
                            eos_token_id=self.current_processor.tokenizer.eos_token_id
                        )
                        
                    except RuntimeError as e:
                        if "device-side assert" in str(e) or "CUDA error" in str(e) or "index out of bounds" in str(e):
                            # Comprehensive fallback strategy
                            self.console.print(f"[yellow]GPU generation failed with device-side assert, trying CPU fallback...[/yellow]")
                            
                            try:
                                # Move everything to CPU temporarily
                                cpu_inputs = {}
                                for k, v in inputs.items():
                                    if isinstance(v, torch.Tensor):
                                        cpu_inputs[k] = v.cpu().contiguous()
                                    else:
                                        cpu_inputs[k] = v
                                
                                model_device = self.current_model.device
                                self.current_model = self.current_model.cpu()
                                
                                # Clear CUDA cache
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                
                                # Re-validate inputs on CPU
                                cpu_inputs = self._validate_input_tensors(cpu_inputs)
                                
                                generated_ids = self.current_model.generate(
                                    **cpu_inputs,
                                    max_new_tokens=200,
                                    temperature=0.7,
                                    top_p=0.9,
                                    do_sample=True,
                                    repetition_penalty=1.1,
                                    pad_token_id=self.current_processor.tokenizer.pad_token_id,
                                    eos_token_id=self.current_processor.tokenizer.eos_token_id
                                )
                                
                                # Move model back to original device
                                self.current_model = self.current_model.to(model_device)
                                
                            except Exception as cpu_error:
                                self.console.print(f"[red]CPU fallback also failed: {cpu_error}[/red]")
                                raise RuntimeError(f"Both GPU and CPU generation failed. GPU error: {e}, CPU error: {cpu_error}")
                        else:
                            raise e
                
                # Decode the generated text
                try:
                    generated_texts = self.current_processor.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )
                except Exception as decode_error:
                    # Fallback decoding
                    self.console.print(f"[yellow]Primary decode failed, trying fallback: {decode_error}[/yellow]")
                    generated_texts = self.current_processor.tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )
                
            elif self.current_tokenizer:
                # Use tokenizer approach (for text-only Gemma 3 models)
                try:
                    inputs = self.current_tokenizer(
                        full_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096  # Limit context to prevent issues
                    )
                    
                    # Validate and fix input tensors
                    inputs = self._validate_input_tensors(inputs)
                    
                    # Move inputs to the model's device safely
                    inputs = self._safe_to_device(inputs, self.current_model.device)
                    
                except Exception as input_error:
                    self.console.print(f"[red]Tokenizer input processing failed: {input_error}[/red]")
                    raise RuntimeError(f"Failed to tokenize input for Gemma 3: {input_error}")
                
                # Generate with appropriate parameters for Gemma 3
                with torch.inference_mode():
                    try:
                        # Set CUDA_LAUNCH_BLOCKING for better error reporting
                        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                        
                        generated_ids = self.current_model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            repetition_penalty=1.1,
                            pad_token_id=self.current_tokenizer.pad_token_id,
                            eos_token_id=self.current_tokenizer.eos_token_id
                        )
                        
                    except RuntimeError as e:
                        if "device-side assert" in str(e) or "CUDA error" in str(e) or "index out of bounds" in str(e):
                            # Comprehensive fallback strategy
                            self.console.print(f"[yellow]GPU generation failed with device-side assert, trying CPU fallback...[/yellow]")
                            
                            try:
                                # Move everything to CPU temporarily  
                                cpu_inputs = {}
                                for k, v in inputs.items():
                                    if isinstance(v, torch.Tensor):
                                        cpu_inputs[k] = v.cpu().contiguous()
                                    else:
                                        cpu_inputs[k] = v
                                
                                model_device = self.current_model.device
                                self.current_model = self.current_model.cpu()
                                
                                # Clear CUDA cache
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                
                                # Re-validate inputs on CPU
                                cpu_inputs = self._validate_input_tensors(cpu_inputs)
                                
                                generated_ids = self.current_model.generate(
                                    **cpu_inputs,
                                    max_new_tokens=200,
                                    temperature=0.7,
                                    top_p=0.9,
                                    do_sample=True,
                                    repetition_penalty=1.1,
                                    pad_token_id=self.current_tokenizer.pad_token_id,
                                    eos_token_id=self.current_tokenizer.eos_token_id
                                )
                                
                                # Move model back to original device
                                self.current_model = self.current_model.to(model_device)
                                
                            except Exception as cpu_error:
                                self.console.print(f"[red]CPU fallback also failed: {cpu_error}[/red]")
                                raise RuntimeError(f"Both GPU and CPU generation failed. GPU error: {e}, CPU error: {cpu_error}")
                        else:
                            raise e
                
                # Decode the generated text
                try:
                    generated_texts = self.current_tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )
                except Exception as decode_error:
                    self.console.print(f"[red]Decoding failed: {decode_error}[/red]")
                    raise RuntimeError(f"Failed to decode generated text: {decode_error}")
                
            else:
                raise RuntimeError("Neither processor nor tokenizer is available for Gemma 3 model")
            
            # Remove the original prompt from the generated text
            if full_prompt in generated_texts:
                generated_text = generated_texts.replace(full_prompt, "").strip()
            else:
                generated_text = generated_texts.strip()
            
            # CRITICAL: Reset model state after generation to prevent cache conflicts
            try:
                self.current_model.eval()  # Reset model to evaluation mode
                
                # Clear any residual cache/state
                if hasattr(self.current_model, '_cache'):
                    self.current_model._cache = None
                if hasattr(self.current_model, 'past_key_values'):
                    self.current_model.past_key_values = None
                    
                # Force garbage collection to clear any lingering tensors
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as reset_error:
                self.console.print(f"[yellow]Warning: Could not reset Gemma 3 model state: {reset_error}[/yellow]")
            
            # Extract SQL from the response
            sql_query = self._clean_generated_sql(generated_text)
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Error generating SQL with Gemma 3: {e}")
            # Enhanced error handling for common GPU issues
            if "device-side assert" in str(e) or "CUDA error" in str(e) or "index out of bounds" in str(e):
                error_msg = (f"GPU generation failed with device-side assert error. "
                           f"This is often caused by index out of bounds or memory issues. "
                           f"The model has been moved to CPU as a fallback. Try using a different model or restart T2S. "
                           f"Original error: {e}")
                raise RuntimeError(error_msg)
            elif "expanded size" in str(e) and "must match" in str(e):
                error_msg = (f"Tensor shape mismatch error. This happens when input_ids and attention_mask have different lengths. "
                           f"Try using a different prompt or restart T2S. If this persists, the model may have tokenizer compatibility issues. "
                           f"Original error: {e}")
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError(f"Error generating SQL with Gemma 3: {e}")
    
    def _clean_generated_sql(self, generated_text: str) -> str:
        """Clean up generated SQL query."""
        if not generated_text:
            return ""
        
        # Remove common AI response prefixes
        text = generated_text.strip()
        
        # Remove markdown code blocks
        if "```sql" in text:
            # Extract SQL from code block
            sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if sql_match:
                text = sql_match.group(1).strip()
        elif "```" in text:
            # Generic code block
            code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if code_match:
                text = code_match.group(1).strip()
        
        # Remove common response patterns
        patterns_to_remove = [
            r'^(SQL|Query|Answer|Response):\s*',
            r'^Here\'s?\s+the\s+SQL\s+(query|statement):\s*',
            r'^The\s+SQL\s+(query|statement)\s+is:\s*',
            r'^Based\s+on.*?:\s*',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up the query
        text = text.strip()
        
        # If it doesn't look like SQL, try to extract the last SQL-like statement
        if not any(keyword in text.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'SHOW']):
            # Look for SQL patterns in the text
            sql_patterns = [
                r'(SELECT[^;]*;?)',
                r'(SHOW[^;]*;?)',
                r'(INSERT[^;]*;?)',
                r'(UPDATE[^;]*;?)',
                r'(DELETE[^;]*;?)',
                r'(CREATE[^;]*;?)',
            ]
            
            for pattern in sql_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    text = match.group(1).strip()
                    break
        
        # Add semicolon if missing and text looks like SQL
        if text and any(keyword in text.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'SHOW']):
            if not text.endswith(';'):
                text += ';'
        
        return text
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.current_model:
            del self.current_model
            self.current_model = None
        
        if self.current_tokenizer:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        if self.current_pipeline:
            del self.current_pipeline
            self.current_pipeline = None
        
        if self.current_processor:
            del self.current_processor
            self.current_processor = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.console.print("[green]Model unloaded successfully[/green]")
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        if not self.config.is_model_downloaded(model_id):
            self.console.print(f"[yellow]Model {model_id} is not downloaded[/yellow]")
            return True
        
        model_path = self.config.get_model_path(model_id)
        
        try:
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)
            
            model_name = self.config.SUPPORTED_MODELS[model_id].name
            self.console.print(f"[green]Successfully deleted {model_name}[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model {model_id}: {e}")
            self.console.print(f"[red]Error deleting model: {e}[/red]")
            return False
    
    async def cleanup_corrupted_model(self, model_id: str) -> bool:
        """Clean up a corrupted model download."""
        model_path = self.config.get_model_path(model_id)
        
        if not model_path.exists():
            return True
        
        try:
            # Check if model directory has essential files
            config_file = model_path / "config.json"
            if not config_file.exists():
                self.console.print(f"[yellow]Model {model_id} appears corrupted (missing config.json), cleaning up...[/yellow]")
                import shutil
                shutil.rmtree(model_path, ignore_errors=True)
                return True
            
            # Check if tokenizer files exist
            tokenizer_files = list(model_path.glob("tokenizer*"))
            if not tokenizer_files:
                self.console.print(f"[yellow]Model {model_id} appears corrupted (missing tokenizer files), cleaning up...[/yellow]")
                import shutil
                shutil.rmtree(model_path, ignore_errors=True)
                return True
            
            return False  # Model appears healthy
            
        except Exception as e:
            self.logger.error(f"Error checking model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_id not in self.config.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_id} is not supported")
        
        model_config = self.config.SUPPORTED_MODELS[model_id]
        model_path = self.config.get_model_path(model_id)
        
        info = {
            "id": model_id,
            "name": model_config.name,
            "description": model_config.description,
            "parameters": model_config.parameters,
            "size": model_config.size.value,
            "downloaded": self.config.is_model_downloaded(model_id),
            "path": str(model_path),
            "compatibility": self.config.check_model_compatibility(model_id)
        }
        
        if info["downloaded"]:
            try:
                # Get actual size on disk
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                info["disk_size_gb"] = total_size / (1024**3)
            except Exception:
                info["disk_size_gb"] = model_config.download_size_gb
        
        return info
    
    async def setup_huggingface_auth(self, token: Optional[str] = None) -> bool:
        """Setup HuggingFace authentication."""
        if token:
            try:
                login(token=token)
                user_info = whoami()
                self.config.set_huggingface_token(token)
                self.console.print(f"[green]Successfully authenticated as {user_info['name']}[/green]")
                return True
            except Exception as e:
                self.console.print(f"[red]Authentication failed: {e}[/red]")
                return False
        else:
            # Interactive authentication
            self.console.print("[blue]Opening HuggingFace authentication...[/blue]")
            self.console.print("Please visit: https://huggingface.co/settings/tokens")
            self.console.print("Create a new token and paste it here.")
            return False
    
    def logout_huggingface(self) -> None:
        """Logout from HuggingFace."""
        try:
            logout()
            self.config.set_huggingface_token("")
            self.console.print("[green]Successfully logged out from HuggingFace[/green]")
        except Exception as e:
            self.console.print(f"[red]Error logging out: {e}[/red]") 