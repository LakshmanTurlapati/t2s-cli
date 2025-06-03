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
    AutoProcessor,  # For multimodal models like SmolVLM
    AutoModelForVision2Seq,  # For vision-language models like SmolVLM
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
        
        # Setup device
        self.device = self._get_optimal_device()
        self.console.print(f"[blue]Using device: {self.device}[/blue]")
    
    def _get_optimal_device(self) -> str:
        """Determine the optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS
            return "mps"
        else:
            return "cpu"
    
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
                # Check if this is a Gemma model that needs special handling
                is_gemma_model = "gemma" in model_config.hf_model_id.lower()
                
                # Check if this is a SmolVLM model that needs multimodal handling
                is_smolvlm_model = "smolvlm" in model_config.hf_model_id.lower()
                
                # Configure model loading based on device and model size
                model_kwargs = self._get_model_loading_config(model_config)
                
                if is_gemma_model:
                    # Use the same approach that worked in our test script
                    # Skip separate tokenizer download - let pipeline handle everything
                    self.console.print(f"[blue]Using pipeline-first approach for Gemma model (bypassing SentencePiece issues)...[/blue]")
                    
                    task1 = progress.add_task(f"Downloading {model_config.name} via pipeline...", total=100)
                    
                    # Create pipeline directly - this worked in our test!
                    test_pipeline = pipeline(
                        "text-generation",
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
                    
                    self.console.print(f"[green]✓ Successfully downloaded Gemma model using pipeline approach[/green]")
                    
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
                
                elif is_smolvlm_model:
                    # Handle SmolVLM multimodal model - requires AutoProcessor and AutoModelForVision2Seq
                    self.console.print(f"[blue]Downloading SmolVLM multimodal model using specialized components...[/blue]")
                    
                    task1 = progress.add_task(f"Downloading {model_config.name} processor...", total=100)
                    
                    # Download processor (replaces tokenizer for multimodal models)
                    processor = AutoProcessor.from_pretrained(
                        model_config.hf_model_id,
                        cache_dir=str(model_path),
                        local_files_only=False,
                        trust_remote_code=False,
                        resume_download=True
                    )
                    progress.update(task1, completed=100)
                    
                    # Download multimodal model
                    task2 = progress.add_task(f"Downloading {model_config.name} model...", total=100)
                    
                    # Simulate download progress for UI
                    for i in range(0, 101, 25):
                        progress.update(task2, completed=i)
                        await asyncio.sleep(0.2)
                    
                    model = AutoModelForVision2Seq.from_pretrained(
                        model_config.hf_model_id,
                        torch_dtype=model_kwargs.get("torch_dtype", torch.bfloat16),
                        device_map=model_kwargs.get("device_map", "auto"),
                        attn_implementation=model_kwargs.get("attn_implementation", "eager"),
                        cache_dir=str(model_path),
                        local_files_only=False,
                        trust_remote_code=False,
                        resume_download=True
                    )
                    
                    progress.update(task2, completed=100)
                    
                    # Save locally
                    processor.save_pretrained(str(model_path))
                    model.save_pretrained(str(model_path))
                    
                    self.console.print(f"[green]✓ Successfully downloaded SmolVLM multimodal model[/green]")
                    
                    # Clean up to free memory
                    del processor
                    del model
                    
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
        """Get model loading configuration based on device and model size."""
        config = {}
        
        # Check if this is a Gemma model that needs special handling
        is_gemma_model = "gemma" in model_config.hf_model_id.lower()
        
        # Check if this is a SmolVLM model that needs multimodal handling
        is_smolvlm_model = "smolvlm" in model_config.hf_model_id.lower()
        
        # Use quantization for large models or limited memory
        if model_config.size.value == "large" and self.device == "cuda":
            # Use 4-bit quantization for CUDA only
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            # Use appropriate precision for device
            if self.device == "mps":
                # For Apple Silicon, use bfloat16 for Gemma models, float16 for others
                config["torch_dtype"] = torch.bfloat16 if is_gemma_model else torch.float16
            elif self.device == "cpu":
                config["torch_dtype"] = torch.float32
            else:  # CUDA
                # For CUDA, use bfloat16 for Gemma models, float16 for others
                config["torch_dtype"] = torch.bfloat16 if is_gemma_model else torch.float16
        
        # Set device map for auto distribution - let accelerate handle device placement
        if self.device != "cpu":
            config["device_map"] = "auto"
        
        # Add attention implementation for Gemma models to avoid conflicts
        if is_gemma_model:
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
                    
                    # Check if this is a Gemma model that benefits from pipeline-first approach
                    is_gemma_model = "gemma" in model_config.hf_model_id.lower()
                    
                    # Check if this is a SmolVLM model that needs multimodal handling
                    is_smolvlm_model = "smolvlm" in model_config.hf_model_id.lower()
                    
                    if is_gemma_model:
                        # Use pipeline-first approach for Gemma models (fixes device mapping issues)
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
                    
                    elif is_smolvlm_model:
                        # Use multimodal approach for SmolVLM models
                        try:
                            self.console.print(f"[blue]Loading {model_config.name} as multimodal model...[/blue]")
                            
                            # Load processor (replaces tokenizer for multimodal models)
                            self.current_processor = AutoProcessor.from_pretrained(str(model_path))
                            
                            # Load multimodal model
                            self.current_model = AutoModelForVision2Seq.from_pretrained(
                                str(model_path),
                                torch_dtype=model_kwargs.get("torch_dtype", torch.bfloat16),
                                device_map=model_kwargs.get("device_map", "auto"),
                                attn_implementation=model_kwargs.get("attn_implementation", "eager")
                            )
                            
                            # For consistency with text models, create a text-only interface
                            # Since SmolVLM can work with text-only inputs too
                            self.current_tokenizer = None  # SmolVLM uses processor, not tokenizer
                            self.current_pipeline = None   # SmolVLM doesn't use pipeline
                            
                            self.console.print(f"[green]✓ Successfully loaded {model_config.name} as multimodal model[/green]")
                            self.console.print(f"[dim]Note: SmolVLM is a multimodal model optimized for image+text, but will work with text-only SQL generation[/dim]")
                            
                        except Exception as e:
                            self.logger.error(f"Multimodal loading failed for SmolVLM: {e}")
                            raise RuntimeError(f"Failed to load SmolVLM model: {e}")
                    
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
    
    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL query using the loaded model."""
        
        # Check if we have a SmolVLM model (uses processor instead of pipeline)
        if self.current_processor and self.current_model:
            return await self._generate_sql_smolvlm(system_prompt, user_prompt)
        
        if not self.current_pipeline:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Replace the user_question placeholder in the prompt
        full_prompt = system_prompt.replace("{user_question}", user_prompt)
        
        try:
            # Get the current model ID to determine appropriate generation parameters
            current_model_id = self.config.config.selected_model
            
            # Set generation parameters based on model type
            if current_model_id and "sqlcoder" in current_model_id.lower():
                # SQLCoder-specific parameters (proven to work)
                generation_params = {
                    "max_new_tokens": 300,
                    "do_sample": False,
                    "num_beams": 5,
                    "repetition_penalty": 1.1,
                    "temperature": None,  # Not used with do_sample=False
                    "pad_token_id": self.current_tokenizer.pad_token_id,
                    "eos_token_id": self.current_tokenizer.eos_token_id,
                    "return_full_text": False
                }
            else:
                # General model parameters - optimized for models like Gemma
                generation_params = {
                    "max_new_tokens": 200,  # Increased for better SQL generation
                    "do_sample": True,      # Enable sampling for better diversity
                    "temperature": 0.7,     # Higher temperature for better generation
                    "top_p": 0.9,          # Nucleus sampling
                    "repetition_penalty": 1.1,  # Lower repetition penalty
                    "pad_token_id": self.current_tokenizer.pad_token_id or self.current_tokenizer.eos_token_id,
                    "eos_token_id": self.current_tokenizer.eos_token_id,
                    "return_full_text": False
                }
            
            # Generate response with model-specific parameters
            with torch.inference_mode():
                response = self.current_pipeline(
                    full_prompt,
                    **generation_params
                )
            
            generated_text = response[0]["generated_text"].strip()
            
            # Extract SQL from the response
            sql_query = self._clean_generated_sql(generated_text)
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Error generating SQL: {e}")
            raise RuntimeError(f"Error generating SQL: {e}")
    
    async def _generate_sql_smolvlm(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL using SmolVLM multimodal model with text-only input."""
        try:
            # Replace the user_question placeholder in the prompt
            full_prompt = system_prompt.replace("{user_question}", user_prompt)
            
            # Create messages in chat format for SmolVLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.current_processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process text input (no image)
            inputs = self.current_processor(text=prompt, images=None, return_tensors="pt")
            inputs = {k: v.to(self.current_model.device) for k, v in inputs.items()}
            
            # Generate with appropriate parameters for SmolVLM
            with torch.inference_mode():
                generated_ids = self.current_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # Decode the generated text
            generated_texts = self.current_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            generated_text = generated_texts[0].strip()
            
            # Remove the original prompt from the generated text
            if full_prompt in generated_text:
                generated_text = generated_text.replace(full_prompt, "").strip()
            
            # Extract SQL from the response
            sql_query = self._clean_generated_sql(generated_text)
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Error generating SQL with SmolVLM: {e}")
            raise RuntimeError(f"Error generating SQL with SmolVLM: {e}")
    
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