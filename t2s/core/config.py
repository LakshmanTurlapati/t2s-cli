"""Configuration management for T2S."""

import os
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class ModelSize(str, Enum):
    """Available model sizes."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class SupportedModel(BaseModel):
    """Model configuration."""
    name: str
    hf_model_id: str
    size: ModelSize
    description: str
    parameters: str
    recommended_ram_gb: int
    download_size_gb: float


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    name: str
    type: str  # sqlite, postgresql, mysql
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    path: Optional[str] = None  # For SQLite


def get_default_download_dir() -> str:
    """Get the default model download directory."""
    system = platform.system()
    if system == "Darwin":  # macOS
        return str(Path.home() / "Library" / "Application Support" / "t2s" / "models")
    elif system == "Windows":
        return str(Path.home() / "AppData" / "Local" / "t2s" / "models")
    else:  # Linux and others
        return str(Path.home() / ".config" / "t2s" / "models")


class T2SConfig(BaseModel):
    """Main T2S configuration."""
    selected_model: Optional[str] = None
    huggingface_token: Optional[str] = None
    default_database: Optional[str] = None
    databases: Dict[str, DatabaseConfig] = Field(default_factory=dict)
    download_directory: str = Field(default_factory=get_default_download_dir)
    max_schema_tokens: int = 2000
    enable_query_validation: bool = True
    enable_auto_correction: bool = True
    show_analysis: bool = True
    theme: str = "dark"


class Config:
    """Configuration manager for T2S."""
    
    # Hardcoded supported models as requested
    SUPPORTED_MODELS = {
        "gemma-3-4b": SupportedModel(
            name="Gemma 3 (4B)",
            hf_model_id="google/gemma-3-4b-it",  # Actual Gemma 3 4B instruction-tuned model
            size=ModelSize.MEDIUM,
            description="Fast and efficient for simple queries",
            parameters="4B",
            recommended_ram_gb=8,
            download_size_gb=2.5
        ),
        "gemma-3-12b": SupportedModel(
            name="Gemma 3 (12B)",
            hf_model_id="google/gemma-3-12b-it",  # Actual Gemma 3 12B instruction-tuned model
            size=ModelSize.LARGE,
            description="More accurate for complex queries",
            parameters="12B",
            recommended_ram_gb=16,
            download_size_gb=6.0
        ),
        "llama-3-4b": SupportedModel(
            name="Llama 3.2 (3B)",
            hf_model_id="unsloth/Llama-3.2-3B-Instruct",  # Unsloth's optimized Llama 3.2 3B, no auth needed
            size=ModelSize.MEDIUM,
            description="Llama 3.2 3B Instruct model optimized for efficient inference",
            parameters="3B",
            recommended_ram_gb=8,
            download_size_gb=6.5
        ),
        "smollm-1.7b": SupportedModel(
            name="SmolLM (1.7B)",
            hf_model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",  # Latest SmolLM2 instruction-tuned model
            size=ModelSize.SMALL,
            description="Lightweight text model - efficient for basic to intermediate queries",
            parameters="1.7B",
            recommended_ram_gb=6,
            download_size_gb=3.2
        ),
        "defog-sqlcoder-7b": SupportedModel(
            name="Defog SQLCoder (7B)",
            hf_model_id="defog/sqlcoder-7b-2",  # ACTUAL SQL generation model - keep this one!
            size=ModelSize.LARGE,
            description="Specialized SQL generation model - best for complex queries",
            parameters="7B",
            recommended_ram_gb=12,
            download_size_gb=4.8
        )
    }
    
    def __init__(self):
        self.config_dir = self.get_config_dir()
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config = self.load_config()
    
    @staticmethod
    def get_config_dir() -> Path:
        """Get the configuration directory based on the OS."""
        system = platform.system()
        if system == "Darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "t2s"
        elif system == "Windows":
            return Path.home() / "AppData" / "Local" / "t2s"
        else:  # Linux and others
            return Path.home() / ".config" / "t2s"
    
    @staticmethod
    def get_default_download_dir() -> Path:
        """Get the default model download directory."""
        return Config.get_config_dir() / "models"
    
    def load_config(self) -> T2SConfig:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return T2SConfig(**data)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                return T2SConfig()
        return T2SConfig()
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config.model_dump(), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    @property
    def config(self) -> T2SConfig:
        """Get the current configuration."""
        return self._config
    
    def set_selected_model(self, model_id: str) -> None:
        """Set the selected model."""
        if model_id in self.SUPPORTED_MODELS:
            self._config.selected_model = model_id
            self.save_config()
        else:
            raise ValueError(f"Model {model_id} not supported")
    
    def add_database(self, name: str, db_config: DatabaseConfig) -> None:
        """Add a database configuration."""
        self._config.databases[name] = db_config
        self.save_config()
    
    def remove_database(self, name: str) -> None:
        """Remove a database configuration."""
        if name in self._config.databases:
            del self._config.databases[name]
            if self._config.default_database == name:
                self._config.default_database = None
            self.save_config()
    
    def set_default_database(self, name: str) -> None:
        """Set the default database."""
        if name in self._config.databases:
            self._config.default_database = name
            self.save_config()
        else:
            raise ValueError(f"Database {name} not configured")
    
    def set_huggingface_token(self, token: str) -> None:
        """Set HuggingFace token."""
        self._config.huggingface_token = token
        self.save_config()
    
    def get_model_path(self, model_id: str) -> Path:
        """Get the local path for a model."""
        download_dir = Path(self._config.download_directory)
        return download_dir / model_id
    
    def get_models_dir(self) -> Path:
        """Get the base models directory."""
        return Path(self._config.download_directory)
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded locally."""
        model_path = self.get_model_path(model_id)
        return model_path.exists() and any(model_path.iterdir())
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for compatibility checks."""
        import psutil
        
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        }
    
    def check_model_compatibility(self, model_id: str) -> Dict[str, Any]:
        """Check if a model is compatible with the current system."""
        if model_id not in self.SUPPORTED_MODELS:
            return {"compatible": False, "reason": "Model not supported"}
        
        model = self.SUPPORTED_MODELS[model_id]
        sys_info = self.get_system_info()
        
        # Check RAM requirements
        available_ram = sys_info["available_memory_gb"]
        required_ram = model.recommended_ram_gb
        
        compatible = available_ram >= required_ram
        
        return {
            "compatible": compatible,
            "required_ram_gb": required_ram,
            "available_ram_gb": available_ram,
            "download_size_gb": model.download_size_gb,
            "reason": None if compatible else f"Insufficient RAM: {required_ram}GB required, {available_ram:.1f}GB available"
        } 