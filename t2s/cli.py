"""Main CLI interface for T2S."""

import asyncio
import sys
from typing import Optional
import logging
import re
import platform
import os
import termios
import tty

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

from .core.engine import T2SEngine
from .core.config import Config
from .models.model_manager import ModelManager
from .database.db_manager import DatabaseManager
from .ui.ascii_art import T2SArt
from . import __version__  # Import version from package


def _getch():
    """Get a single character from stdin without echo (Unix/Mac only)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle arrow keys and other escape sequences
        if ch == '\x1b':  # ESC
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return '\x1b[' + ch3
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def simple_select(title: str, choices: list, default_index: int = 0) -> str:
    """Interactive selection menu with arrow key navigation and number input.

    Navigation:
    - Up/Down arrows: Move selection
    - Type number: Direct selection (press Enter to confirm)
    - Enter: Confirm selection (uses typed number if available, otherwise highlighted item)
    - Ctrl+C/Escape: Use default selection
    """
    console = Console()
    current_selection = default_index
    number_buffer = ""

    def display_menu():
        """Display the menu with current selection highlighted."""
        # Use ANSI codes for colors directly to avoid Rich's line handling
        # Cyan color for title
        output = f"\n\033[1;36m{title}\033[0m\n"

        for i, choice in enumerate(choices):
            marker = ">" if i == current_selection else " "
            # Cyan color for numbers
            output += f" {marker} \033[36m{i+1}.\033[0m {choice}\n"

        # Add input prompt in dim gray
        output += "\n"
        if number_buffer:
            output += f"\033[2mNumber input: {number_buffer}_\033[0m"
        else:
            output += f"\033[2mUse ↑↓ arrows to navigate, or type a number (1-{len(choices)})\033[0m"

        # Print without newline at end
        sys.stdout.write(output)
        sys.stdout.flush()

        # Return the number of lines printed for clearing
        return output.count('\n') + 1  # +1 for the last line without \n

    def clear_menu(num_lines):
        """Clear the previously displayed menu."""
        # First, clear the current line (where cursor is at end of prompt)
        sys.stdout.write('\r')       # Move to start of current line
        sys.stdout.write('\033[2K')  # Clear entire line

        # Now move up and clear each previous line
        for _ in range(num_lines - 1):
            sys.stdout.write('\033[A')  # Move up one line
            sys.stdout.write('\r')       # Move to start of line
            sys.stdout.write('\033[2K')  # Clear entire line

        sys.stdout.flush()

    # Initial display
    lines_displayed = display_menu()

    try:
        while True:
            ch = _getch()

            # Handle arrow keys
            if ch == '\x1b[A':  # Up arrow
                if not number_buffer:  # Only move if no number typed
                    current_selection = (current_selection - 1) % len(choices)
                    clear_menu(lines_displayed)
                    lines_displayed = display_menu()

            elif ch == '\x1b[B':  # Down arrow
                if not number_buffer:  # Only move if no number typed
                    current_selection = (current_selection + 1) % len(choices)
                    clear_menu(lines_displayed)
                    lines_displayed = display_menu()

            # Handle number input
            elif ch.isdigit():
                number_buffer += ch
                clear_menu(lines_displayed)
                lines_displayed = display_menu()

            # Handle backspace
            elif ch in ('\x7f', '\x08'):  # Backspace or Delete
                if number_buffer:
                    number_buffer = number_buffer[:-1]
                    clear_menu(lines_displayed)
                    lines_displayed = display_menu()

            # Handle Enter
            elif ch in ('\r', '\n'):
                clear_menu(lines_displayed)
                sys.stdout.write('\n')
                sys.stdout.flush()

                if number_buffer:
                    # Use typed number
                    try:
                        idx = int(number_buffer) - 1
                        if 0 <= idx < len(choices):
                            return choices[idx]
                        else:
                            console.print(f"[red]Invalid selection. Please enter 1-{len(choices)}[/red]")
                            number_buffer = ""
                            lines_displayed = display_menu()
                    except ValueError:
                        console.print(f"[red]Invalid input[/red]")
                        number_buffer = ""
                        lines_displayed = display_menu()
                else:
                    # Use arrow-selected item
                    return choices[current_selection]

            # Handle Escape
            elif ch == '\x1b':
                clear_menu(lines_displayed)
                console.print(f"[yellow]Cancelled - using default: {choices[default_index]}[/yellow]")
                return choices[default_index]

    except KeyboardInterrupt:
        clear_menu(lines_displayed)
        console.print(f"[yellow]Interrupted - using default: {choices[default_index]}[/yellow]")
        return choices[default_index]


class T2SCLI:
    """Main CLI application for T2S."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.config = Config()
        self.engine = None

        # Detect theme based on default database type
        theme = self._detect_theme()
        self.art = T2SArt(theme=theme)

        # Setup logging
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _detect_theme(self) -> str:
        """Detect theme based on default database type."""
        default_db = self.config.config.default_database
        if default_db and default_db in self.config.config.databases:
            db_config = self.config.config.databases[default_db]
            if db_config.type.lower() == "mongodb":
                return "mongodb"
        return "sql"
    
    async def initialize(self):
        """Initialize the T2S engine."""
        if not self.engine:
            self.engine = T2SEngine(self.config)
            await self.engine.initialize()
    
    async def interactive_mode(self):
        """Start interactive T2S session."""
        self.console.clear()
        
        # Show welcome banner
        self.console.print(self.art.get_welcome_banner(self.console))
        self.console.print()
        
        # Initialize engine
        await self.initialize()
        
        # Check if we have a model and database configured
        if not self.config.config.selected_model:
            self.console.print(self.art.get_status_indicator("warning", "No AI model selected. Let's set one up!"))
            await self.model_selection_wizard()
        
        if not self.config.config.default_database:
            self.console.print(self.art.get_status_indicator("warning", "No default database selected. Let's set one up!"))
            await self.database_selection_wizard()
        
        # Main interactive loop
        while True:
            try:
                self.console.print(self.art.get_separator())
                self.console.print(self.art.get_compact_header())
                
                query = Prompt.ask(
                    f"\n[bold {self.art.colors['primary']}]Enter your natural language query[/bold {self.art.colors['primary']}] [dim](or 'q' to quit)[/dim]",
                    default="",
                    show_default=False
                )
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                elif query.lower() in ['help', 'h']:
                    self.console.print(self.art.get_help_panel())
                    continue
                elif query.lower() in ['config', 'settings']:
                    await self.configuration_menu()
                    continue
                
                # Process the query
                self.console.print(self.art.get_query_result_header(query))
                result = await self.engine.process_query(query)
                
                # Display results
                self.engine.display_results(result)
                
                # Automatically continue to next question (removed confirmation prompt)
                # User can type 'q' to quit when they want to exit
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted by user[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        self.console.print("[blue]Thanks for using T2S! 👋[/blue]")
    
    async def direct_query(self, query: str, database: Optional[str] = None):
        """Process a direct query."""
        await self.initialize()
        
        if not self.engine:
            self.console.print(self.art.get_status_indicator("error", "Failed to initialize T2S engine"))
            return
        
        # Show compact header
        self.console.print(self.art.get_compact_header())
        self.console.print(self.art.get_query_result_header(query))
        
        # Process query
        result = await self.engine.process_query(query, database)
        
        # Display results
        self.engine.display_results(result)
    
    async def configuration_menu(self):
        """Show configuration menu."""
        while True:
            self.console.clear()
            self.console.print(Panel("T2S Configuration", style=f"bold {self.art.colors['primary']}"))

            choices = [
                "Manage AI Models",
                "Manage Databases",
                "External API Keys",
                "HuggingFace Authentication",
                "General Settings",
                "System Information",
                "Back to Main Menu"
            ]

            choice = simple_select("What would you like to configure?", choices)

            if choice == choices[0]:  # Models
                await self.model_management_menu()
            elif choice == choices[1]:  # Databases
                await self.database_management_menu()
            elif choice == choices[2]:  # External API Keys
                await self.external_api_keys_menu()
            elif choice == choices[3]:  # HuggingFace
                await self.huggingface_auth_menu()
            elif choice == choices[4]:  # Settings
                await self.general_settings_menu()
            elif choice == choices[5]:  # System Info
                await self.show_system_info()
            else:  # Back
                break
    
    async def model_management_menu(self):
        """Model management interface."""
        model_manager = ModelManager(self.config)
        
        while True:
            self.console.clear()
            self.console.print(Panel("AI Model Management", style=f"bold {self.art.colors['primary']}"))
            
            # Show current active model prominently
            current_model = self.config.config.selected_model
            if current_model:
                if current_model in self.config.SUPPORTED_MODELS:
                    model_name = self.config.SUPPORTED_MODELS[current_model].name
                    model_type = "Local"
                elif current_model in self.config.EXTERNAL_API_MODELS:
                    model_name = self.config.EXTERNAL_API_MODELS[current_model]["name"]
                    model_type = "API"
                else:
                    model_name = current_model
                    model_type = "Unknown"

                self.console.print(Panel(
                    f"🎯 Current Active Model: {model_name} ({current_model}) [{model_type}]",
                    style="bold green",
                    title="Active Model"
                ))
            else:
                self.console.print(Panel(
                    "⚠️  No model currently selected",
                    style="bold yellow",
                    title="Active Model"
                ))

            self.console.print()

            # Show local models
            self.console.print(f"[bold {self.art.colors['primary']}]Local Models:[/bold {self.art.colors['primary']}]")
            models_info = await self.engine.get_available_models() if self.engine else {}

            if not models_info:
                # Fallback: show from config
                models_info = {}
                for model_id, model_config in self.config.SUPPORTED_MODELS.items():
                    models_info[model_id] = {
                        "name": model_config.name,
                        "description": model_config.description,
                        "parameters": model_config.parameters,
                        "downloaded": self.config.is_model_downloaded(model_id),
                        "compatibility": self.config.check_model_compatibility(model_id)
                    }
            
            # Display model cards with active indicator and memory warnings
            for model_id, info in models_info.items():
                status = "downloaded" if info.get("downloaded", False) else "available"
                
                # Check memory compatibility and add warning
                compatibility = info.get("compatibility", self.config.check_model_compatibility(model_id))
                
                # Add memory warning if incompatible
                if not compatibility.get("compatible", True):
                    # Create warning message about memory
                    memory_warning = f"⚠️  MEMORY WARNING: {compatibility.get('reason', 'Insufficient memory')}"
                    
                    # Add warning to console before the model card
                    self.console.print(f"[red]{memory_warning}[/red]")
                
                # Add active indicator to the card
                if model_id == current_model:
                    # Modify the card title to show it's active
                    info_copy = info.copy()
                    info_copy["name"] = f"{info['name']} (ACTIVE)"
                    card = self.art.get_model_card(info_copy["name"], status, info_copy)
                else:
                    card = self.art.get_model_card(info["name"], status, info)
                self.console.print(card)

            # Show API models
            self.console.print()
            self.console.print("[bold magenta]External API Models:[/bold magenta]")
            for model_id, model_info in self.config.EXTERNAL_API_MODELS.items():
                # Check if API key is configured
                provider = model_info["provider"]
                has_api_key = bool(self.config.get_api_key(provider))

                # Create simple display for API models
                is_active = (model_id == current_model)
                active_marker = " (ACTIVE)" if is_active else ""

                # Add availability status
                if has_api_key:
                    availability = "[green]✓ Ready to use[/green]"
                    panel_style = "magenta" if is_active else "dim"
                else:
                    availability = "[red]✗ API key not configured[/red]"
                    panel_style = "dim red"

                self.console.print(Panel(
                    f"[bold]{model_info['name']}{active_marker}[/bold]\n"
                    f"[dim]{model_info['description']}[/dim]\n"
                    f"[yellow]Pricing: {model_info['pricing']}[/yellow]\n"
                    f"[cyan]Provider: {model_info['provider']}[/cyan]\n"
                    f"{availability}",
                    title=f"API Model: {model_id}",
                    style=panel_style
                ))

            self.console.print()

            choices = [
                "Download Local Model",
                "Select Active Model (Local or API)",
                "Delete Local Model",
                "Back"
            ]
            
            choice = simple_select("What would you like to do?", choices)
            
            if choice == choices[0]:  # Download
                await self.download_model_wizard(model_manager)
            elif choice == choices[1]:  # Select
                await self.select_model_wizard()
            elif choice == choices[2]:  # Delete
                await self.delete_model_wizard(model_manager)
            else:  # Back
                break
    
    async def download_model_wizard(self, model_manager: ModelManager):
        """Guide user through model download."""
        # Show ALL models with their download status
        all_models = []
        for model_id, config in self.config.SUPPORTED_MODELS.items():
            is_downloaded = self.config.is_model_downloaded(model_id)
            status = " [DOWNLOADED]" if is_downloaded else " [NOT DOWNLOADED]"
            all_models.append((model_id, config.name, is_downloaded, f"{config.name} ({model_id}){status}"))
        
        if not all_models:
            self.console.print(self.art.get_status_indicator("info", "No models available!"))
            input("Press Enter to continue...")
            return
        
        # Select model to download
        model_choices = [display_name for _, _, _, display_name in all_models]
        model_choices.append("Cancel")
        
        model_choice = simple_select("Which model would you like to download?", model_choices)
        
        if model_choice == "Cancel":
            return
        
        # Extract model ID from the choice
        match = re.search(r'\([^)]+\)\s*\(([^)]+)\)', model_choice)
        if match:
            model_id = match.group(1)
        else:
            # Fallback: try to extract any model ID pattern
            for supported_id in self.config.SUPPORTED_MODELS.keys():
                if supported_id in model_choice:
                    model_id = supported_id
                    break
            else:
                self.console.print(self.art.get_status_indicator("error", "Could not extract model ID from selection"))
                return
        
        if not model_id:
            return
        
        # Check if already downloaded
        if self.config.is_model_downloaded(model_id):
            self.console.print(self.art.get_status_indicator("info", f"Model {self.config.SUPPORTED_MODELS[model_id].name} is already downloaded!"))
            if Confirm.ask("Would you like to re-download it?"):
                # Continue with download
                pass
            else:
                input("Press Enter to continue...")
                return
        
        # Confirm download
        model_config = self.config.SUPPORTED_MODELS[model_id]
        
        # Check memory compatibility and show warning
        compatibility = self.config.check_model_compatibility(model_id)
        
        self.console.print(f"\n[bold]Model:[/bold] {model_config.name}")
        self.console.print(f"[bold]Size:[/bold] {model_config.download_size_gb:.1f} GB")
        self.console.print(f"[bold]RAM Required:[/bold] {model_config.recommended_ram_gb} GB")
        
        # Show memory compatibility status
        if compatibility.get("compatible"):
            self.console.print(f"[bold]Memory Status:[/bold] [green]✓ Compatible[/green]")
            self.console.print(f"[dim]Available RAM: {compatibility['available_ram_gb']:.1f}GB[/dim]")
        else:
            self.console.print(f"[bold]Memory Status:[/bold] [red]⚠️  {compatibility.get('reason', 'Insufficient memory')}[/red]")
            self.console.print(f"[red]Required: {compatibility['required_ram_gb']}GB | Available: {compatibility['available_ram_gb']:.1f}GB[/red]")
            self.console.print(f"[yellow]Warning: This model may not run properly on your system![/yellow]")
        
        if not Confirm.ask("Proceed with download?"):
            return
        
        # Attempt download
        await self._download_with_auth_retry(model_manager, model_id, model_config)
    
    async def _download_with_auth_retry(self, model_manager: ModelManager, model_id: str, model_config):
        """Download model with proper error handling and retry logic."""
        max_retries = 3  # Increased for network issues
        
        for attempt in range(max_retries):
            self.console.print(f"\n[blue]Downloading {model_config.name}... (Attempt {attempt + 1})[/blue]")
            
            # Store the last error for detailed analysis
            last_error = None
            
            # Capture error details from model manager
            try:
                success = await model_manager.download_model(model_id)
                
                if success:
                    self.console.print(self.art.get_status_indicator("success", f"Successfully downloaded {model_config.name}!"))
                    
                    # Ask if user wants to select this model
                    if Confirm.ask("Set this as the active model?"):
                        self.config.set_selected_model(model_id)
                        self.console.print(self.art.get_status_indicator("success", f"Active model set to {model_config.name}"))
                    
                    input("Press Enter to continue...")
                    return
                    
            except Exception as e:
                last_error = str(e)
                self.console.print(f"[red]Download failed: {last_error}[/red]")
            
            # Analyze the actual error type, not just model type
            error_type = self._analyze_error_type(last_error, model_config)
            
            if error_type == "timeout" or error_type == "network":
                self.console.print(self.art.get_status_indicator("warning", f"Network/timeout error. Retrying... ({attempt + 1}/{max_retries})"))
                
                if attempt < max_retries - 1:
                    # For network errors, just retry
                    continue
                else:
                    # Final timeout attempt
                    self.console.print(self.art.get_status_indicator("error", "Multiple network failures. Please check your internet connection and try again later."))
                    break
                    
            elif error_type == "auth":
                self.console.print(self.art.get_status_indicator("warning", "This model requires HuggingFace authentication."))
                
                if attempt < max_retries - 1:  # Not the last attempt
                    auth_choices = [
                        "Login to HuggingFace and retry",
                        "Open HuggingFace signup/login page", 
                        "Skip this model"
                    ]
                    
                    auth_choice = simple_select("What would you like to do?", auth_choices)
                    
                    if auth_choice == auth_choices[0]:  # Login and retry
                        if await self._quick_huggingface_auth(model_manager):
                            continue  # Retry download
                        else:
                            break  # Authentication failed
                    elif auth_choice == auth_choices[1]:  # Open browser
                        import webbrowser
                        webbrowser.open("https://huggingface.co/join")
                        self.console.print(self.art.get_status_indicator("info", "Opened HuggingFace signup page"))
                        self.console.print("After creating an account and getting a token, come back and try again!")
                        input("Press Enter to continue...")
                        break
                    else:  # Skip
                        break
                else:
                    # Last attempt failed
                    self.console.print(self.art.get_status_indicator("error", "Authentication required but not provided."))
                    break
                    
            elif error_type == "disk_space":
                self.console.print(self.art.get_status_indicator("error", "Insufficient disk space. Please free up space and try again."))
                break
                
            elif error_type == "model_not_found":
                self.console.print(self.art.get_status_indicator("error", f"Model {model_config.hf_model_id} not found on HuggingFace."))
                break
                
            elif error_type == "dependency":
                self.console.print(self.art.get_status_indicator("error", f"Missing dependency: {last_error}"))
                self.console.print(self.art.get_status_indicator("info", "This appears to be a code issue, not a download issue. Please check the application setup."))
                break
                
            else:
                # Generic error
                self.console.print(self.art.get_status_indicator("error", f"Failed to download {model_config.name}: {last_error}"))
                if attempt < max_retries - 1:
                    retry_choice = simple_select("What would you like to do?", ["Retry", "Skip this model"])
                    if retry_choice == "Retry":
                        continue
                break
        
        input("Press Enter to continue...")
    
    def _analyze_error_type(self, error_message: str, model_config) -> str:
        """Analyze error message to determine the type of error."""
        if not error_message:
            return "unknown"
        
        error_lower = error_message.lower()
        
        # Check for dependency/import errors first
        if any(pattern in error_lower for pattern in ["no module named", "import error", "modulenotfounderror", "cannot import"]):
            return "dependency"
        
        # Check for specific error patterns
        if any(pattern in error_lower for pattern in ["timeout", "read timed out", "connection timeout"]):
            return "timeout"
        
        if any(pattern in error_lower for pattern in ["connection", "network", "resolve", "unreachable"]):
            return "network"
        
        if any(pattern in error_lower for pattern in ["401", "unauthorized", "authentication", "access denied", "token"]):
            return "auth"
        
        if any(pattern in error_lower for pattern in ["403", "forbidden", "gated", "you need to share your contact information"]):
            return "auth"  # Gated models
            
        if any(pattern in error_lower for pattern in ["404", "not found", "repository not found"]):
            return "model_not_found"
        
        if any(pattern in error_lower for pattern in ["disk", "space", "storage", "no space left"]):
            return "disk_space"
        
        # Fallback: check if model typically requires auth (but only as last resort)
        if self._is_potentially_gated_model(model_config):
            return "auth"
        
        return "unknown"
    
    def _is_potentially_gated_model(self, model_config) -> bool:
        """Check if model might be gated based on model ID (used as fallback only)."""
        # These models are typically gated
        gated_patterns = [
            "meta-llama",  # Llama models
            "google/gemma",  # Some Gemma models
            "microsoft/DialoGPT"  # Some Dialog models
        ]
        
        return any(pattern in model_config.hf_model_id.lower() for pattern in gated_patterns)
    
    def _is_auth_error_likely(self, model_config) -> bool:
        """Check if the download failure was likely due to authentication."""
        # This method is now deprecated in favor of _analyze_error_type
        # Keeping for backward compatibility but should use _analyze_error_type
        return self._is_potentially_gated_model(model_config)
    
    async def _quick_huggingface_auth(self, model_manager: ModelManager) -> bool:
        """Quick HuggingFace authentication flow."""
        self.console.print("\n[bold cyan]HuggingFace Authentication[/bold cyan]")
        self.console.print("To download gated models, you need a HuggingFace token.")
        self.console.print("Get one at: https://huggingface.co/settings/tokens")
        
        # Check if already authenticated
        if self.config.config.huggingface_token:
            self.console.print("Token already configured, retrying download...")
            return True
        
        token = input("\nEnter your HuggingFace token (or press Enter to skip): ").strip()
        
        if not token:
            self.console.print(self.art.get_status_indicator("warning", "No token provided, skipping authentication."))
            return False
        
        # Attempt authentication
        try:
            success = await model_manager.setup_huggingface_auth(token)
            if success:
                self.console.print(self.art.get_status_indicator("success", "Authentication successful!"))
                return True
            else:
                self.console.print(self.art.get_status_indicator("error", "Authentication failed. Please check your token."))
                return False
        except Exception as e:
            self.console.print(self.art.get_status_indicator("error", f"Authentication error: {e}"))
            return False
    
    async def select_model_wizard(self):
        """Guide user through model selection."""
        # Get downloaded local models
        downloaded_models = [
            (model_id, config.name, "local")
            for model_id, config in self.config.SUPPORTED_MODELS.items()
            if self.config.is_model_downloaded(model_id)
        ]

        # Add API models (only if API key is configured for the provider)
        api_models = [
            (model_id, model_info["name"], "api")
            for model_id, model_info in self.config.EXTERNAL_API_MODELS.items()
            if self.config.get_api_key(model_info["provider"])  # Only show if API key is set
        ]

        all_models = downloaded_models + api_models

        if not all_models:
            self.console.print(self.art.get_status_indicator("warning", "No models available. Please download a local model or configure API keys."))
            input("Press Enter to continue...")
            return

        # Show current selection
        current_model = self.config.config.selected_model
        if current_model:
            if current_model in self.config.SUPPORTED_MODELS:
                current_name = self.config.SUPPORTED_MODELS[current_model].name
                model_type = "Local"
            elif current_model in self.config.EXTERNAL_API_MODELS:
                current_name = self.config.EXTERNAL_API_MODELS[current_model]["name"]
                model_type = "API"
            else:
                current_name = current_model
                model_type = "Unknown"

            self.console.print(Panel(
                f"Currently Active: {current_name} ({current_model}) [{model_type}]",
                style="bold blue",
                title="Current Selection"
            ))
        else:
            self.console.print(Panel(
                "No model currently selected",
                style="bold yellow",
                title="Current Selection"
            ))

        self.console.print()

        # Format choices with more detail and active indicator
        model_choices = []
        for model_id, name, model_type in all_models:
            active_indicator = " (CURRENTLY ACTIVE)" if model_id == current_model else ""
            if model_type == "local":
                model_config = self.config.SUPPORTED_MODELS[model_id]
                choice_text = f"[Local] {name} ({model_id}) - {model_config.parameters}{active_indicator}"
            else:
                choice_text = f"[API] {name} ({model_id}){active_indicator}"
            model_choices.append((choice_text, model_id))

        # Create display choices
        display_choices = [choice_text for choice_text, _ in model_choices]
        display_choices.append("Cancel")
        
        selected_choice = simple_select("Which model would you like to set as active?", display_choices)

        if selected_choice == "Cancel":
            return

        # Find the corresponding model_id
        choice_index = display_choices.index(selected_choice)
        if choice_index < len(model_choices):
            _, model_id = model_choices[choice_index]
        else:
            return

        # Don't change if it's already the active model
        if model_id == current_model:
            if model_id in self.config.SUPPORTED_MODELS:
                model_name = self.config.SUPPORTED_MODELS[model_id].name
            else:
                model_name = self.config.EXTERNAL_API_MODELS[model_id]["name"]
            self.console.print(self.art.get_status_indicator("info", f"Model {model_name} is already active!"))
            input("Press Enter to continue...")
            return

        # Set as active model
        self.config.set_selected_model(model_id)
        if model_id in self.config.SUPPORTED_MODELS:
            model_name = self.config.SUPPORTED_MODELS[model_id].name
        else:
            model_name = self.config.EXTERNAL_API_MODELS[model_id]["name"]
        self.console.print(self.art.get_status_indicator("success", f"Active model set to {model_name}"))

        input("Press Enter to continue...")
    
    async def delete_model_wizard(self, model_manager: ModelManager):
        """Guide user through model deletion."""
        downloaded_models = [
            (model_id, config.name)
            for model_id, config in self.config.SUPPORTED_MODELS.items()
            if self.config.is_model_downloaded(model_id)
        ]
        
        if not downloaded_models:
            self.console.print(self.art.get_status_indicator("info", "No downloaded models to delete."))
            input("Press Enter to continue...")
            return
        
        model_choices = [f"{name} ({model_id})" for model_id, name in downloaded_models]
        model_choices.append("Cancel")
        
        model_choice = simple_select("Which model would you like to delete?", model_choices)
        
        if model_choice == "Cancel":
            return
        
        # Extract model ID
        model_id = model_choice.split("(")[-1].rstrip(")")
        model_name = self.config.SUPPORTED_MODELS[model_id].name
        
        # Confirm deletion
        if not Confirm.ask(f"Are you sure you want to delete {model_name}? This cannot be undone."):
            return
        
        # Delete model
        success = await model_manager.delete_model(model_id)
        
        if success:
            self.console.print(self.art.get_status_indicator("success", f"Successfully deleted {model_name}"))
            
            # If this was the active model, clear selection
            if self.config.config.selected_model == model_id:
                self.config.config.selected_model = None
                self.config.save_config()
                self.console.print(self.art.get_status_indicator("info", "Active model cleared. Please select a new one."))
        else:
            self.console.print(self.art.get_status_indicator("error", f"Failed to delete {model_name}"))
        
        input("Press Enter to continue...")
    
    async def database_management_menu(self):
        """Database management interface."""
        db_manager = DatabaseManager(self.config)

        while True:
            # Clear screen more reliably using ANSI codes
            import os
            os.system('clear' if os.name != 'nt' else 'cls')
            self.console.print(Panel("Database Management", style=f"bold {self.art.colors['primary']}"))
            
            # Show configured databases
            databases = db_manager.list_databases()
            
            if databases:
                for db in databases:
                    if db["type"] == "sqlite":
                        connection_info = f"File: {db.get('path', 'N/A')}"
                    else:
                        connection_info = f"{db.get('host', 'N/A')}:{db.get('port', 'N/A')}/{db.get('database', 'N/A')}"
                    
                    card = self.art.get_database_card(
                        db["name"], db["type"], db["status"], connection_info
                    )
                    self.console.print(card)
            else:
                self.console.print(self.art.get_status_indicator("info", "No databases configured yet."))
            
            self.console.print()
            
            choices = [
                "Add Database",
                "Discover MongoDB Databases",
                "Set Default Database",
                "Refresh Connection Status",
                "Remove Database",
                "Back"
            ]

            choice = simple_select("What would you like to do?", choices)

            if choice == choices[0]:  # Add
                await self.add_database_wizard(db_manager)
            elif choice == choices[1]:  # Discover MongoDB
                await self.discover_mongodb_wizard(db_manager)
            elif choice == choices[2]:  # Set Default
                await self.set_default_database_wizard()
            elif choice == choices[3]:  # Refresh
                await db_manager.initialize()
                self.console.print(self.art.get_status_indicator("success", "Connection status refreshed"))
                input("Press Enter to continue...")
            elif choice == choices[4]:  # Remove
                await self.remove_database_wizard(db_manager)
            else:  # Back
                break
    
    async def add_database_wizard(self, db_manager: DatabaseManager):
        """Guide user through adding a database."""
        db_choices = ["SQLite", "PostgreSQL", "MySQL", "MongoDB", "Cancel"]
        db_type = simple_select("What type of database?", db_choices)
        
        if db_type == "Cancel":
            return
        
        db_name = input("Database name (for reference): ").strip()
        if not db_name:
            return
        
        if db_type == "SQLite":
            db_path = input("Path to SQLite database file: ").strip()
            if not db_path:
                return

            success = await db_manager.add_database(
                db_name, "sqlite", path=db_path
            )
        elif db_type == "MongoDB":
            host = input("Host (default: localhost): ").strip() or "localhost"
            port = input("Port (default: 27017): ").strip() or "27017"
            database = input("Database name: ").strip()
            username = input("Username (optional): ").strip() or None
            password = input("Password (optional): ").strip() or None

            success = await db_manager.add_database(
                db_name,
                "mongodb",
                host=host,
                port=int(port),
                database=database,
                username=username,
                password=password
            )
        else:
            host = input("Host (default: localhost): ").strip() or "localhost"
            port_default = "5432" if db_type == "PostgreSQL" else "3306"
            port = input(f"Port (default: {port_default}): ").strip() or port_default
            database = input("Database name: ").strip()
            username = input("Username: ").strip()
            password = input("Password (optional): ").strip()
            
            success = await db_manager.add_database(
                db_name,
                db_type.lower(),
                host=host,
                port=int(port),
                database=database,
                username=username,
                password=password or None
            )
        
        if success and not self.config.config.default_database:
            if Confirm.ask("Set this as the default database?"):
                self.config.set_default_database(db_name)
        
        input("Press Enter to continue...")
    
    async def set_default_database_wizard(self):
        """Guide user through setting default database."""
        databases = list(self.config.config.databases.keys())
        
        if not databases:
            self.console.print(self.art.get_status_indicator("warning", "No databases configured yet."))
            input("Press Enter to continue...")
            return
        
        db_choices = databases + ["Cancel"]
        db_choice = simple_select("Which database should be the default?", db_choices)
        
        if db_choice != "Cancel":
            self.config.set_default_database(db_choice)
            self.console.print(self.art.get_status_indicator("success", f"Default database set to {db_choice}"))
        
        input("Press Enter to continue...")
    
    async def remove_database_wizard(self, db_manager: DatabaseManager):
        """Guide user through removing a database."""
        databases = list(self.config.config.databases.keys())
        
        if not databases:
            self.console.print(self.art.get_status_indicator("info", "No databases to remove."))
            input("Press Enter to continue...")
            return
        
        db_choices = databases + ["Cancel"]
        db_choice = simple_select("Which database would you like to remove?", db_choices)
        
        if db_choice == "Cancel":
            return
        
        if Confirm.ask(f"Are you sure you want to remove {db_choice}?"):
            await db_manager.remove_database(db_choice)

        input("Press Enter to continue...")

    async def discover_mongodb_wizard(self, db_manager: DatabaseManager):
        """Discover and connect to MongoDB databases."""
        self.console.clear()
        self.console.print(f"[bold {self.art.colors['primary']}]Discovering MongoDB Instances...[/bold {self.art.colors['primary']}]\n")

        # Discover local MongoDB instances
        mongo_manager = db_manager.mongodb_manager
        discovered = mongo_manager.discover_local_mongodb()

        if not discovered:
            self.console.print(self.art.get_status_indicator("warning", "No MongoDB instances found on localhost"))
            self.console.print("\n[dim]Make sure MongoDB is running on standard ports (27017-27019)[/dim]")
            input("\nPress Enter to continue...")
            return

        # Show discovered instances
        self.console.print(f"[bold green]Found {len(discovered)} MongoDB instance(s):[/bold green]\n")

        for i, instance in enumerate(discovered, 1):
            self.console.print(f"[bold cyan]{i}. MongoDB at {instance['host']}:{instance['port']}[/bold cyan]")
            self.console.print(f"   Version: {instance['version']}")
            self.console.print(f"   Databases: {instance['database_count']}")
            self.console.print(f"   [dim]Available: {', '.join(instance['databases'][:5])}{'...' if len(instance['databases']) > 5 else ''}[/dim]\n")

        # Select instance
        instance_choices = [f"{inst['host']}:{inst['port']}" for inst in discovered]
        instance_choices.append("Cancel")

        selected = simple_select("Select MongoDB instance to connect:", instance_choices)

        if selected == "Cancel":
            return

        # Find selected instance
        selected_instance = next((inst for inst in discovered if f"{inst['host']}:{inst['port']}" == selected), None)

        if not selected_instance:
            return

        # Show databases and let user select
        self.console.print(f"\n[bold {self.art.colors['primary']}]Available Databases:[/bold {self.art.colors['primary']}]")
        for db in selected_instance['databases']:
            self.console.print(f"  - {db}")

        self.console.print()

        # Ask which database to connect to
        db_choices = selected_instance['databases'] + ["Cancel"]
        selected_db = simple_select("Select database to connect:", db_choices)

        if selected_db == "Cancel":
            return

        # Ask for a reference name
        default_name = selected_db.replace("_", "-")
        db_name = input(f"\nDatabase name (for reference) [default: {default_name}]: ").strip() or default_name

        # Add the database
        success = await db_manager.add_database(
            db_name,
            "mongodb",
            host=selected_instance['host'],
            port=selected_instance['port'],
            database=selected_db,
            username=None,
            password=None
        )

        if success and not self.config.config.default_database:
            if Confirm.ask("Set this as the default database?"):
                self.config.set_default_database(db_name)

        input("\nPress Enter to continue...")

    async def huggingface_auth_menu(self):
        """HuggingFace authentication menu."""
        self.console.clear()
        self.console.print(Panel("HuggingFace Authentication", style=f"bold {self.art.colors['highlight']}"))
        
        model_manager = ModelManager(self.config)
        
        # Check current authentication status
        current_token = self.config.config.huggingface_token
        if current_token:
            self.console.print(self.art.get_status_indicator("success", "Currently authenticated with HuggingFace"))
        else:
            self.console.print(self.art.get_status_indicator("info", "Not currently authenticated"))
        
        self.console.print("\n[dim]Some models require HuggingFace authentication to download.[/dim]")
        self.console.print("[dim]Visit https://huggingface.co/settings/tokens to create a token.[/dim]\n")
        
        choices = [
            "Set HuggingFace Token",
            "Open HuggingFace Token Page",
            "Logout",
            "Back"
        ]
        
        choice = simple_select("What would you like to do?", choices)
        
        if choice == choices[0]:  # Set token
            token = input("Enter your HuggingFace token: ").strip()
            if token:
                success = await model_manager.setup_huggingface_auth(token)
                if success:
                    self.console.print(self.art.get_status_indicator("success", "Authentication successful!"))
                else:
                    self.console.print(self.art.get_status_indicator("error", "Authentication failed"))
        elif choice == choices[1]:  # Open browser
            import webbrowser
            webbrowser.open("https://huggingface.co/settings/tokens")
            self.console.print(self.art.get_status_indicator("info", "Opened HuggingFace token page in browser"))
        elif choice == choices[2]:  # Logout
            model_manager.logout_huggingface()
        
        if choice != choices[3]:  # Not back
            input("Press Enter to continue...")

    async def external_api_keys_menu(self):
        """External API keys management menu."""
        self.console.clear()
        self.console.print(Panel("External API Keys Management", style=f"bold {self.art.colors['primary']}"))

        # Show current API key status
        providers = {
            "anthropic": "Anthropic (Claude)",
            "xai": "XAI (Grok)",
            "google": "Google (Gemini)",
            "openai": "OpenAI (GPT)"
        }

        self.console.print("[bold]Current API Key Status:[/bold]")
        for provider, display_name in providers.items():
            api_key = self.config.get_api_key(provider)
            if api_key:
                masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
                self.console.print(f"  ✓ {display_name}: {masked_key}")
            else:
                self.console.print(f"  ✗ {display_name}: [dim]Not configured[/dim]")

        self.console.print()

        choices = [
            "Set Anthropic API Key (Claude)",
            "Set XAI API Key (Grok)",
            "Set Google API Key (Gemini)",
            "Set OpenAI API Key (GPT)",
            "Remove API Key",
            "Test API Key",
            "Back"
        ]

        choice = simple_select("What would you like to do?", choices)

        if choice == choices[0]:  # Anthropic
            await self._set_api_key_wizard("anthropic", "Anthropic", "https://console.anthropic.com/")
        elif choice == choices[1]:  # XAI
            await self._set_api_key_wizard("xai", "XAI", "https://x.ai/api")
        elif choice == choices[2]:  # Google
            await self._set_api_key_wizard("google", "Google AI Studio", "https://makersuite.google.com/app/apikey")
        elif choice == choices[3]:  # OpenAI
            await self._set_api_key_wizard("openai", "OpenAI", "https://platform.openai.com/api-keys")
        elif choice == choices[4]:  # Remove
            await self._remove_api_key_wizard()
        elif choice == choices[5]:  # Test
            await self._test_api_key_wizard()

        if choice != choices[6]:  # Not back
            input("Press Enter to continue...")

    async def _set_api_key_wizard(self, provider: str, display_name: str, url: str):
        """Wizard to set API key for a provider."""
        self.console.print(f"\n[bold cyan]Set {display_name} API Key[/bold cyan]")
        self.console.print(f"[dim]Get your API key at: {url}[/dim]\n")

        api_key = input(f"Enter your {display_name} API key (or press Enter to cancel): ").strip()

        if not api_key:
            self.console.print(self.art.get_status_indicator("info", "Cancelled"))
            return

        # Save the API key
        self.config.set_api_key(provider, api_key)
        self.console.print(self.art.get_status_indicator("success", f"{display_name} API key saved!"))

        # Optionally test the key
        if Confirm.ask("Would you like to test this API key now?"):
            from .models.external_api_manager import ExternalAPIManager
            api_manager = ExternalAPIManager(self.config)
            if api_manager.validate_api_key(provider, api_key):
                self.console.print(self.art.get_status_indicator("success", "API key is valid!"))
            else:
                self.console.print(self.art.get_status_indicator("error", "API key validation failed. Please check your key."))

    async def _remove_api_key_wizard(self):
        """Wizard to remove API key."""
        providers = {
            "anthropic": "Anthropic (Claude)",
            "xai": "XAI (Grok)",
            "google": "Google (Gemini)",
            "openai": "OpenAI (GPT)"
        }

        # Show only configured keys
        configured = [(p, n) for p, n in providers.items() if self.config.get_api_key(p)]

        if not configured:
            self.console.print(self.art.get_status_indicator("info", "No API keys configured"))
            return

        choices = [name for _, name in configured]
        choices.append("Cancel")

        choice = simple_select("Which API key would you like to remove?", choices)

        if choice == "Cancel":
            return

        # Find the provider
        provider = next(p for p, n in configured if n == choice)

        if Confirm.ask(f"Are you sure you want to remove the {choice} API key?"):
            self.config.remove_api_key(provider)
            self.console.print(self.art.get_status_indicator("success", f"{choice} API key removed"))

    async def _test_api_key_wizard(self):
        """Wizard to test API keys."""
        providers = {
            "anthropic": "Anthropic (Claude)",
            "xai": "XAI (Grok)",
            "google": "Google (Gemini)",
            "openai": "OpenAI (GPT)"
        }

        # Show only configured keys
        configured = [(p, n) for p, n in providers.items() if self.config.get_api_key(p)]

        if not configured:
            self.console.print(self.art.get_status_indicator("info", "No API keys configured"))
            return

        choices = [name for _, name in configured]
        choices.append("Cancel")

        choice = simple_select("Which API key would you like to test?", choices)

        if choice == "Cancel":
            return

        # Find the provider
        provider = next(p for p, n in configured if n == choice)
        api_key = self.config.get_api_key(provider)

        self.console.print(f"[blue]Testing {choice} API key...[/blue]")

        from .models.external_api_manager import ExternalAPIManager
        api_manager = ExternalAPIManager(self.config)

        if api_manager.validate_api_key(provider, api_key):
            self.console.print(self.art.get_status_indicator("success", "API key is valid!"))
        else:
            self.console.print(self.art.get_status_indicator("error", "API key validation failed. Please check your key."))

    async def general_settings_menu(self):
        """General settings menu."""
        self.console.clear()
        self.console.print(Panel("General Settings", style=f"bold {self.art.colors['primary']}"))
        
        current_config = self.config.config
        
        # Show current settings
        self.console.print(f"[bold]Current Settings:[/bold]")
        self.console.print(f"  Max Schema Tokens: {current_config.max_schema_tokens}")
        self.console.print(f"  Query Validation: {'Enabled' if current_config.enable_query_validation else 'Disabled'}")
        self.console.print(f"  Auto Correction: {'Enabled' if current_config.enable_auto_correction else 'Disabled'}")
        self.console.print(f"  Show Analysis: {'Enabled' if current_config.show_analysis else 'Disabled'}")
        self.console.print(f"  Theme: {current_config.theme}")
        self.console.print()
        
        # Allow editing
        if Confirm.ask("Would you like to modify any settings?"):
            # Max schema tokens
            new_tokens = input(f"Max schema tokens (current: {current_config.max_schema_tokens}): ").strip()
            if new_tokens.isdigit():
                current_config.max_schema_tokens = int(new_tokens)
            
            # Boolean settings
            current_config.enable_query_validation = Confirm.ask(
                "Enable query validation?",
                default=current_config.enable_query_validation
            )
            
            current_config.enable_auto_correction = Confirm.ask(
                "Enable auto correction?",
                default=current_config.enable_auto_correction
            )
            
            current_config.show_analysis = Confirm.ask(
                "Show query analysis?",
                default=current_config.show_analysis
            )
            
            # Save changes
            self.config.save_config()
            self.console.print(self.art.get_status_indicator("success", "Settings saved!"))
        
        input("Press Enter to continue...")
    
    async def show_system_info(self):
        """Show system information."""
        self.console.clear()
        self.console.print(Panel("System Information", style=f"bold {self.art.colors['primary']}"))
        
        sys_info = self.config.get_system_info()
        
        self.console.print(f"[bold]Platform:[/bold] {sys_info['platform']}")
        self.console.print(f"[bold]System:[/bold] {sys_info['system']}")
        self.console.print(f"[bold]Architecture:[/bold] {sys_info['machine']}")
        self.console.print(f"[bold]Python Version:[/bold] {sys_info['python_version']}")
        self.console.print(f"[bold]Total Memory:[/bold] {sys_info['total_memory_gb']:.1f} GB")
        self.console.print(f"[bold]Available Memory:[/bold] {sys_info['available_memory_gb']:.1f} GB")
        
        # Show model compatibility
        self.console.print("\n[bold]Model Compatibility:[/bold]")
        for model_id, model_config in self.config.SUPPORTED_MODELS.items():
            compatibility = self.config.check_model_compatibility(model_id)
            status = "✓" if compatibility["compatible"] else "✗"
            self.console.print(f"  {status} {model_config.name}")
            if not compatibility["compatible"]:
                self.console.print(f"    [dim]{compatibility['reason']}[/dim]")
        
        input("Press Enter to continue...")
    
    async def model_selection_wizard(self):
        """Quick model selection wizard for first-time setup."""
        self.console.print("\n[bold cyan]Let's set up your first AI model![/bold cyan]")
        
        # Show ALL available models - no compatibility restrictions
        auth_free_models = []
        auth_required_models = []
        
        for model_id, model_config in self.config.SUPPORTED_MODELS.items():
            # Skip compatibility check as requested by user
            # compatibility = self.config.check_model_compatibility(model_id)
            # if compatibility["compatible"]:
            if self._is_auth_error_likely(model_config):
                auth_required_models.append((model_id, model_config))
            else:
                auth_free_models.append((model_id, model_config))
        
        # Combine lists with auth-free models first
        all_models = auth_free_models + auth_required_models
        
        if not all_models:
            self.console.print(self.art.get_status_indicator("error", "No models defined"))
            return
        
        # Show information about authentication
        if auth_required_models:
            self.console.print("\n[dim]Note: Some models require HuggingFace authentication to download.[/dim]")
            self.console.print("[dim]Models marked with 🔒 need a free HuggingFace account.[/dim]")
        
        model_choices = []
        for model_id, config in all_models:
            auth_icon = "🔒 " if self._is_auth_error_likely(config) else "🆓 "
            model_choices.append(f"{auth_icon}{config.name} - {config.description}")
        
        model_choices.append("Skip for now")
        
        model_choice = simple_select("Which model would you like to download?", model_choices)
        
        if model_choice == "Skip for now":
            return
        
        # Get the selected model
        selected_index = model_choices.index(model_choice)
        model_id, model_config = all_models[selected_index]
        
        # Download and set as active
        model_manager = ModelManager(self.config)
        
        # Use the improved download flow
        await self._download_with_auth_retry(model_manager, model_id, model_config)
        
        # Set as active if download was successful
        if self.config.is_model_downloaded(model_id):
            self.config.set_selected_model(model_id)
            self.console.print(self.art.get_status_indicator("success", f"Successfully set up {model_config.name}!"))
        else:
            self.console.print(self.art.get_status_indicator("info", "Model setup incomplete. You can try again later with 't2s config'."))
            
    def _get_recommended_free_model(self) -> str:
        """Get a recommended model that doesn't require authentication."""
        # Prefer models that are typically open without gating
        free_model_preferences = [
            "smollm-1.7b",  # Usually open
            "defog-sqlcoder-7b",  # Often open for SQL tasks
        ]
        
        for model_id in free_model_preferences:
            if model_id in self.config.SUPPORTED_MODELS:
                # Skip compatibility check as requested by user
                # compatibility = self.config.check_model_compatibility(model_id)
                # if compatibility["compatible"]:
                return model_id
        
        return None
    
    async def database_selection_wizard(self):
        """Quick database selection wizard for first-time setup."""
        self.console.print("\n[bold cyan]Let's set up your first database![/bold cyan]")
        
        # Check if we found any databases during auto-discovery
        databases = list(self.config.config.databases.keys())
        
        if databases:
            self.console.print("Found these databases:")
            for db_name in databases:
                db_config = self.config.config.databases[db_name]
                self.console.print(f"  • {db_name} ({db_config.type})")
            
            db_choices = databases + ["Add a new database", "Skip for now"]
            db_choice = simple_select("Which database would you like to use as default?", db_choices)
            
            if db_choice in databases:
                self.config.set_default_database(db_choice)
                self.console.print(self.art.get_status_indicator("success", f"Set {db_choice} as default database!"))
                return
            elif db_choice == "Skip for now":
                return
        
        # Add new database
        db_manager = DatabaseManager(self.config)
        await self.add_database_wizard(db_manager)


# CLI Commands
@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', is_flag=True, help='Show version information')
def main(ctx, version):
    """T2S - Text to SQL CLI tool."""
    if version:
        click.echo(f"T2S v{__version__}")
        click.echo("Created by Lakshman Turlapati")
        return
    
    if ctx.invoked_subcommand is None:
        # Default to interactive mode
        cli = T2SCLI()
        try:
            asyncio.run(cli.interactive_mode())
        except KeyboardInterrupt:
            click.echo("\nGoodbye! 👋")


@main.command()
@click.argument('query', required=True)
@click.option('--database', '-d', help='Database to query against')
def query(query, database):
    """Execute a direct query."""
    cli = T2SCLI()
    try:
        asyncio.run(cli.direct_query(query, database))
    except KeyboardInterrupt:
        click.echo("\nQuery interrupted")


@main.command()
def config():
    """Open configuration menu."""
    cli = T2SCLI()
    try:
        asyncio.run(cli.configuration_menu())
    except KeyboardInterrupt:
        click.echo("\nConfiguration cancelled")


@main.command()
def models():
    """Manage AI models."""
    cli = T2SCLI()
    try:
        asyncio.run(cli.model_management_menu())
    except KeyboardInterrupt:
        click.echo("\nModel management cancelled")


@main.command()
def databases():
    """Manage databases."""
    cli = T2SCLI()
    try:
        asyncio.run(cli.database_management_menu())
    except KeyboardInterrupt:
        click.echo("\nDatabase management cancelled")


@main.command()
def info():
    """Show system information."""
    cli = T2SCLI()
    try:
        asyncio.run(cli.show_system_info())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main() 