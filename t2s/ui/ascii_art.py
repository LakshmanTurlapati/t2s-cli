"""ASCII art and branding for T2S terminal interface."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import random
from .. import __version__  # Import version from package


class T2SArt:
    """ASCII art and branding for T2S."""
    
    # Main T2S logo
    LOGO = """
████████╗██████╗ ███████╗
╚══██╔══╝╚════██╗██╔════╝
   ██║    █████╔╝███████╗
   ██║   ██╔═══╝ ╚════██║
   ██║   ███████╗███████║
   ╚═╝   ╚══════╝╚══════╝
"""
    
    # Alternative compact logo
    COMPACT_LOGO = """
████████╗██████╗ ███████╗
╚══██╔══╝╚════██╗██╔════╝
   ██║   █████╔╝ ███████╗
   ╚═╝   ╚══════╝╚══════╝
"""
    
    # Banner text
    BANNER_TEXT = "Text to SQL"
    SUBTITLE_TEXT = "AI-Powered Database Query Generator"
    AUTHOR_TEXT = "Created by Lakshman Turlapati"
    REPO_TEXT = "https://github.com/lakshmanturlapati/t2s-cli"
    
    # Loading animations
    LOADING_FRAMES = [
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
    ]
    
    # Success/Error symbols
    SUCCESS_SYMBOLS = ["✓", "✅", "🎉", "✨"]
    ERROR_SYMBOLS = ["✗", "❌", "💥", "⚠️"]
    INFO_SYMBOLS = ["ℹ️", "💡", "📝", "🔍"]
    
    @classmethod
    def get_welcome_banner(cls, console: Console) -> Panel:
        """Get the main welcome banner."""
        # Create the logo text with gradient colors
        logo_text = Text()
        logo_lines = cls.LOGO.strip().split('\n')
        
        colors = ["bright_cyan", "cyan", "blue", "bright_blue", "magenta", "bright_magenta"]
        
        for i, line in enumerate(logo_lines):
            color = colors[i % len(colors)]
            logo_text.append(line + "\n", style=color)
        
        # Add banner text
        banner = Text(cls.BANNER_TEXT, style="bold bright_yellow", justify="center")
        subtitle = Text(cls.SUBTITLE_TEXT, style="italic bright_green", justify="center")
        
        # Combine all elements
        content = Text()
        content.append_text(logo_text)
        content.append("\n")
        content.append_text(banner)
        content.append("\n")
        content.append_text(subtitle)
        content.append("\n\n")
        content.append(cls.AUTHOR_TEXT, style="dim cyan")
        content.append("\n")
        content.append(cls.REPO_TEXT, style="dim blue link")
        
        return Panel(
            Align.center(content),
            border_style="bright_cyan",
            title="[bold bright_yellow]Welcome to T2S[/bold bright_yellow]",
            title_align="center",
            padding=(1, 2)
        )
    
    @classmethod
    def get_compact_header(cls) -> Text:
        """Get a compact header for ongoing sessions."""
        text = Text()
        text.append("T2S", style="bold bright_cyan")
        text.append(" | ", style="dim")
        text.append("Text to SQL", style="bright_yellow")
        return text
    
    @classmethod
    def get_status_indicator(cls, status: str, message: str = "") -> Text:
        """Get a status indicator with appropriate symbol and color."""
        text = Text()
        
        if status == "success":
            symbol = random.choice(cls.SUCCESS_SYMBOLS)
            text.append(f"{symbol} ", style="bright_green")
            text.append(message, style="green")
        elif status == "error":
            symbol = random.choice(cls.ERROR_SYMBOLS)
            text.append(f"{symbol} ", style="bright_red")
            text.append(message, style="red")
        elif status == "info":
            symbol = random.choice(cls.INFO_SYMBOLS)
            text.append(f"{symbol} ", style="bright_blue")
            text.append(message, style="blue")
        elif status == "warning":
            text.append("⚠️ ", style="bright_yellow")
            text.append(message, style="yellow")
        else:
            text.append(message)
        
        return text
    
    @classmethod
    def get_loading_animation(cls, frame_index: int) -> str:
        """Get the current frame of loading animation."""
        return cls.LOADING_FRAMES[frame_index % len(cls.LOADING_FRAMES)]
    
    @classmethod
    def get_separator(cls, width: int = 60, style: str = "─") -> Text:
        """Get a separator line."""
        return Text(style * width, style="dim")
    
    @classmethod
    def get_model_card(cls, model_name: str, status: str, details: dict) -> Panel:
        """Get a card display for a model."""
        content = Text()
        
        # Model name
        content.append(f"🤖 {model_name}\n", style="bold bright_cyan")
        
        # Status with appropriate color
        if status == "downloaded":
            content.append("Status: ", style="dim")
            content.append("Downloaded ✓", style="bright_green")
        elif status == "available":
            content.append("Status: ", style="dim")
            content.append("Available for download", style="yellow")
        else:
            content.append("Status: ", style="dim")
            content.append(status.title(), style="red")
        
        content.append("\n")
        
        # Model details
        if "parameters" in details:
            content.append(f"Parameters: {details['parameters']}\n", style="cyan")
        if "description" in details:
            content.append(f"Description: {details['description']}\n", style="dim")
        if "download_size_gb" in details:
            content.append(f"Size: {details['download_size_gb']:.1f} GB\n", style="magenta")
        
        # Compatibility info
        if "compatibility" in details:
            comp = details["compatibility"]
            if comp.get("compatible"):
                content.append("Memory: ✓ Compatible", style="green")
            else:
                content.append("Memory: ✗ ", style="red")
                content.append(f"{comp.get('reason', 'Insufficient RAM')}", style="bright_red")
                # Show required vs available memory if available
                if "required_ram_gb" in comp and "available_ram_gb" in comp:
                    content.append(f"\nRequired: {comp['required_ram_gb']}GB | Available: {comp['available_ram_gb']:.1f}GB", style="red")
        
        # Determine border color based on compatibility
        if "compatibility" in details and not details["compatibility"].get("compatible"):
            border_color = "red"  # Red for incompatible models
        elif status == "downloaded":
            border_color = "green"
        elif status == "available":
            border_color = "yellow" 
        else:
            border_color = "red"
        
        return Panel(
            content,
            border_style=border_color,
            title=f"[bold]{model_name}[/bold]",
            title_align="left",
            padding=(0, 1)
        )
    
    @classmethod
    def get_database_card(cls, db_name: str, db_type: str, status: str, connection_info: str) -> Panel:
        """Get a card display for a database."""
        content = Text()
        
        # Database name and type
        content.append(f"🗄️ {db_name} ", style="bold bright_blue")
        content.append(f"({db_type})\n", style="dim")
        
        # Connection info
        content.append("Connection: ", style="dim")
        content.append(f"{connection_info}\n", style="cyan")
        
        # Status
        content.append("Status: ", style="dim")
        if status == "connected":
            content.append("Connected ✓", style="bright_green")
        elif status == "error":
            content.append("Connection Error ✗", style="bright_red")
        else:
            content.append(status.title(), style="yellow")
        
        border_color = "green" if status == "connected" else "red" if status == "error" else "yellow"
        
        return Panel(
            content,
            border_style=border_color,
            title=f"[bold]{db_name}[/bold]",
            title_align="left",
            padding=(0, 1)
        )
    
    @classmethod
    def get_help_panel(cls) -> Panel:
        """Get the help panel with available commands."""
        content = Text()
        
        commands = [
            ("t2s", "Launch interactive mode"),
            ("t2s query <text>", "Direct query mode"),
            ("t2s config", "Configuration menu"),
            ("t2s models", "Manage AI models"),
            ("t2s databases", "Manage databases"),
            ("t2s --help", "Show detailed help"),
        ]
        
        content.append("Available Commands:\n\n", style="bold bright_yellow")
        
        for cmd, desc in commands:
            content.append(f"  {cmd}", style="bright_cyan")
            content.append(f" - {desc}\n", style="dim")
        
        content.append("\nTip: Start with 't2s config' to set up your first model and database!", 
                      style="italic bright_green")
        
        return Panel(
            content,
            title="[bold bright_blue]T2S Help[/bold bright_blue]",
            border_style="blue",
            padding=(1, 2)
        )
    
    @classmethod
    def get_progress_bar_style(cls) -> dict:
        """Get the style configuration for progress bars."""
        return {
            "bar_width": 40,
            "complete_style": "bright_green",
            "finished_style": "bright_cyan",
            "progress_style": "bright_blue",
        }
    
    @classmethod
    def get_query_result_header(cls, query: str) -> Panel:
        """Get a header panel for query results."""
        content = Text()
        content.append("🔍 Query: ", style="bright_blue")
        content.append(query, style="cyan")
        
        return Panel(
            content,
            title="[bold bright_yellow]Query Execution[/bold bright_yellow]",
            border_style="yellow",
            padding=(0, 1)
        )
    
    @classmethod
    def get_footer(cls) -> Text:
        """Get the footer text."""
        text = Text()
        text.append(f"T2S v{__version__}", style="dim")
        text.append(" | ", style="dim")
        text.append("Created by Lakshman Turlapati", style="dim cyan")
        text.append(" | ", style="dim")
        text.append("Press Ctrl+C to exit", style="dim")
        return text 