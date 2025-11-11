"""ASCII art and branding for T2S terminal interface."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import random
from .. import __version__  # Import version from package


class T2SArt:
    """ASCII art and branding for T2S."""

    # Color themes based on database type
    THEMES = {
        "sql": {
            "primary": "bright_cyan",
            "secondary": "cyan",
            "accent": "blue",
            "bright_accent": "bright_blue",
            "highlight": "bright_yellow",
            "logo_colors": ["bright_cyan", "cyan", "blue", "bright_blue", "magenta", "bright_magenta"]
        },
        "mongodb": {
            "primary": "bright_green",
            "secondary": "green",
            "accent": "bright_green",
            "bright_accent": "green",
            "highlight": "bright_yellow",
            "logo_colors": ["bright_green", "green", "bright_green", "green", "bright_green", "green"]
        }
    }

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

    def __init__(self, theme="sql"):
        """Initialize with a theme (sql or mongodb)."""
        self.theme = theme if theme in self.THEMES else "sql"
        self.colors = self.THEMES[self.theme]

    def get_theme_color(self, color_key):
        """Get a color from the current theme."""
        return self.colors.get(color_key, "white")

    def get_welcome_banner(self, console: Console) -> Panel:
        """Get the main welcome banner."""
        # Create the logo text with gradient colors from theme
        logo_text = Text()
        logo_lines = self.LOGO.strip().split('\n')

        colors = self.colors["logo_colors"]

        for i, line in enumerate(logo_lines):
            color = colors[i % len(colors)]
            logo_text.append(line + "\n", style=color)

        # Add banner text
        banner = Text(self.BANNER_TEXT, style=f"bold {self.colors['highlight']}", justify="center")
        subtitle = Text(self.SUBTITLE_TEXT, style=f"italic {self.colors['primary']}", justify="center")

        # Combine all elements
        content = Text()
        content.append_text(logo_text)
        content.append("\n")
        content.append_text(banner)
        content.append("\n")
        content.append_text(subtitle)
        content.append("\n\n")
        content.append(self.AUTHOR_TEXT, style=f"dim {self.colors['secondary']}")
        content.append("\n")
        content.append(self.REPO_TEXT, style=f"dim {self.colors['accent']} link")

        return Panel(
            Align.center(content),
            border_style=self.colors["primary"],
            title=f"[bold {self.colors['highlight']}]Welcome to T2S[/bold {self.colors['highlight']}]",
            title_align="center",
            padding=(1, 2)
        )
    
    def get_compact_header(self) -> Text:
        """Get a compact header for ongoing sessions."""
        text = Text()
        text.append("T2S", style=f"bold {self.colors['primary']}")
        text.append(" | ", style="dim")
        # Show MQL for MongoDB, SQL for others
        label = "Text to MQL" if self.theme == "mongodb" else "Text to SQL"
        text.append(label, style=f"{self.colors['highlight']}")
        return text

    def get_status_indicator(self, status: str, message: str = "") -> Text:
        """Get a status indicator with appropriate symbol and color."""
        text = Text()

        if status == "success":
            symbol = random.choice(self.SUCCESS_SYMBOLS)
            text.append(f"{symbol} ", style="bright_green")
            text.append(message, style="green")
        elif status == "error":
            symbol = random.choice(self.ERROR_SYMBOLS)
            text.append(f"{symbol} ", style="bright_red")
            text.append(message, style="red")
        elif status == "info":
            symbol = random.choice(self.INFO_SYMBOLS)
            text.append(f"{symbol} ", style=self.colors['primary'])
            text.append(message, style=self.colors['accent'])
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

    def get_model_card(self, model_name: str, status: str, details: dict) -> Panel:
        """Get a card display for a model."""
        content = Text()

        # Model name
        content.append(f"🤖 {model_name}\n", style=f"bold {self.colors['primary']}")
        
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
    
    def get_database_card(self, db_name: str, db_type: str, status: str, connection_info: str) -> Panel:
        """Get a card display for a database."""
        content = Text()

        # Database name and type
        content.append(f"🗄️ {db_name} ", style=f"bold {self.colors['primary']}")
        content.append(f"({db_type})\n", style="dim")

        # Connection info
        content.append("Connection: ", style="dim")
        content.append(f"{connection_info}\n", style=self.colors['secondary'])
        
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
    
    def get_help_panel(self) -> Panel:
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

        content.append("Available Commands:\n\n", style=f"bold {self.colors['highlight']}")

        for cmd, desc in commands:
            content.append(f"  {cmd}", style=self.colors['primary'])
            content.append(f" - {desc}\n", style="dim")

        content.append("\nTip: Start with 't2s config' to set up your first model and database!",
                      style=f"italic {self.colors['primary']}")

        return Panel(
            content,
            title=f"[bold {self.colors['primary']}]T2S Help[/bold {self.colors['primary']}]",
            border_style=self.colors['accent'],
            padding=(1, 2)
        )
    
    def get_progress_bar_style(self) -> dict:
        """Get the style configuration for progress bars."""
        return {
            "bar_width": 40,
            "complete_style": "bright_green",
            "finished_style": self.colors['primary'],
            "progress_style": self.colors['bright_accent'],
        }

    def get_query_result_header(self, query: str) -> Panel:
        """Get a header panel for query results."""
        content = Text()
        content.append("🔍 Query: ", style=self.colors['primary'])
        content.append(query, style=self.colors['secondary'])

        return Panel(
            content,
            title=f"[bold {self.colors['highlight']}]Query Execution[/bold {self.colors['highlight']}]",
            border_style=self.colors['highlight'],
            padding=(0, 1)
        )

    def get_footer(self) -> Text:
        """Get the footer text."""
        text = Text()
        text.append(f"T2S v{__version__}", style="dim")
        text.append(" | ", style="dim")
        text.append("Created by Lakshman Turlapati", style=f"dim {self.colors['secondary']}")
        text.append(" | ", style="dim")
        text.append("Press Ctrl+C to exit", style="dim")
        return text