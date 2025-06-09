"""Database management for T2S - handles connections and query execution."""

import asyncio
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError
from rich.console import Console
from rich.table import Table

from ..core.config import Config, DatabaseConfig


class DatabaseManager:
    """Manages database connections and query execution."""
    
    def __init__(self, config: Config):
        """Initialize the database manager."""
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, Any] = {}
        self.engines: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        self.console.print("[blue]Initializing database connections...[/blue]")
        
        # Auto-discover databases if none configured
        if not self.config.config.databases:
            await self.auto_discover_databases()
        
        # Test configured connections
        for db_name in self.config.config.databases.keys():
            try:
                await self.test_connection(db_name)
            except Exception as e:
                self.logger.warning(f"Failed to connect to {db_name}: {e}")
        
        self.console.print("[green]Database manager initialized[/green]")
    
    async def auto_discover_databases(self) -> None:
        """Auto-discover local databases."""
        discovered = []
        
        # Look for SQLite databases in common locations
        search_paths = [
            Path.home(),
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path.cwd(),
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                # Find .db, .sqlite, .sqlite3 files
                for pattern in ["*.db", "*.sqlite", "*.sqlite3"]:
                    for db_file in search_path.glob(pattern):
                        if db_file.is_file() and db_file.stat().st_size > 0:
                            db_name = db_file.stem
                            db_config = DatabaseConfig(
                                name=db_name,
                                type="sqlite",
                                path=str(db_file)
                            )
                            self.config.add_database(db_name, db_config)
                            discovered.append(db_name)
        
        # Try to connect to common local database services
        common_services = [
            ("localhost_postgres", "postgresql", "localhost", 5432, "postgres"),
            ("localhost_mysql", "mysql", "localhost", 3306, "mysql"),
        ]
        
        for name, db_type, host, port, database in common_services:
            try:
                if db_type == "postgresql":
                    # Try common PostgreSQL credentials
                    for user in ["postgres", "admin"]:
                        db_config = DatabaseConfig(
                            name=name,
                            type=db_type,
                            host=host,
                            port=port,
                            database=database,
                            username=user,
                            password=""  # Try without password first
                        )
                        if await self._test_connection_config(db_config):
                            self.config.add_database(name, db_config)
                            discovered.append(name)
                            break
                
                elif db_type == "mysql":
                    # Try common MySQL credentials
                    for user in ["root", "admin"]:
                        db_config = DatabaseConfig(
                            name=name,
                            type=db_type,
                            host=host,
                            port=port,
                            database=database,
                            username=user,
                            password=""  # Try without password first
                        )
                        if await self._test_connection_config(db_config):
                            self.config.add_database(name, db_config)
                            discovered.append(name)
                            break
            except Exception:
                continue  # Service not available
        
        if discovered:
            self.console.print(f"[green]Auto-discovered databases: {', '.join(discovered)}[/green]")
        else:
            self.console.print("[yellow]No databases auto-discovered. You can add them manually.[/yellow]")
    
    def _create_connection_string(self, db_config: DatabaseConfig) -> str:
        """Create a SQLAlchemy connection string."""
        if db_config.type == "sqlite":
            return f"sqlite:///{db_config.path}"
        
        elif db_config.type == "postgresql":
            password_part = f":{db_config.password}" if db_config.password else ""
            return f"postgresql://{db_config.username}{password_part}@{db_config.host}:{db_config.port}/{db_config.database}"
        
        elif db_config.type == "mysql":
            password_part = f":{db_config.password}" if db_config.password else ""
            return f"mysql+pymysql://{db_config.username}{password_part}@{db_config.host}:{db_config.port}/{db_config.database}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_config.type}")
    
    async def _test_connection_config(self, db_config: DatabaseConfig) -> bool:
        """Test a database configuration."""
        try:
            connection_string = self._create_connection_string(db_config)
            engine = create_engine(connection_string, pool_timeout=5, pool_recycle=300)
            
            with engine.connect() as conn:
                # Simple test query
                if db_config.type == "sqlite":
                    conn.execute(text("SELECT 1"))
                else:
                    conn.execute(text("SELECT 1"))
            
            engine.dispose()
            return True
            
        except Exception:
            return False
    
    async def test_connection(self, db_name: str) -> bool:
        """Test a database connection."""
        if db_name not in self.config.config.databases:
            raise ValueError(f"Database {db_name} not configured")
        
        db_config = self.config.config.databases[db_name]
        return await self._test_connection_config(db_config)
    
    def get_connection(self, db_name: str) -> Any:
        """Get a database connection/engine."""
        if db_name not in self.config.config.databases:
            raise ValueError(f"Database {db_name} not configured")
        
        if db_name not in self.engines:
            db_config = self.config.config.databases[db_name]
            connection_string = self._create_connection_string(db_config)
            self.engines[db_name] = create_engine(
                connection_string,
                pool_timeout=10,
                pool_recycle=300,
                echo=False
            )
        
        return self.engines[db_name]
    
    async def execute_query(self, sql: str, db_name: str) -> Dict[str, Any]:
        """Execute a SQL query and return results."""
        engine = self.get_connection(db_name)
        
        try:
            with engine.connect() as conn:
                # Execute the query
                result = conn.execute(text(sql))
                
                # Handle different query types
                if sql.strip().upper().startswith('SELECT'):
                    # For SELECT queries, return data as DataFrame
                    data = pd.read_sql(sql, conn)
                    return {
                        "data": data,
                        "rows_affected": len(data),
                        "query_type": "select"
                    }
                else:
                    # For INSERT, UPDATE, DELETE, etc.
                    rows_affected = result.rowcount
                    conn.commit()
                    return {
                        "data": None,
                        "rows_affected": rows_affected,
                        "query_type": "modification"
                    }
                    
        except SQLAlchemyError as e:
            self.logger.error(f"Database error executing query: {e}")
            raise RuntimeError(f"Database error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {e}")
            raise RuntimeError(f"Query execution error: {str(e)}")
    
    async def get_schema_info(self, db_name: str) -> Dict[str, Any]:
        """Get comprehensive schema information for a database."""
        engine = self.get_connection(db_name)
        
        try:
            inspector = inspect(engine)
            schema_info = {
                "tables": {},
                "relationships": []
            }
            
            # Get all table names
            table_names = inspector.get_table_names()
            
            for table_name in table_names:
                table_info = {
                    "columns": [],
                    "column_types": {},
                    "primary_keys": [],
                    "foreign_keys": [],
                    "indexes": []
                }
                
                # Get column information
                columns = inspector.get_columns(table_name)
                for column in columns:
                    column_name = column['name']
                    table_info["columns"].append(column_name)
                    table_info["column_types"][column_name] = str(column['type'])
                
                # Get primary keys
                pk_constraint = inspector.get_pk_constraint(table_name)
                if pk_constraint:
                    table_info["primary_keys"] = pk_constraint.get('constrained_columns', [])
                
                # Get foreign keys
                fk_constraints = inspector.get_foreign_keys(table_name)
                for fk in fk_constraints:
                    table_info["foreign_keys"].append({
                        "constrained_columns": fk.get('constrained_columns', []),
                        "referred_table": fk.get('referred_table'),
                        "referred_columns": fk.get('referred_columns', [])
                    })
                    
                    # Add to relationships
                    rel_desc = f"{table_name}({', '.join(fk.get('constrained_columns', []))}) -> {fk.get('referred_table')}({', '.join(fk.get('referred_columns', []))})"
                    schema_info["relationships"].append(rel_desc)
                
                # Get indexes
                indexes = inspector.get_indexes(table_name)
                for index in indexes:
                    table_info["indexes"].append({
                        "name": index.get('name'),
                        "columns": index.get('column_names', []),
                        "unique": index.get('unique', False)
                    })
                
                schema_info["tables"][table_name] = table_info
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {e}")
            raise RuntimeError(f"Schema analysis error: {str(e)}")
    
    async def add_database(self, name: str, db_type: str, **kwargs) -> bool:
        """Add a new database configuration."""
        try:
            db_config = DatabaseConfig(
                name=name,
                type=db_type,
                **kwargs
            )
            
            # Test the connection
            if await self._test_connection_config(db_config):
                self.config.add_database(name, db_config)
                self.console.print(f"[green]Successfully added database '{name}'[/green]")
                return True
            else:
                self.console.print(f"[red]Failed to connect to database '{name}'[/red]")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding database: {e}")
            self.console.print(f"[red]Error adding database: {e}[/red]")
            return False
    
    async def remove_database(self, name: str) -> bool:
        """Remove a database configuration."""
        try:
            if name in self.engines:
                self.engines[name].dispose()
                del self.engines[name]
            
            self.config.remove_database(name)
            self.console.print(f"[green]Successfully removed database '{name}'[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing database: {e}")
            self.console.print(f"[red]Error removing database: {e}[/red]")
            return False
    
    def list_databases(self) -> List[Dict[str, Any]]:
        """List all configured databases with their status."""
        databases = []
        
        for name, db_config in self.config.config.databases.items():
            db_info = {
                "name": name,
                "type": db_config.type,
                "status": "unknown"
            }
            
            # Add connection details based on type
            if db_config.type == "sqlite":
                db_info["path"] = db_config.path
            else:
                db_info["host"] = db_config.host
                db_info["port"] = db_config.port
                db_info["database"] = db_config.database
                db_info["username"] = db_config.username
            
            # Test connection status
            try:
                # Quick synchronous test
                engine = self.get_connection(name)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                db_info["status"] = "connected"
            except Exception:
                db_info["status"] = "error"
            
            databases.append(db_info)
        
        return databases
    
    def display_databases(self) -> None:
        """Display databases in a formatted table."""
        databases = self.list_databases()
        
        if not databases:
            self.console.print("[yellow]No databases configured[/yellow]")
            return
        
        table = Table(title="Configured Databases")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Connection", style="blue")
        table.add_column("Status", style="green")
        
        for db in databases:
            if db["type"] == "sqlite":
                connection = f"File: {db.get('path', 'N/A')}"
            else:
                connection = f"{db.get('host', 'N/A')}:{db.get('port', 'N/A')}/{db.get('database', 'N/A')}"
            
            status_color = "green" if db["status"] == "connected" else "red"
            
            table.add_row(
                db["name"],
                db["type"],
                connection,
                f"[{status_color}]{db['status']}[/{status_color}]"
            )
        
        self.console.print(table)
    
    def close_all_connections(self) -> None:
        """Close all database connections."""
        for engine in self.engines.values():
            try:
                engine.dispose()
            except Exception as e:
                self.logger.warning(f"Error closing connection: {e}")
        
        self.engines.clear()
        self.console.print("[green]All database connections closed[/green]") 