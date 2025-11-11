"""Main T2S Engine for text-to-SQL conversion."""

import re
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import sqlparse
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config
from ..models.model_manager import ModelManager
from ..database.db_manager import DatabaseManager
from ..utils.schema_analyzer import SchemaAnalyzer
from ..utils.sql_validator import SQLValidator
from .sql_engine import SQLEngine
from .mql_engine import MQLEngine
from ..utils.mql_validator import MQLValidator


@dataclass
class QueryResult:
    """Result of a T2S query execution."""
    original_query: str
    generated_sql: str
    validated_sql: str
    execution_time: float
    rows_affected: int
    data: Optional[pd.DataFrame]
    analysis: str
    error: Optional[str] = None


class T2SEngine:
    """Main engine for Text-to-SQL conversion and execution."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the T2S engine."""
        self.config = config or Config()
        self.console = Console()
        self.model_manager = ModelManager(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.schema_analyzer = SchemaAnalyzer(self.config)

        # Initialize SQL and MQL engines
        self.sql_engine = SQLEngine(self.config, self.model_manager)
        self.mql_engine = MQLEngine(self.config, self.model_manager)
        self.mql_validator = MQLValidator()

        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the engine components."""
        self.console.print("[blue]Initializing T2S Engine...[/blue]")
        
        # Show active database information
        default_db = self.config.config.default_database
        if default_db:
            # Get database type for display
            if default_db in self.config.config.databases:
                db_config = self.config.config.databases[default_db]
                db_type = db_config.type.upper()
                self.console.print(f"[blue]Active database: {default_db} ({db_type})[/blue]")
            else:
                self.console.print(f"[blue]Active database: {default_db}[/blue]")
        else:
            self.console.print("[yellow]No default database configured[/yellow]")
        
        # Check if a model is selected and available
        if not self.config.config.selected_model:
            self.console.print("[yellow]No model selected. Please configure a model first.[/yellow]")
            return

        # Initialize model manager
        # Only show spinner for local models (not external API models)
        selected_model = self.config.config.selected_model
        is_external_api = self.model_manager.external_api_manager.is_api_model(selected_model)

        if is_external_api:
            # External API models don't need initialization - just initialize silently
            await self.model_manager.initialize()
        else:
            # Local models need loading - show spinner
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Initializing AI model[/cyan]"),
                console=self.console,
                transient=True  # Make it disappear after completion
            ) as progress:
                task = progress.add_task("", total=None)
                await self.model_manager.initialize()

        # Initialize database connections
        await self.db_manager.initialize()
        
        self.console.print("[green]T2S Engine initialized successfully![/green]")
    
    async def process_query(self, natural_language_query: str, database_name: Optional[str] = None) -> QueryResult:
        """Process a natural language query and return results."""
        # Detect database type and route to appropriate engine
        db_type = self._detect_database_type()

        if db_type == "mongodb":
            return await self._process_mql_query(natural_language_query, database_name)
        else:
            return await self._process_sql_query(natural_language_query, database_name)

    async def _process_sql_query(self, natural_language_query: str, database_name: Optional[str] = None) -> QueryResult:
        """Process a SQL query using the SQL engine."""
        start_time = datetime.now()
        generated_sql = ""  # Initialize to preserve in error cases
        validated_sql = ""

        try:
            # Step 1: Analyze and get relevant schema
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Analyzing database schema[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                schema_info = await self._get_relevant_schema(natural_language_query, database_name)

            # Step 2: Generate SQL using SQL engine
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Generating SQL query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                generated_sql = await self.sql_engine.generate_sql(natural_language_query, schema_info)

            # Step 3: Validate and correct SQL
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Validating SQL query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                validated_sql = await self.sql_engine.validate_sql(generated_sql)

            # Step 4: Execute SQL
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Executing query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                execution_result = await self._execute_sql(validated_sql, database_name)

            # Step 5: Generate analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Generating analysis[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                analysis = await self._generate_analysis(
                    natural_language_query,
                    validated_sql,
                    execution_result["data"]
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                original_query=natural_language_query,
                generated_sql=generated_sql,
                validated_sql=validated_sql,
                execution_time=execution_time,
                rows_affected=execution_result.get("rows_affected", 0),
                data=execution_result.get("data"),
                analysis=analysis
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error processing SQL query: {e}")
            if generated_sql:
                self.logger.info(f"Generated SQL before error: {generated_sql}")

            return QueryResult(
                original_query=natural_language_query,
                generated_sql=generated_sql,  # Preserve generated SQL even on error
                validated_sql=validated_sql,  # Preserve validated SQL if available
                execution_time=execution_time,
                rows_affected=0,
                data=None,
                analysis="",
                error=str(e)
            )

    async def _process_mql_query(self, natural_language_query: str, database_name: Optional[str] = None) -> QueryResult:
        """Process a MongoDB query using the MQL engine."""
        start_time = datetime.now()
        generated_mql = ""  # Initialize to preserve in error cases
        validated_mql = ""

        try:
            # Step 1: Analyze and get relevant schema
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Analyzing database schema[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                schema_info = await self._get_relevant_schema(natural_language_query, database_name)

            # Step 2: Generate MQL using MQL engine
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Generating MQL query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                generated_mql = await self.mql_engine.generate_mql(natural_language_query, schema_info)

            # Step 3: Validate MQL
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Validating MQL query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                validated_mql = await self.mql_validator.validate_and_correct(generated_mql)

            # Step 3.5: Validate fields against schema (skip for aggregation pipelines)
            # Aggregation pipelines create dynamic fields through stages, so field validation produces false positives
            if '.aggregate(' not in validated_mql:
                is_valid, error_msg = self.mql_engine.validate_mql_fields(validated_mql, schema_info)
                if not is_valid:
                    self.logger.warning(f"MQL field validation failed for query '{validated_mql}': {error_msg}")
                    self.console.print(f"[yellow]Warning: {error_msg}[/yellow]")
                    self.console.print(f"[yellow]Generated query: {validated_mql}[/yellow]")
                    # Continue anyway, but log the issue
            else:
                self.logger.debug("Skipping field validation for aggregation pipeline (dynamic fields)")

            # Step 4: Execute MQL
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Executing query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                execution_result = await self._execute_sql(validated_mql, database_name)  # Uses db_manager which routes to MongoDB

            # Step 5: Generate analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Generating analysis[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                analysis = await self._generate_analysis(
                    natural_language_query,
                    validated_mql,
                    execution_result["data"]
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                original_query=natural_language_query,
                generated_sql=generated_mql,
                validated_sql=validated_mql,
                execution_time=execution_time,
                rows_affected=execution_result.get("rows_affected", 0),
                data=execution_result.get("data"),
                analysis=analysis
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error processing MQL query: {e}")
            if generated_mql:
                self.logger.info(f"Generated MQL before error: {generated_mql}")

            return QueryResult(
                original_query=natural_language_query,
                generated_sql=generated_mql,  # Preserve generated MQL even on error
                validated_sql=validated_mql,  # Preserve validated MQL if available
                execution_time=execution_time,
                rows_affected=0,
                data=None,
                analysis="",
                error=str(e)
            )
    
    async def _get_relevant_schema(self, query: str, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get optimized database schema information that fits within model token limits."""
        db_name = database_name or self.config.config.default_database
        if not db_name:
            raise ValueError("No database specified and no default database configured")

        # Get database connection
        db_connection = self.db_manager.get_connection(db_name)

        # Get full schema first
        from ..database.db_manager import DatabaseManager
        db_manager = DatabaseManager(self.config)
        full_schema = await db_manager.get_schema_info(db_name)

        # Detect database type
        db_config = self.config.config.databases[db_name]
        db_type = db_config.type.lower()

        # For MongoDB, return the schema as-is (collections structure)
        if db_type == "mongodb":
            self.logger.info(f"Sending MongoDB schema to model: {len(full_schema.get('collections', {}))} collections")
            self.logger.debug(f"MongoDB schema: {full_schema}")
            return full_schema

        # For SQL databases, optimize the schema structure
        optimized_schema = {
            "tables": {},
            "relationships": []
        }

        # Take all tables but optimize the information
        for table_name, table_info in full_schema.get("tables", {}).items():
            optimized_table = {
                "columns": table_info.get("columns", []),
                "column_types": table_info.get("column_types", {}),
                "primary_keys": table_info.get("primary_keys", []),
                "foreign_keys": table_info.get("foreign_keys", [])
            }

            optimized_schema["tables"][table_name] = optimized_table

        # Add key relationships
        optimized_schema["relationships"] = full_schema.get("relationships", [])[:5]  # Top 5 relationships

        self.logger.info(f"Sending optimized SQL schema to model: {len(optimized_schema.get('tables', {}))} tables")

        # Debug logging to see exact schema being sent
        self.logger.debug(f"Full optimized schema: {optimized_schema}")
        for table_name, table_data in optimized_schema.get("tables", {}).items():
            self.logger.debug(f"Table {table_name}: columns={table_data.get('columns', [])}, types={table_data.get('column_types', {})}")

        return optimized_schema

    def _detect_database_type(self) -> str:
        """Detect the current database type from configuration."""
        default_db = self.config.config.default_database
        if default_db and default_db in self.config.config.databases:
            db_config = self.config.config.databases[default_db]
            return db_config.type.lower()
        return "sqlite"  # Default fallback

    async def _execute_sql(self, sql: str, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute SQL query against the database."""
        db_name = database_name or self.config.config.default_database
        if not db_name:
            raise ValueError("No database specified")
        
        return await self.db_manager.execute_query(sql, db_name)
    
    async def _generate_analysis(self, original_query: str, sql: str, data: Optional[pd.DataFrame]) -> str:
        """Generate analysis of the query results."""
        if not self.config.config.show_analysis or data is None:
            return ""
        
        # Basic analysis
        analysis_parts = []
        
        # Query summary
        query_type = sql.strip().upper().split()[0]
        analysis_parts.append(f"Query Type: {query_type}")
        
        # Data summary
        if not data.empty:
            row_count = len(data)
            col_count = len(data.columns)
            analysis_parts.append(f"Results: {row_count} rows, {col_count} columns")
            
            # Column summary
            if col_count <= 5:  # Only for small result sets
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        if data[col].notna().any():
                            mean_val = data[col].mean()
                            analysis_parts.append(f"{col} average: {mean_val:.2f}")
        else:
            analysis_parts.append("Results: No data returned")
        
        return " | ".join(analysis_parts)
    
    def display_results(self, result: QueryResult) -> None:
        """Display query results in a formatted way."""
        # Detect database type for proper labeling
        db_type = self._detect_database_type()
        query_label = "MQL" if db_type == "mongodb" else "SQL"

        # Display query information
        query_panel = Panel(
            f"[green]Original Query:[/green] {result.original_query}\n"
            f"[blue]Generated {query_label}:[/blue] {result.generated_sql}\n"
            f"[yellow]Execution Time:[/yellow] {result.execution_time:.2f}s",
            title="Query Information",
            border_style="blue"
        )
        self.console.print(query_panel)

        # Display query syntax highlighted
        if result.validated_sql:
            syntax_lang = "javascript" if db_type == "mongodb" else "sql"
            syntax_title = f"Final {query_label} Query"
            query_syntax = Syntax(result.validated_sql, syntax_lang, theme="monokai", line_numbers=False)
            syntax_panel = Panel(query_syntax, title=syntax_title, border_style="green")
            self.console.print(syntax_panel)
        
        # Display error if any
        if result.error:
            error_panel = Panel(f"[red]Error: {result.error}[/red]", title="Error", border_style="red")
            self.console.print(error_panel)
            return
        
        # Display data table
        if result.data is not None and not result.data.empty:
            table = Table(title="Query Results")
            
            # Add columns
            for col in result.data.columns:
                table.add_column(str(col), style="cyan")
            
            # Add rows (limit to first 10 for display)
            display_data = result.data.head(10)
            for _, row in display_data.iterrows():
                table.add_row(*[str(val) for val in row])
            
            if len(result.data) > 10:
                table.add_row(*["..." for _ in result.data.columns])
                table.add_row(*[f"({len(result.data)} total rows)" for _ in result.data.columns])
            
            self.console.print(table)
        
        # Display analysis
        if result.analysis:
            analysis_panel = Panel(result.analysis, title="Analysis", border_style="yellow")
            self.console.print(analysis_panel)
    
    async def get_available_databases(self) -> List[str]:
        """Get list of available databases."""
        return list(self.config.config.databases.keys())
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models."""
        models_info = {}
        for model_id, model_config in self.config.SUPPORTED_MODELS.items():
            models_info[model_id] = {
                "name": model_config.name,
                "description": model_config.description,
                "parameters": model_config.parameters,
                "downloaded": self.config.is_model_downloaded(model_id),
                "compatibility": self.config.check_model_compatibility(model_id)
            }
        return models_info

    # MongoDB-specific methods

    def _build_mongodb_schema_string(self, schema_info: Dict[str, Any]) -> str:
        """Build MongoDB schema string from schema info."""
        schema_parts = []

        for collection_name, collection_data in schema_info.get("collections", {}).items():
            fields = collection_data.get("fields", [])
            field_types = collection_data.get("field_types", {})
            doc_count = collection_data.get("document_count", 0)

            # Build field list with types
            field_list = []
            for field in fields:
                field_type = field_types.get(field, "unknown")
                field_list.append(f"{field}: {field_type}")

            collection_info = f"Collection: {collection_name}\n"
            collection_info += f"  Documents: {doc_count}\n"
            collection_info += f"  Fields: {', '.join(field_list)}"

            schema_parts.append(collection_info)

        return "\n\n".join(schema_parts)

    def _get_mongodb_expert_prompt(self, schema_string: str) -> str:
        """Get MongoDB expert prompt for specialized models."""
        return f"""### Instructions:
Your task is to convert a natural language question into a MongoDB query (MQL), given a MongoDB database schema.

Adhere to these rules:
- **Use MongoDB Query Language (MQL)** - use proper MongoDB syntax
- **Return queries in this format**: db.collection.operation({{query}})
- **Common operations**: find(), aggregate(), countDocuments(), distinct()
- **Use proper operators**: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or, $not
- **For aggregation**, use pipeline stages: $match, $group, $sort, $limit, $project, $lookup
- **Field references in aggregation**: Use "$fieldName" syntax
- **Always include the collection name** in your query

### MongoDB Schema:
{schema_string}

### Examples:
Question: "Find all users"
MQL: db.users.find({{}})

Question: "Count documents where age is greater than 25"
MQL: db.users.countDocuments({{age: {{$gt: 25}}}})

Question: "Get average salary by department"
MQL: db.employees.aggregate([
  {{$group: {{_id: "$department", avgSalary: {{$avg: "$salary"}}}}}},
  {{$sort: {{avgSalary: -1}}}}
])

### Input:
Generate a MongoDB query that answers the question `{{user_question}}`.

### Response:
Based on your instructions, here is the MongoDB query I have generated to answer the question `{{user_question}}`:
```javascript"""

    def _get_mongodb_intermediate_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Intermediate MongoDB prompt for medium-sized models."""
        schema_string = self._build_mongodb_schema_string(schema_info)

        # Get first collection for example
        first_collection = list(schema_info.get("collections", {}).keys())[0] if schema_info.get("collections") else "users"

        return f"""You are a specialized Text-to-MongoDB model. Your task is to convert natural language queries into MongoDB queries (MQL).

Follow these steps:
1. Understand what data the user is asking for
2. Identify the relevant collection from the schema
3. Choose the right MongoDB operation (find, aggregate, countDocuments, distinct)
4. Construct the query using proper MQL syntax

MongoDB Schema:
{schema_string}

MongoDB Query Format:
- Simple queries: db.collection.find({{field: value}})
- Aggregations: db.collection.aggregate([{{$match: {{}}}}, {{$group: {{}}}}])
- Counts: db.collection.countDocuments({{field: value}})

Example:
db.{first_collection}.find({{}})

Query: {{user_question}}
MQL:"""

    def _get_mongodb_simple_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Simple MongoDB prompt for smaller models."""
        # Get collections list
        collections = list(schema_info.get("collections", {}).keys())[:5]

        collections_str = ", ".join(collections) if collections else "users, orders"

        return f"""Convert the question to a MongoDB query.

Collections: {collections_str}

Question: {{user_question}}

MongoDB query:"""

    def _get_intelligence_based_mongodb_prompt(self, intelligence_level: str, schema_info: Dict[str, Any]) -> str:
        """Get MongoDB prompt based on model intelligence level."""
        if intelligence_level == "expert":
            schema_string = self._build_mongodb_schema_string(schema_info)
            return self._get_mongodb_expert_prompt(schema_string)
        elif intelligence_level == "advanced":
            return self._get_mongodb_intermediate_prompt(schema_info)
        elif intelligence_level == "simple":
            return self._get_mongodb_simple_prompt(schema_info)
        else:
            return self._get_mongodb_intermediate_prompt(schema_info)