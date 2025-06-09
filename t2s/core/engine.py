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
        self.sql_validator = SQLValidator()
        
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
        
        # Initialize model manager with loading animation
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Initializing AI model[/cyan]"),
            console=self.console,
            transient=False
        ) as progress:
            task = progress.add_task("", total=None)
            await self.model_manager.initialize()
        
        # Initialize database connections
        await self.db_manager.initialize()
        
        self.console.print("[green]T2S Engine initialized successfully![/green]")
    
    async def process_query(self, natural_language_query: str, database_name: Optional[str] = None) -> QueryResult:
        """Process a natural language query and return results."""
        start_time = datetime.now()
        
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
            
            # Step 2: Generate SQL using AI model
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Generating SQL query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                generated_sql = await self._generate_sql(natural_language_query, schema_info)
            
            # Step 3: Validate and correct SQL
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Validating SQL query[/cyan]"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)
                validated_sql = await self._validate_and_correct_sql(generated_sql)
            
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
            self.logger.error(f"Error processing query: {e}")
            
            return QueryResult(
                original_query=natural_language_query,
                generated_sql="",
                validated_sql="",
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
        
        # Optimize schema for the model - keep it comprehensive but concise
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
        
        self.logger.info(f"Sending optimized schema to model: {len(optimized_schema.get('tables', {}))} tables")
        
        # Debug logging to see exact schema being sent
        self.logger.debug(f"Full optimized schema: {optimized_schema}")
        for table_name, table_data in optimized_schema.get("tables", {}).items():
            self.logger.debug(f"Table {table_name}: columns={table_data.get('columns', [])}, types={table_data.get('column_types', {})}")
        
        return optimized_schema
    
    async def _generate_sql(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query using ONLY the AI model - no fallbacks, no repairs."""
        # Create a structured prompt with full schema information
        system_prompt = self._create_system_prompt(schema_info)
        user_prompt = natural_query
        
        # Debug logging to see exact prompts being sent
        self.logger.debug(f"System prompt being sent to model: {system_prompt[:500]}...")
        self.logger.debug(f"User prompt: {user_prompt}")
        
        try:
            # Generate using AI model - this is the ONLY attempt
            self.logger.info("Attempting AI model generation (NO FALLBACKS)")
            generated_text = await self.model_manager.generate_sql(system_prompt, user_prompt)
            
            # Extract SQL query using basic regex - NO REPAIRS
            sql_query = self._extract_sql_query(generated_text)
            
            if not sql_query or sql_query.strip() == "":
                raise ValueError("No SQL query could be extracted from model output")
            
            self.logger.info(f"✅ AI model successfully generated valid SQL: {sql_query}")
            return sql_query
        
        except Exception as e:
            # NO FALLBACKS - throw the error
            error_msg = f"AI model failed to generate valid SQL: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _extract_sql_query(self, generated_text: str) -> str:
        """Basic SQL extraction with minimal processing - NO REPAIRS."""
        if not generated_text:
            return ""
        
        text = generated_text.strip()
        
        # First, try to find SQL in code blocks
        code_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            sql_content = code_block_match.group(1).strip()
            # If there are multiple statements in code block, take only the first
            statements = sql_content.split(';')
            if statements and statements[0].strip():
                return statements[0].strip() + ';'
        
        # Split text into lines and process line by line to find the first SQL statement
        lines = text.split('\n')
        sql_statement = ""
        collecting_sql = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Start collecting when we find a SQL keyword at the beginning of a line
            if re.match(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\s+', line_stripped, re.IGNORECASE):
                if not collecting_sql:  # Only start if we're not already collecting
                    collecting_sql = True
                    sql_statement = line_stripped
                    # If this line ends with semicolon, we're done
                    if line_stripped.endswith(';'):
                        break
                else:
                    # We hit another SQL statement while collecting, stop here
                    break
            elif collecting_sql:
                # We're collecting a multi-line SQL statement
                if line_stripped:
                    # Skip lines that look like explanatory text or other queries
                    if re.search(r'(what|list|show|find|get|query:|sql:|answer:|result:)', line_stripped, re.IGNORECASE):
                        break  # Stop collecting if we hit explanatory text
                    
                    sql_statement += " " + line_stripped
                    
                    # If this line ends with semicolon, we're done
                    if line_stripped.endswith(';'):
                        break
                else:
                    # Empty line while collecting - might indicate end of statement
                    if sql_statement and not sql_statement.endswith(';'):
                        # Add semicolon if missing and stop
                        sql_statement += ';'
                    break
        
        # Clean up the extracted SQL
        if sql_statement:
            # Remove any trailing explanatory text
            sql_statement = re.sub(r'\s+(what|list|show|find|get|query|sql|answer|result).*$', '', sql_statement, flags=re.IGNORECASE)
            # Ensure it ends with semicolon
            if not sql_statement.endswith(';'):
                sql_statement += ';'
            
            # Validate it looks like SQL
            if self._is_basic_sql_structure(sql_statement):
                return sql_statement.strip()
        
        # Ultimate fallback: try regex but only take content before any explanatory keywords
        sql_patterns = [
            r'(SELECT[^;]*?)(?:\s+(?:what|list|show|find|get|query|sql|answer|result)|;|$)',
            r'(INSERT[^;]*?)(?:\s+(?:what|list|show|find|get|query|sql|answer|result)|;|$)',
            r'(UPDATE[^;]*?)(?:\s+(?:what|list|show|find|get|query|sql|answer|result)|;|$)',
            r'(DELETE[^;]*?)(?:\s+(?:what|list|show|find|get|query|sql|answer|result)|;|$)',
            r'(CREATE[^;]*?)(?:\s+(?:what|list|show|find|get|query|sql|answer|result)|;|$)',
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                if self._is_basic_sql_structure(sql):
                    if not sql.endswith(';'):
                        sql += ';'
                    return sql
        
        # If nothing worked, return empty string
        return ""
    
    def _is_basic_sql_structure(self, sql: str) -> bool:
        """Basic SQL structure validation - minimal checks only."""
        if not sql:
            return False
        
        sql_upper = sql.upper().strip()
        
        # Must start with a valid SQL command
        valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "SHOW", "DESCRIBE"]
        return any(sql_upper.startswith(start) for start in valid_starts)
    
    def _create_system_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Create system prompt optimized for the current model's intelligence level."""
        
        # Get current model information
        current_model_id = self.config.config.selected_model
        if not current_model_id or current_model_id not in self.config.SUPPORTED_MODELS:
            # Fallback to SQLCoder format for unknown models
            return self._get_sqlcoder_prompt(schema_info)
        
        model_config = self.config.SUPPORTED_MODELS[current_model_id]
        
        # Determine model intelligence level based on parameters and specialization
        intelligence_level = self._determine_model_intelligence(current_model_id, model_config)
        
        # Create prompt based on intelligence level
        return self._get_intelligence_based_prompt(intelligence_level, schema_info, current_model_id)
    
    def _determine_model_intelligence(self, model_id: str, model_config) -> str:
        """Determine the intelligence level of a model based on its characteristics."""
        
        # SQLCoder is specialized for SQL - treat as expert
        if "sqlcoder" in model_id.lower():
            return "expert"
        
        # Llama/Mistral/Qwen models are generally quite capable - treat as advanced
        if any(model_name in model_id.lower() for model_name in ["llama", "phi", "mistral", "qwen"]):
            return "advanced"
        
        # SmolLM needs simpler prompts
        if "smollm" in model_id.lower():
            return "simple"
            
        # Everything else uses intermediate level  
        return "intermediate"
    
    def _get_intelligence_based_prompt(self, intelligence_level: str, schema_info: Dict[str, Any], model_id: str) -> str:
        """Get system prompt based on model intelligence level."""
        
        # Build table metadata string
        table_metadata_string = self._build_table_metadata_string(schema_info)
        db_type = self._detect_database_type()
        
        # Log the intelligence level for debugging
        self.logger.info(f"🧠 Model '{model_id}' classified as '{intelligence_level}' intelligence level")
        
        if intelligence_level == "expert":
            # SQLCoder prompt for expert models
            return self._get_sqlcoder_prompt(schema_info)
        elif intelligence_level == "advanced":
            # Advanced prompt for capable models like Llama
            return self._get_advanced_prompt(db_type, schema_info)
        elif intelligence_level == "simple":
            # Simple prompt for smaller models like SmolLM
            return self._get_simple_prompt(db_type, schema_info)
        else:
            # Intermediate prompt for all other models
            return self._get_intermediate_prompt(db_type, schema_info)
    
    def _build_table_metadata_string(self, schema_info: Dict[str, Any]) -> str:
        """Build table metadata string in a consistent format."""
        tables_info = []
        
        for table_name, table_data in schema_info.get("tables", {}).items():
            columns = table_data.get("columns", [])
            column_types = table_data.get("column_types", {})
            primary_keys = table_data.get("primary_keys", [])
            foreign_keys = table_data.get("foreign_keys", [])
            
            # Create column definitions with types and constraints
            column_defs = []
            for col in columns:
                col_type = column_types.get(col, "TEXT")
                pk_marker = " PRIMARY KEY" if col in primary_keys else ""
                column_defs.append(f"  {col} {col_type}{pk_marker}")
            
            # Add foreign key comments
            fk_comments = []
            for fk in foreign_keys:
                if isinstance(fk, dict) and "constrained_columns" in fk and "referred_table" in fk:
                    fk_col = fk["constrained_columns"][0] if fk["constrained_columns"] else "unknown"
                    ref_table = fk["referred_table"]
                    ref_col = fk.get("referred_columns", ["id"])[0]
                    fk_comments.append(f"-- {fk_col} references {ref_table}({ref_col})")
            
            table_def = f"CREATE TABLE {table_name} (\n" + ",\n".join(column_defs) + "\n);"
            if fk_comments:
                table_def += "\n" + "\n".join(fk_comments)
            
            tables_info.append(table_def)
        
        return "\n\n".join(tables_info)

    def _build_simplified_schema_string(self, schema_info: Dict[str, Any]) -> str:
        """Build simplified schema string with only essential information - no sample data."""
        schema_parts = []
        
        # Tables with columns and types, including explicit FK notation
        for table_name, table_data in schema_info.get("tables", {}).items():
            columns = table_data.get("columns", [])
            column_types = table_data.get("column_types", {})
            primary_keys = table_data.get("primary_keys", [])
            foreign_keys = table_data.get("foreign_keys", [])
            
            # Build foreign key map for this table
            fk_map = {}
            for fk in foreign_keys:
                if isinstance(fk, dict) and "constrained_columns" in fk and "referred_table" in fk:
                    fk_col = fk["constrained_columns"][0] if fk["constrained_columns"] else "unknown"
                    ref_table = fk["referred_table"]
                    ref_col = fk.get("referred_columns", ["id"])[0]
                    fk_map[fk_col] = f"{ref_table}.{ref_col}"
            
            # Simple column list with types and FK notation
            column_list = []
            for col in columns:
                col_type = column_types.get(col, "TEXT")
                if col in primary_keys:
                    column_list.append(f"{col}: {col_type} (PK)")
                elif col in fk_map:
                    column_list.append(f"{col}: {col_type} (FK to {fk_map[col]})")
                else:
                    column_list.append(f"{col}: {col_type}")
            
            table_info = f"{table_name}: {', '.join(column_list)}"
            schema_parts.append(table_info)
        
        # Add relationships section
        relationships = []
        for table_name, table_data in schema_info.get("tables", {}).items():
            for fk in table_data.get("foreign_keys", []):
                if isinstance(fk, dict) and "constrained_columns" in fk and "referred_table" in fk:
                    fk_col = fk["constrained_columns"][0] if fk["constrained_columns"] else "unknown"
                    ref_table = fk["referred_table"]
                    ref_col = fk.get("referred_columns", ["id"])[0]
                    relationships.append(f"{table_name}.{fk_col} -> {ref_table}.{ref_col}")
        
        if relationships:
            schema_parts.append("")  # Add blank line before relationships
            schema_parts.append("Relationships:")
            schema_parts.extend(relationships)
        
        return "\n".join(schema_parts)
    
    def _get_sqlcoder_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Get the original SQLCoder prompt - for expert models."""
        db_type = self._detect_database_type()
        table_metadata_string = self._build_table_metadata_string(schema_info)
        return self._get_database_specific_prompt(db_type, table_metadata_string)
    
    def _get_intermediate_prompt(self, db_type: str, schema_info: Dict[str, Any]) -> str:
        """Intermediate prompt for medium models (4B) - detailed yet concise, guiding them through SQL generation steps."""
        
        # Use simplified schema with explicit FK notation
        simplified_schema = self._build_simplified_schema_string(schema_info)
        
        # Get the first table for example
        first_table = list(schema_info.get("tables", {}).keys())[0] if schema_info.get("tables") else "customers"
        
        return f"""You are a specialized Text-to-SQL model. Your primary task is to translate the natural language 'Query' into an accurate SQL statement using the provided 'Tables' schema.

Follow these steps:
1. Carefully analyze the 'Query' to understand the information being requested.
2. Examine the 'Tables' schema to identify the relevant tables and columns.
3. Construct a single, syntactically correct SQL query that retrieves the requested information.
4. Refer to the 'Example' for the general format of the expected SQL output.
5. Respond ONLY with the generated SQL query. Do not add any explanations, comments, or any text other than the SQL itself.

Tables:
{simplified_schema}

Example:
SELECT * FROM {first_table};

Query: {{user_question}}
SQL:"""
    
    def _get_advanced_prompt(self, db_type: str, schema_info: Dict[str, Any]) -> str:
        """Advanced prompt for capable models like Llama - more contextual and flexible."""
        
        # Use simplified schema but with more context
        simplified_schema = self._build_simplified_schema_string(schema_info)
        db_rules = self._get_db_specific_rules(db_type)
        
        # Get the first table for example
        first_table = list(schema_info.get("tables", {}).keys())[0] if schema_info.get("tables") else "customers"
        
        return f"""You are an expert SQL developer. Your task is to convert natural language questions into accurate, efficient SQL queries.

{db_rules}

DATABASE SCHEMA:
{simplified_schema}

INSTRUCTIONS:
1. Analyze the user's question carefully to understand what data they want
2. Use the provided schema to identify relevant tables and columns
3. Write clean, efficient SQL that follows {db_type.upper()} syntax
4. Use appropriate JOINs when data spans multiple tables
5. Include proper WHERE clauses for filtering
6. Use ORDER BY and LIMIT when appropriate for the question
7. Return ONLY the SQL query, no explanations or additional text

EXAMPLE:
Question: "Show me all customers"
SQL: SELECT * FROM {first_table};

Now convert this question to SQL:
{{user_question}}

SQL:"""
    
    def _get_simple_prompt(self, db_type: str, schema_info: Dict[str, Any]) -> str:
        """Simple prompt for smaller models like SmolLM - very direct and minimal."""
        
        # Get the first few tables only for simplicity
        tables = list(schema_info.get("tables", {}).items())[:5]
        
        # Build a very simple schema string
        schema_lines = []
        for table_name, table_data in tables:
            columns = table_data.get("columns", [])
            schema_lines.append(f"Table {table_name}: {', '.join(columns[:10])}")  # Limit columns too
        
        simple_schema = "\n".join(schema_lines)
        
        return f"""Convert the question to a SQL query.

Database tables:
{simple_schema}

Question: {{user_question}}

SQL query:"""

    def _get_db_specific_rules(self, db_type: str) -> str:
        """Get database-specific rules for advanced prompts."""
        
        if db_type == "sqlite":
            return """DATABASE-SPECIFIC RULES:
- Use SQLite functions: date('now'), datetime(), COALESCE, IFNULL
- Use CAST(column AS FLOAT) for ratios
- Use double quotes for identifiers with spaces
- Use LIMIT n for row limits
- Query system tables: sqlite_master"""
        
        elif db_type == "postgresql":
            return """DATABASE-SPECIFIC RULES:
- Use PostgreSQL functions: NOW(), CURRENT_DATE, STRING_AGG, ARRAY_AGG
- Use column::FLOAT for type casting
- Use LIMIT n OFFSET m for pagination
- Query system: information_schema.tables, information_schema.columns
- Use double quotes for identifiers, single quotes for strings"""
        
        elif db_type == "mysql":
            return """DATABASE-SPECIFIC RULES:
- Use MySQL functions: NOW(), CURDATE(), GROUP_CONCAT
- Use CAST(column AS DECIMAL) for precise numbers
- Use LIMIT offset, count for pagination
- Query system: information_schema.tables, information_schema.columns
- Use backticks for identifiers, single quotes for strings"""
        
        else:
            return """DATABASE-SPECIFIC RULES:
- Use standard SQL functions: COALESCE, LENGTH, SUBSTRING
- Avoid database-specific features
- Use proper data type casting for calculations"""
    
    def _detect_database_type(self) -> str:
        """Detect the current database type from configuration."""
        default_db = self.config.config.default_database
        if default_db and default_db in self.config.config.databases:
            db_config = self.config.config.databases[default_db]
            return db_config.type.lower()
        return "sqlite"  # Default fallback
    
    def _get_database_specific_prompt(self, db_type: str, table_metadata_string: str) -> str:
        """Get database-specific system prompt for SQLCoder."""
        
        if db_type == "sqlite":
            return f"""### Instructions:
Your task is to convert a question into a SQL query, given a SQLite database schema.
Adhere to these rules:
- **Use SQLite syntax only** - no PostgreSQL or MySQL specific features
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id`
- **For LIMIT queries**, use SQLite syntax: `LIMIT n`
- **For date/time**, use SQLite functions like `date('now')`, `datetime('now')`
- **For schema queries**, use `sqlite_master` table: `SELECT name FROM sqlite_master WHERE type='table'`
- When creating ratios, always cast the numerator as float: `CAST(column AS FLOAT)`
- **Use double quotes for identifiers** if they contain spaces or special characters
- **Common SQLite functions**: COALESCE, IFNULL, LENGTH, SUBSTR, REPLACE, ROUND

### Input:
Generate a SQL query that answers the question `{{user_question}}`.
This query will run on a SQLite database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{{user_question}}`:
```sql"""
        
        elif db_type == "postgresql":
            return f"""### Instructions:
Your task is to convert a question into a SQL query, given a PostgreSQL database schema.
Adhere to these rules:
- **Use PostgreSQL syntax** - leverage PostgreSQL-specific features when beneficial
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id`
- **For LIMIT queries**, use PostgreSQL syntax: `LIMIT n` or `LIMIT n OFFSET m`
- **For date/time**, use PostgreSQL functions like `NOW()`, `CURRENT_DATE`, `CURRENT_TIMESTAMP`
- **For schema queries**, use `information_schema.tables` and `information_schema.columns`
- When creating ratios, always cast the numerator as float: `column::FLOAT`
- **Use double quotes for identifiers** and single quotes for strings
- **PostgreSQL functions**: COALESCE, NULLIF, LENGTH, SUBSTRING, REPLACE, ROUND, STRING_AGG, ARRAY_AGG

### Input:
Generate a SQL query that answers the question `{{user_question}}`.
This query will run on a PostgreSQL database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{{user_question}}`:
```sql"""
        
        elif db_type == "mysql":
            return f"""### Instructions:
Your task is to convert a question into a SQL query, given a MySQL database schema.
Adhere to these rules:
- **Use MySQL syntax** - leverage MySQL-specific features when beneficial
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id`
- **For LIMIT queries**, use MySQL syntax: `LIMIT n` or `LIMIT offset, count`
- **For date/time**, use MySQL functions like `NOW()`, `CURDATE()`, `CURTIME()`
- **For schema queries**, use `information_schema.tables` and `information_schema.columns`
- When creating ratios, always cast the numerator as float: `CAST(column AS DECIMAL)`
- **Use backticks for identifiers** and single quotes for strings
- **MySQL functions**: COALESCE, IFNULL, LENGTH, SUBSTRING, REPLACE, ROUND, GROUP_CONCAT

### Input:
Generate a SQL query that answers the question `{{user_question}}`.
This query will run on a MySQL database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{{user_question}}`:
```sql"""
        
        else:
            # Fallback to generic SQL
            return f"""### Instructions:
Your task is to convert a question into a SQL query, given a database schema.
Adhere to these rules:
- **Use standard SQL syntax** - avoid database-specific features
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT c.name, o.total FROM customers c JOIN orders o ON c.id = o.customer_id`
- When creating ratios, always cast the numerator as float
- **Use standard SQL functions**: COALESCE, LENGTH, SUBSTRING, REPLACE, ROUND

### Input:
Generate a SQL query that answers the question `{{user_question}}`.
This query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{{user_question}}`:
```sql"""
    
    async def _validate_and_correct_sql(self, sql: str) -> str:
        """Validate and correct SQL query."""
        if not self.config.config.enable_query_validation:
            return sql
        
        return await self.sql_validator.validate_and_correct(sql)
    
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
        # Display query information
        query_panel = Panel(
            f"[green]Original Query:[/green] {result.original_query}\n"
            f"[blue]Generated SQL:[/blue] {result.generated_sql}\n"
            f"[yellow]Execution Time:[/yellow] {result.execution_time:.2f}s",
            title="Query Information",
            border_style="blue"
        )
        self.console.print(query_panel)
        
        # Display SQL syntax highlighted
        if result.validated_sql:
            sql_syntax = Syntax(result.validated_sql, "sql", theme="monokai", line_numbers=False)
            sql_panel = Panel(sql_syntax, title="Final SQL Query", border_style="green")
            self.console.print(sql_panel)
        
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