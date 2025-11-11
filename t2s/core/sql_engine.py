"""SQL Engine for Text-to-SQL conversion."""

import re
import logging
from typing import Dict, Any, Optional

import sqlparse
from rich.console import Console

from .config import Config
from ..models.model_manager import ModelManager
from ..utils.sql_validator import SQLValidator


class SQLEngine:
    """Handles all SQL-specific logic for query generation and validation."""

    def __init__(self, config: Config, model_manager: ModelManager):
        """Initialize SQL engine."""
        self.config = config
        self.model_manager = model_manager
        self.sql_validator = SQLValidator()
        self.console = Console()
        self.logger = logging.getLogger(__name__)

    async def generate_sql(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query from natural language."""
        # Create system prompt based on model intelligence
        system_prompt = self._create_system_prompt(schema_info)
        user_prompt = natural_query

        # Log prompts for debugging
        self.logger.debug(f"System prompt: {system_prompt[:500]}...")
        self.logger.debug(f"User prompt: {user_prompt}")

        # Generate using AI model
        generated_text = await self.model_manager.generate_sql(system_prompt, user_prompt)

        # Extract SQL query
        sql_query = self.extract_sql(generated_text)

        if not sql_query:
            raise ValueError("Failed to extract SQL query from model output")

        self.logger.info(f"Generated SQL: {sql_query}")
        return sql_query

    async def validate_sql(self, sql: str) -> str:
        """Validate and potentially correct SQL query."""
        if not self.config.config.enable_query_validation:
            return sql

        return await self.sql_validator.validate_and_correct(sql)

    def extract_sql(self, generated_text: str) -> str:
        """Extract SQL query from generated text."""
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

        # Split text into lines and process line by line
        lines = text.split('\n')
        sql_statement = ""
        collecting_sql = False

        for line in lines:
            line_stripped = line.strip()

            # Start collecting when we find a SQL keyword
            if re.match(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\s+', line_stripped, re.IGNORECASE):
                if not collecting_sql:
                    collecting_sql = True
                    sql_statement = line_stripped
                    if line_stripped.endswith(';'):
                        break
                else:
                    # Hit another SQL statement while collecting, stop here
                    break
            elif collecting_sql:
                # We're collecting a multi-line SQL statement
                if line_stripped:
                    # Skip lines that look like explanatory text
                    if re.search(r'(what|list|show|find|get|query:|sql:|answer:|result:)', line_stripped, re.IGNORECASE):
                        break

                    sql_statement += " " + line_stripped

                    if line_stripped.endswith(';'):
                        break
                else:
                    # Empty line while collecting
                    if sql_statement and not sql_statement.endswith(';'):
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
            if self._is_valid_sql_structure(sql_statement):
                return sql_statement.strip()

        # Fallback: try regex patterns
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
                if self._is_valid_sql_structure(sql):
                    if not sql.endswith(';'):
                        sql += ';'
                    return sql

        return ""

    def _is_valid_sql_structure(self, sql: str) -> bool:
        """Basic SQL structure validation."""
        if not sql:
            return False

        sql_upper = sql.upper().strip()
        valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "SHOW", "DESCRIBE"]
        return any(sql_upper.startswith(start) for start in valid_starts)

    def _create_system_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Create system prompt optimized for the current model's intelligence level."""
        current_model_id = self.config.config.selected_model

        if not current_model_id or current_model_id not in self.config.SUPPORTED_MODELS:
            return self._get_sqlcoder_prompt(schema_info)

        model_config = self.config.SUPPORTED_MODELS[current_model_id]
        intelligence_level = self._determine_model_intelligence(current_model_id, model_config)

        self.logger.info(f"Model '{current_model_id}' classified as '{intelligence_level}' intelligence level")

        return self._get_intelligence_based_prompt(intelligence_level, schema_info)

    def _determine_model_intelligence(self, model_id: str, model_config) -> str:
        """Determine the intelligence level of a model."""
        if "sqlcoder" in model_id.lower():
            return "expert"

        if any(model_name in model_id.lower() for model_name in ["llama", "phi", "mistral", "qwen"]):
            return "advanced"

        if "smollm" in model_id.lower():
            return "simple"

        return "intermediate"

    def _get_intelligence_based_prompt(self, intelligence_level: str, schema_info: Dict[str, Any]) -> str:
        """Get system prompt based on model intelligence level."""
        if intelligence_level == "expert":
            return self._get_sqlcoder_prompt(schema_info)
        elif intelligence_level == "advanced":
            return self._get_advanced_prompt(schema_info)
        elif intelligence_level == "simple":
            return self._get_simple_prompt(schema_info)
        else:
            return self._get_intermediate_prompt(schema_info)

    def _get_sqlcoder_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Get the SQLCoder prompt for expert models."""
        table_metadata_string = self._build_table_metadata_string(schema_info)

        return f"""### Instructions:
Your task is to convert a natural language question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{{user_question}}`.

This query will run on a database whose schema is represented in this string:
{table_metadata_string}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{{user_question}}`:
```sql"""

    def _get_intermediate_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Intermediate prompt for medium models."""
        simplified_schema = self._build_simplified_schema_string(schema_info)

        return f"""You are a specialized Text-to-SQL model. Your task is to convert natural language queries into SQL queries.

Follow these steps:
1. Understand what data the user is asking for
2. Identify the relevant tables from the schema
3. Determine what columns are needed
4. Construct the SQL query using proper JOIN statements if multiple tables are needed

Database Schema:
{simplified_schema}

SQL Query Format:
- Use proper JOIN syntax when accessing multiple tables
- Always use table aliases to prevent ambiguity
- Use WHERE clauses for filtering
- Use aggregate functions (COUNT, SUM, AVG) when needed
- Use GROUP BY when using aggregate functions on grouped data

Query: {{user_question}}
SQL:"""

    def _get_advanced_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Advanced prompt for capable models like Llama."""
        simplified_schema = self._build_simplified_schema_string(schema_info)

        return f"""You are an expert at converting natural language to SQL queries.

Database Schema:
{simplified_schema}

Rules:
- Analyze the question carefully to identify required tables and columns
- Use appropriate JOIN types (INNER, LEFT, etc.) based on the relationships
- Apply WHERE filters, GROUP BY for aggregations, ORDER BY for sorting
- Use table aliases to avoid ambiguity
- Return only the SQL query without explanations

Question: {{user_question}}
SQL Query:"""

    def _get_simple_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Simple prompt for smaller models."""
        # Get first few tables for example
        tables = list(schema_info.get("tables", {}).keys())[:3]
        tables_str = ", ".join(tables) if tables else "users, orders"

        return f"""Convert the question to a SQL query.

Tables: {tables_str}

Question: {{user_question}}

SQL query:"""

    def _build_table_metadata_string(self, schema_info: Dict[str, Any]) -> str:
        """Build table metadata string in CREATE TABLE format."""
        tables_info = []

        for table_name, table_data in schema_info.get("tables", {}).items():
            columns = table_data.get("columns", [])
            column_types = table_data.get("column_types", {})
            primary_keys = table_data.get("primary_keys", [])
            foreign_keys = table_data.get("foreign_keys", [])

            # Create column definitions
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
        """Build simplified schema string with only essential information."""
        schema_parts = []

        for table_name, table_data in schema_info.get("tables", {}).items():
            columns = table_data.get("columns", [])
            column_types = table_data.get("column_types", {})
            primary_keys = table_data.get("primary_keys", [])
            foreign_keys = table_data.get("foreign_keys", [])

            # Build foreign key map
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
            schema_parts.append("")
            schema_parts.append("Relationships:")
            schema_parts.extend(relationships)

        return "\n".join(schema_parts)
