"""Schema analysis and intelligent table/column selection for T2S."""

import re
import asyncio
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import logging

from ..core.config import Config


class SchemaAnalyzer:
    """Analyzes database schemas and selects relevant tables/columns for queries."""
    
    def __init__(self, config: Config):
        """Initialize the schema analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Common keywords that suggest table/column relevance
        self.entity_keywords = {
            'user': ['user', 'customer', 'account', 'person', 'member'],
            'product': ['product', 'item', 'goods', 'merchandise'],
            'order': ['order', 'purchase', 'sale', 'transaction'],
            'payment': ['payment', 'billing', 'invoice', 'charge'],
            'date': ['date', 'time', 'created', 'updated', 'modified'],
            'amount': ['amount', 'price', 'cost', 'total', 'value'],
            'name': ['name', 'title', 'label', 'description'],
            'id': ['id', 'key', 'identifier', 'code'],
            'status': ['status', 'state', 'condition', 'active'],
            'location': ['location', 'address', 'city', 'country', 'region'],
        }
        
        # Common SQL operations and their indicators
        self.operation_keywords = {
            'count': ['count', 'number', 'total', 'how many'],
            'sum': ['sum', 'total', 'amount', 'revenue'],
            'average': ['average', 'avg', 'mean'],
            'max': ['maximum', 'max', 'highest', 'largest'],
            'min': ['minimum', 'min', 'lowest', 'smallest'],
            'group': ['group', 'category', 'type', 'by'],
            'filter': ['where', 'filter', 'with', 'having'],
            'join': ['join', 'relate', 'connect', 'link'],
            'recent': ['recent', 'latest', 'new', 'last'],
            'old': ['old', 'previous', 'past', 'earlier'],
        }
    
    async def get_relevant_schema(self, query: str, db_connection: Any, max_tokens: int) -> Dict[str, Any]:
        """Get relevant schema information based on the query."""
        try:
            # Get full schema first
            from ..database.db_manager import DatabaseManager
            db_manager = DatabaseManager(self.config)
            
            # Extract database name from connection - this is a bit hacky but works
            db_name = None
            for name, engine in db_manager.engines.items():
                if engine == db_connection:
                    db_name = name
                    break
            
            if not db_name:
                # Fallback: use default database
                db_name = self.config.config.default_database
            
            if not db_name:
                raise ValueError("Could not determine database name")
            
            full_schema = await db_manager.get_schema_info(db_name)
            
            # Analyze query to find relevant tables and columns
            relevant_tables = self._find_relevant_tables(query, full_schema)
            
            # Build optimized schema with only relevant information
            optimized_schema = self._build_optimized_schema(
                relevant_tables, full_schema, max_tokens
            )
            
            return optimized_schema
            
        except Exception as e:
            self.logger.error(f"Error analyzing schema: {e}")
            # Fallback: return a minimal schema structure
            return {"tables": {}, "relationships": []}
    
    def _find_relevant_tables(self, query: str, full_schema: Dict[str, Any]) -> List[str]:
        """Find tables that are likely relevant to the query."""
        query_lower = query.lower()
        relevant_tables = []
        table_scores = defaultdict(float)
        
        # Score tables based on various factors
        for table_name, table_info in full_schema.get("tables", {}).items():
            table_name_lower = table_name.lower()
            
            # Direct table name mentions (highest score)
            if table_name_lower in query_lower:
                table_scores[table_name] += 10.0
            
            # Partial table name matches
            for word in query_lower.split():
                if word in table_name_lower or table_name_lower in word:
                    table_scores[table_name] += 5.0
            
            # Column name relevance
            for column_name in table_info.get("columns", []):
                column_lower = column_name.lower()
                
                # Direct column mentions
                if column_lower in query_lower:
                    table_scores[table_name] += 3.0
                
                # Keyword matching
                for entity_type, keywords in self.entity_keywords.items():
                    for keyword in keywords:
                        if keyword in query_lower and keyword in column_lower:
                            table_scores[table_name] += 2.0
            
            # Semantic relevance based on entity keywords
            for entity_type, keywords in self.entity_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        # Check if table name suggests this entity type
                        if keyword in table_name_lower:
                            table_scores[table_name] += 1.5
                        
                        # Check if table has columns related to this entity
                        for column_name in table_info.get("columns", []):
                            if keyword in column_name.lower():
                                table_scores[table_name] += 1.0
        
        # Sort tables by relevance score
        sorted_tables = sorted(
            table_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top scored tables, but ensure we have at least one
        relevant_tables = [table for table, score in sorted_tables if score > 0]
        
        if not relevant_tables and full_schema.get("tables"):
            # Fallback: if no tables scored, take the first few tables
            relevant_tables = list(full_schema["tables"].keys())[:3]
        
        # Limit number of tables to avoid token overflow
        return relevant_tables[:5]
    
    def _build_optimized_schema(
        self, 
        relevant_tables: List[str], 
        full_schema: Dict[str, Any], 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Build an optimized schema with only relevant information."""
        optimized = {
            "tables": {},
            "relationships": []
        }
        
        # Estimate tokens (rough approximation)
        current_tokens = 0
        token_per_char = 0.25  # Rough estimate
        
        # Add tables in order of relevance
        for table_name in relevant_tables:
            if table_name not in full_schema.get("tables", {}):
                continue
            
            table_info = full_schema["tables"][table_name]
            
            # Build table info
            optimized_table = {
                "columns": table_info.get("columns", []),
                "column_types": table_info.get("column_types", {}),
                "primary_keys": table_info.get("primary_keys", []),
                "foreign_keys": table_info.get("foreign_keys", [])
            }
            
            # Estimate tokens for this table
            table_text = f"Table: {table_name}\nColumns: {', '.join(optimized_table['columns'])}\n"
            table_tokens = len(table_text) * token_per_char
            
            if current_tokens + table_tokens < max_tokens:
                optimized["tables"][table_name] = optimized_table
                current_tokens += table_tokens
            else:
                break  # Stop if we're approaching token limit
        
        # Add relevant relationships
        for relationship in full_schema.get("relationships", []):
            # Check if relationship involves any of our selected tables
            rel_tables = self._extract_tables_from_relationship(relationship)
            if any(table in optimized["tables"] for table in rel_tables):
                optimized["relationships"].append(relationship)
        
        return optimized
    
    def _extract_tables_from_relationship(self, relationship: str) -> List[str]:
        """Extract table names from a relationship string."""
        # Relationship format: "table1(col) -> table2(col)"
        tables = []
        
        # Simple regex to extract table names
        table_pattern = r'(\w+)\('
        matches = re.findall(table_pattern, relationship)
        
        return matches
    
    def _estimate_query_complexity(self, query: str) -> str:
        """Estimate the complexity of the query to adjust schema detail level."""
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_score = 0
        
        # JOIN indicators
        join_keywords = ['join', 'relate', 'connect', 'link', 'combine']
        for keyword in join_keywords:
            if keyword in query_lower:
                complexity_score += 2
        
        # Aggregation indicators
        agg_keywords = ['count', 'sum', 'average', 'max', 'min', 'group']
        for keyword in agg_keywords:
            if keyword in query_lower:
                complexity_score += 1
        
        # Filtering indicators
        filter_keywords = ['where', 'filter', 'with', 'having', 'condition']
        for keyword in filter_keywords:
            if keyword in query_lower:
                complexity_score += 1
        
        # Multiple table indicators
        if len(re.findall(r'\b\w+\s+and\s+\w+\b', query_lower)) > 0:
            complexity_score += 1
        
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def get_column_suggestions(self, query: str, table_columns: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Suggest the most relevant columns for each table based on the query."""
        query_lower = query.lower()
        suggestions = {}
        
        for table_name, columns in table_columns.items():
            column_scores = defaultdict(float)
            
            for column in columns:
                column_lower = column.lower()
                
                # Direct column mention
                if column_lower in query_lower:
                    column_scores[column] += 10.0
                
                # Keyword matching
                for entity_type, keywords in self.entity_keywords.items():
                    for keyword in keywords:
                        if keyword in query_lower and keyword in column_lower:
                            column_scores[column] += 3.0
                
                # Operation-based relevance
                for operation, keywords in self.operation_keywords.items():
                    for keyword in keywords:
                        if keyword in query_lower:
                            # Certain columns are more relevant for certain operations
                            if operation in ['sum', 'average', 'max', 'min'] and any(
                                numeric_hint in column_lower 
                                for numeric_hint in ['amount', 'price', 'cost', 'value', 'count']
                            ):
                                column_scores[column] += 2.0
                            elif operation == 'count':
                                column_scores[column] += 1.0
            
            # Sort columns by relevance
            sorted_columns = sorted(
                column_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top columns or all if few
            relevant_columns = [col for col, score in sorted_columns if score > 0]
            if not relevant_columns:
                relevant_columns = columns[:10]  # Fallback to first 10 columns
            
            suggestions[table_name] = relevant_columns[:10]  # Limit to 10 columns per table
        
        return suggestions 