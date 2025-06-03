"""SQL validation and correction utilities for T2S."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging

import sqlparse
from sqlparse import sql, tokens as T


class SQLValidator:
    """Validates and corrects SQL queries."""
    
    def __init__(self):
        """Initialize the SQL validator."""
        self.logger = logging.getLogger(__name__)
        
        # Common SQL syntax errors and their fixes
        self.common_fixes = {
            # Missing semicolon
            r'(?<!;)\s*$': ';',
            
            # Double quotes instead of single quotes for strings
            r'"([^"]*)"(?=\s*(=|!=|<>|LIKE|IN))': r"'\1'",
            
            # Missing spaces around operators
            r'(\w)=(\w)': r'\1 = \2',
            r'(\w)!=(\w)': r'\1 != \2',
            r'(\w)<>(\w)': r'\1 <> \2',
            r'(\w)<(\w)': r'\1 < \2',
            r'(\w)>(\w)': r'\1 > \2',
            
            # Extra spaces in keywords
            r'\bSELEC T\b': 'SELECT',
            r'\bFRO M\b': 'FROM',
            r'\bWHER E\b': 'WHERE',
            r'\bORDER  BY\b': 'ORDER BY',
            r'\bGROUP  BY\b': 'GROUP BY',
            
            # Common misspellings
            r'\bSELECT\s+\*\s+FORM\b': 'SELECT * FROM',
            r'\bWHEAR\b': 'WHERE',
            r'\bORDRE\b': 'ORDER',
        }
        
        # Reserved words that need proper casing
        self.reserved_words = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'full',
            'outer', 'on', 'and', 'or', 'not', 'in', 'like', 'between', 'is',
            'null', 'order', 'by', 'group', 'having', 'count', 'sum', 'avg',
            'max', 'min', 'distinct', 'as', 'case', 'when', 'then', 'else',
            'end', 'insert', 'into', 'values', 'update', 'set', 'delete',
            'create', 'table', 'alter', 'drop', 'index', 'primary', 'key',
            'foreign', 'references', 'unique', 'constraint', 'check', 'default',
            'auto_increment', 'limit', 'offset', 'union', 'all', 'exists',
            'desc', 'asc'
        }
    
    async def validate_and_correct(self, sql: str) -> str:
        """Validate and correct a SQL query."""
        try:
            # Clean up the SQL first
            cleaned_sql = self._clean_sql(sql)
            
            # Apply common fixes
            corrected_sql = self._apply_common_fixes(cleaned_sql)
            
            # Parse and validate
            parsed = sqlparse.parse(corrected_sql)
            
            if not parsed:
                self.logger.warning("Could not parse SQL query")
                return corrected_sql
            
            # Check for common issues
            enhanced_sql = self._enhance_sql(corrected_sql, parsed[0])
            
            # Final validation
            if self._is_valid_sql(enhanced_sql):
                return enhanced_sql
            else:
                self.logger.warning("SQL validation failed, returning corrected version")
                return corrected_sql
                
        except Exception as e:
            self.logger.error(f"Error validating SQL: {e}")
            return sql  # Return original if validation fails
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up basic SQL formatting issues."""
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Split on common separators that indicate multiple queries or explanations
        separators = [
            r'\s+SQL query:\s*',
            r'\s+Query:\s*', 
            r'\s+Here\'s\s+',
            r'\s+The\s+query\s+is:\s*',
            r'\s+Answer:\s*'
        ]
        
        for separator in separators:
            parts = re.split(separator, sql, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Take the first part, but if it's very short, take the second part
                first_part = parts[0].strip()
                if len(first_part) > 10:  # Reasonable SQL query length
                    sql = first_part
                elif len(parts) > 1:
                    sql = parts[1].strip()
                break
        
        # If we have multiple statements separated by semicolons, take the first valid one
        if sql.count(';') > 1:
            statements = sql.split(';')
            for stmt in statements:
                stmt = stmt.strip()
                if len(stmt) > 5 and any(keyword in stmt.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']):
                    sql = stmt + ';'
                    break
        
        # Remove common prefixes/suffixes from AI responses
        prefixes_to_remove = [
            r'^(sql|query|here\'s|here is|the query is|answer):?\s*',
            r'^```sql\s*',
            r'^```\s*',
            r'^\*\*SQL\*\*:?\s*',
            r'^SQL query:\s*',
            r'^Query:\s*',
        ]
        
        suffixes_to_remove = [
            r'\s*```$',
            r'\s*;?\s*$',
        ]
        
        for prefix in prefixes_to_remove:
            sql = re.sub(prefix, '', sql, flags=re.IGNORECASE)
        
        # Ensure it ends with semicolon
        if not sql.rstrip().endswith(';'):
            sql = sql.rstrip() + ';'
        
        return sql
    
    def _apply_common_fixes(self, sql: str) -> str:
        """Apply common SQL syntax fixes."""
        corrected = sql
        
        for pattern, replacement in self.common_fixes.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def _enhance_sql(self, sql: str, parsed_stmt: sql.Statement) -> str:
        """Enhance SQL with better formatting and structure."""
        try:
            # Format the SQL for better readability
            formatted = sqlparse.format(
                sql,
                reindent=True,
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=False,
                use_space_around_operators=True
            )
            
            # Additional enhancements
            enhanced = self._add_missing_elements(formatted, parsed_stmt)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Could not enhance SQL: {e}")
            return sql
    
    def _add_missing_elements(self, sql: str, parsed_stmt: sql.Statement) -> str:
        """Add missing SQL elements that might improve the query."""
        enhanced = sql
        
        # Check if it's a SELECT statement
        if self._is_select_statement(parsed_stmt):
            # Add LIMIT if it's a simple SELECT without one (to prevent large result sets)
            if not re.search(r'\bLIMIT\s+\d+', enhanced, re.IGNORECASE):
                # Only add LIMIT for simple queries without aggregations
                if not re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|GROUP\s+BY)\b', enhanced, re.IGNORECASE):
                    # Remove the semicolon temporarily
                    enhanced = enhanced.rstrip(';')
                    enhanced += ' LIMIT 1000;'  # Reasonable default limit
        
        return enhanced
    
    def _is_select_statement(self, parsed_stmt: sql.Statement) -> bool:
        """Check if the parsed statement is a SELECT statement."""
        try:
            first_token = next(token for token in parsed_stmt.flatten() if not token.is_whitespace)
            return first_token.ttype is T.Keyword.DML and first_token.value.upper() == 'SELECT'
        except (StopIteration, AttributeError):
            return False
    
    def _is_valid_sql(self, sql: str) -> bool:
        """Perform basic SQL validation."""
        try:
            # Parse the SQL
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                return False
            
            # Check for basic structure
            stmt = parsed[0]
            tokens = list(stmt.flatten())
            
            # Must have at least some tokens
            if len(tokens) < 3:
                return False
            
            # Check for balanced parentheses
            paren_count = 0
            for token in tokens:
                if token.value == '(':
                    paren_count += 1
                elif token.value == ')':
                    paren_count -= 1
                    if paren_count < 0:
                        return False
            
            if paren_count != 0:
                return False
            
            # Check for basic SQL keywords
            sql_upper = sql.upper()
            has_sql_keywords = any(
                keyword in sql_upper 
                for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
            )
            
            if not has_sql_keywords:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_syntax_errors(self, sql: str) -> List[Dict[str, Any]]:
        """Get a list of syntax errors in the SQL."""
        errors = []
        
        try:
            # Parse the SQL
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                errors.append({
                    "type": "parse_error",
                    "message": "Could not parse SQL",
                    "severity": "high"
                })
                return errors
            
            # Check for common issues
            stmt = parsed[0]
            tokens = list(stmt.flatten())
            
            # Check for unbalanced parentheses
            paren_count = 0
            for i, token in enumerate(tokens):
                if token.value == '(':
                    paren_count += 1
                elif token.value == ')':
                    paren_count -= 1
                    if paren_count < 0:
                        errors.append({
                            "type": "syntax_error",
                            "message": f"Unmatched closing parenthesis at position {i}",
                            "severity": "high",
                            "position": i
                        })
            
            if paren_count > 0:
                errors.append({
                    "type": "syntax_error",
                    "message": f"Missing {paren_count} closing parenthesis(es)",
                    "severity": "high"
                })
            
            # Check for missing semicolon
            if not sql.rstrip().endswith(';'):
                errors.append({
                    "type": "syntax_warning",
                    "message": "Missing semicolon at end of statement",
                    "severity": "low"
                })
            
            # Check for potential issues with quotes
            single_quotes = sql.count("'")
            double_quotes = sql.count('"')
            
            if single_quotes % 2 != 0:
                errors.append({
                    "type": "syntax_error",
                    "message": "Unmatched single quotes",
                    "severity": "high"
                })
            
            if double_quotes % 2 != 0:
                errors.append({
                    "type": "syntax_warning",
                    "message": "Unmatched double quotes (consider using single quotes for strings)",
                    "severity": "medium"
                })
            
        except Exception as e:
            errors.append({
                "type": "validation_error",
                "message": f"Error during validation: {str(e)}",
                "severity": "medium"
            })
        
        return errors
    
    def suggest_improvements(self, sql: str) -> List[Dict[str, Any]]:
        """Suggest improvements for the SQL query."""
        suggestions = []
        
        try:
            sql_upper = sql.upper()
            
            # Suggest using LIMIT for large result sets
            if 'SELECT' in sql_upper and 'LIMIT' not in sql_upper:
                if not any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                    suggestions.append({
                        "type": "performance",
                        "message": "Consider adding LIMIT clause to prevent large result sets",
                        "suggestion": "Add 'LIMIT <number>' at the end of your query"
                    })
            
            # Suggest using WHERE clause for better performance
            if 'SELECT' in sql_upper and 'FROM' in sql_upper and 'WHERE' not in sql_upper:
                suggestions.append({
                    "type": "performance",
                    "message": "Consider adding WHERE clause to filter results",
                    "suggestion": "Add 'WHERE <condition>' to filter your data"
                })
            
            # Suggest using indexes if ORDER BY is present
            if 'ORDER BY' in sql_upper:
                suggestions.append({
                    "type": "performance",
                    "message": "Ensure columns in ORDER BY clause are indexed for better performance",
                    "suggestion": "Create indexes on columns used in ORDER BY"
                })
            
            # Suggest explicit column names instead of SELECT *
            if 'SELECT *' in sql_upper:
                suggestions.append({
                    "type": "best_practice",
                    "message": "Consider selecting specific columns instead of using SELECT *",
                    "suggestion": "Replace 'SELECT *' with 'SELECT column1, column2, ...'"
                })
            
            # Check for potential SQL injection vulnerabilities (basic check)
            if any(suspicious in sql.lower() for suspicious in ['drop', 'delete', 'truncate']) and 'where' not in sql.lower():
                suggestions.append({
                    "type": "security",
                    "message": "Destructive operations without WHERE clause detected",
                    "suggestion": "Always use WHERE clause with DELETE, UPDATE, or DROP statements"
                })
            
        except Exception as e:
            self.logger.warning(f"Error generating suggestions: {e}")
        
        return suggestions
    
    def format_sql(self, sql: str, style: str = "standard") -> str:
        """Format SQL according to different style guidelines."""
        try:
            if style == "compact":
                return sqlparse.format(
                    sql,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=True,
                    reindent=False
                )
            elif style == "readable":
                return sqlparse.format(
                    sql,
                    reindent=True,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=False,
                    use_space_around_operators=True,
                    indent_width=2
                )
            else:  # standard
                return sqlparse.format(
                    sql,
                    reindent=True,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=False,
                    use_space_around_operators=True
                )
        except Exception as e:
            self.logger.warning(f"Error formatting SQL: {e}")
            return sql 