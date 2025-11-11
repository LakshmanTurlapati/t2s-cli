"""MQL (MongoDB Query Language) validator for T2S."""

import re
import logging
from typing import Optional


class MQLValidator:
    """Validates and provides basic checks for MongoDB queries."""

    def __init__(self):
        """Initialize the MQL validator."""
        self.logger = logging.getLogger(__name__)

        self.valid_operations = [
            'find', 'findOne', 'aggregate', 'countDocuments', 'distinct',
            'insertOne', 'insertMany', 'updateOne', 'updateMany',
            'deleteOne', 'deleteMany', 'replaceOne'
        ]

        self.direct_db_operations = [
            'getCollectionNames', 'listCollections', 'getCollectionInfos',
            'adminCommand', 'runCommand', 'stats'
        ]

        self.valid_operators = [
            '$eq', '$ne', '$gt', '$gte', '$lt', '$lte',
            '$in', '$nin', '$and', '$or', '$not', '$nor',
            '$exists', '$type', '$regex', '$where',
            '$all', '$elemMatch', '$size',
            '$match', '$group', '$sort', '$limit', '$skip',
            '$project', '$unwind', '$lookup', '$out',
            '$sum', '$avg', '$min', '$max', '$push', '$addToSet',
            '$first', '$last', '$count'
        ]

    async def validate_and_correct(self, mql: str) -> str:
        """Validate and attempt to correct MQL query."""
        if not mql or not mql.strip():
            raise ValueError("Empty MQL query")

        mql = mql.strip()

        # Remove trailing semicolons (MongoDB doesn't use them)
        mql = mql.rstrip(';').strip()

        # Basic validation
        if not mql.startswith('db.'):
            self.logger.warning(f"MQL query does not start with 'db.': {mql}")
            raise ValueError("Invalid MQL query format: must start with 'db.'")

        # Check for valid collection operation
        has_valid_operation = any(f'.{op}(' in mql for op in self.valid_operations)

        # Check for valid direct DB operation
        has_direct_operation = any(re.match(rf'^db\.{op}\s*\(', mql) for op in self.direct_db_operations)

        if not has_valid_operation and not has_direct_operation:
            self.logger.warning(f"MQL query does not contain a valid operation: {mql}")
            raise ValueError(f"Invalid MQL operation. Valid operations: {', '.join(self.valid_operations + self.direct_db_operations)}")

        # Basic syntax checks
        if not self._check_balanced_brackets(mql):
            self.logger.warning(f"MQL query has unbalanced brackets: {mql}")
            raise ValueError("Invalid MQL syntax: unbalanced brackets")

        # Try to validate JSON-like structures
        try:
            self._validate_query_structure(mql)
        except Exception as e:
            self.logger.warning(f"MQL structure validation failed: {e}")
            # Don't raise here, just log the warning

        return mql

    def _check_balanced_brackets(self, mql: str) -> bool:
        """Check if brackets are balanced in the query."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in mql:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack[-1]] != char:
                    return False
                stack.pop()

        return len(stack) == 0

    def _validate_query_structure(self, mql: str) -> None:
        """Validate the structure of the MQL query."""
        # Check if this is a direct DB operation (no collection)
        is_direct_db_op = any(re.match(rf'^db\.{op}\s*\(', mql) for op in self.direct_db_operations)

        if is_direct_db_op:
            # For direct DB operations, just validate the operation exists
            operation_match = re.search(r'^db\.(\w+)\(', mql)
            if not operation_match:
                raise ValueError("Cannot extract direct DB operation")

            operation = operation_match.group(1)
            if operation not in self.direct_db_operations:
                raise ValueError(f"Invalid direct DB operation: {operation}")
            return  # Skip collection-based validation

        # Extract collection name (for collection-based operations)
        collection_match = re.search(r'db\.(\w+)\.', mql)
        if not collection_match:
            raise ValueError("Cannot extract collection name")

        # Extract operation
        operation_match = re.search(r'\.(\w+)\(', mql)
        if not operation_match:
            raise ValueError("Cannot extract operation")

        operation = operation_match.group(1)
        if operation not in self.valid_operations:
            raise ValueError(f"Invalid operation: {operation}")

        # Check for common mistakes
        if operation == 'aggregate':
            # Aggregate should have array parameter
            if not re.search(r'aggregate\s*\(\s*\[', mql):
                self.logger.warning("Aggregate operation should use an array parameter")

        # Validate operators used
        operators_found = re.findall(r'\$\w+', mql)
        for op in operators_found:
            if op not in self.valid_operators:
                self.logger.warning(f"Unknown operator: {op}")

    def validate_syntax(self, mql: str) -> dict:
        """Validate MQL syntax and return detailed validation results."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        if not mql or not mql.strip():
            results["valid"] = False
            results["errors"].append("Empty query")
            return results

        mql = mql.strip()

        # Check start with db.
        if not mql.startswith('db.'):
            results["valid"] = False
            results["errors"].append("Query must start with 'db.'")

        # Check for operation
        has_operation = any(f'.{op}(' in mql for op in self.valid_operations)
        if not has_operation:
            results["valid"] = False
            results["errors"].append("No valid operation found")

        # Check balanced brackets
        if not self._check_balanced_brackets(mql):
            results["valid"] = False
            results["errors"].append("Unbalanced brackets")

        # Check for common issues
        if '  ' in mql:
            results["warnings"].append("Query contains multiple spaces")

        # Check operator usage
        operators_found = re.findall(r'\$\w+', mql)
        for op in operators_found:
            if op not in self.valid_operators:
                results["warnings"].append(f"Unknown operator: {op}")

        return results
