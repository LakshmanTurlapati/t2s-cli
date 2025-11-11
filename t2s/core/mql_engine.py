"""MQL Engine for Text-to-MongoDB conversion."""

import re
import logging
from typing import Dict, Any, Optional

from rich.console import Console

from .config import Config
from ..models.model_manager import ModelManager


class MQLEngine:
    """Handles all MQL-specific logic for MongoDB query generation and validation."""

    def __init__(self, config: Config, model_manager: ModelManager):
        """Initialize MQL engine."""
        self.config = config
        self.model_manager = model_manager
        self.console = Console()
        self.logger = logging.getLogger(__name__)

    async def generate_mql(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate MQL query from natural language."""
        # Create system prompt based on model intelligence
        system_prompt = self._create_system_prompt(schema_info)
        user_prompt = natural_query

        # Log prompts for debugging
        self.logger.debug(f"System prompt: {system_prompt[:500]}...")
        self.logger.debug(f"User prompt: {user_prompt}")

        # Generate using AI model
        generated_text = await self.model_manager.generate_sql(system_prompt, user_prompt)

        # Extract MQL query
        mql_query = self.extract_mql(generated_text)

        if not mql_query:
            raise ValueError("Failed to extract MQL query from model output")

        self.logger.info(f"Generated MQL: {mql_query}")
        return mql_query

    async def validate_mql(self, mql: str) -> str:
        """Validate MQL query - basic validation only."""
        # For MongoDB, we do basic structure validation
        # No complex validation needed as MongoDB is flexible
        if not self._is_valid_mql_structure(mql):
            raise ValueError(f"Invalid MQL structure: {mql}")

        return mql

    def validate_mql_fields(self, mql: str, schema_info: Dict[str, Any]) -> tuple[bool, str]:
        """Validate that MQL only uses fields present in schema.

        Returns:
            Tuple of (is_valid, error_message)
        """
        import re

        # Extract collection name from MQL
        collection_match = re.search(r'db\.(\w+)', mql)
        if not collection_match:
            return True, ""  # Can't validate without collection name

        collection_name = collection_match.group(1)

        # Check if collection exists in schema
        if collection_name not in schema_info.get("collections", {}):
            # Check if it's a direct DB operation (no collection)
            direct_ops = ['getCollectionNames', 'listCollections', 'getCollectionInfos',
                         'adminCommand', 'runCommand', 'stats']
            if collection_name in direct_ops:
                return True, ""  # Direct operations don't need field validation

            return False, f"Collection '{collection_name}' not found in schema"

        collection_schema = schema_info["collections"][collection_name]
        allowed_fields = set(collection_schema.get("fields", []))
        allowed_fields.add("_id")  # _id is always allowed

        # Extract field references from MQL
        # Look for patterns like {field: or "$field or 'field': or "field":
        field_patterns = [
            r'[{\s,]([a-zA-Z_]\w*)\s*:',      # {field: value or , field: value
            r'\$([a-zA-Z_]\w+)',               # $field (in aggregations)
        ]

        used_fields = set()
        for pattern in field_patterns:
            matches = re.findall(pattern, mql)
            used_fields.update(matches)

        # Remove MongoDB operators and aggregation stage keywords from used fields
        mongo_keywords = {
            # Query operators
            'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'nin', 'and', 'or', 'not', 'nor',
            'exists', 'type', 'regex', 'where', 'all', 'elemMatch', 'size',
            # Aggregation pipeline stages and operators
            'match', 'group', 'sort', 'limit', 'skip', 'project', 'unwind', 'lookup', 'out',
            'sum', 'avg', 'min', 'max', 'push', 'addToSet', 'first', 'last', 'count',
            'multiply', 'divide', 'add', 'subtract', 'mod', 'concat', 'substr', 'toLower', 'toUpper',
            # $lookup stage parameters
            'from', 'localField', 'foreignField', 'as',
            # $group/_id references
            '_id', 'id',
            # Common projection/computed field names (more lenient validation)
            'total', 'result', 'value', 'name', 'title', 'description'
        }
        used_fields = {f for f in used_fields if f not in mongo_keywords}

        # Check for invalid fields
        invalid_fields = used_fields - allowed_fields
        if invalid_fields:
            return False, f"Invalid fields in query: {', '.join(sorted(invalid_fields))}. Available fields: {', '.join(sorted(allowed_fields))}"

        return True, ""

    def extract_mql(self, generated_text: str) -> str:
        """Extract MongoDB query from generated text."""
        if not generated_text:
            self.logger.warning("No text provided for MQL extraction")
            return ""

        self.logger.debug(f"Raw model output for MQL extraction (first 500 chars): {generated_text[:500]}")
        text = generated_text.strip()

        # Clean up common prefixes
        prefixes_to_remove = [
            r'^\s*MQL:\s*',
            r'^\s*MongoDB Query:\s*',
            r'^\s*Query:\s*',
            r'^\s*Answer:\s*',
        ]
        for prefix_pattern in prefixes_to_remove:
            text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        text = text.strip()

        # Try to find MQL in code blocks
        code_block_patterns = [
            r'```(?:javascript|js|json|mql|mongodb)?\s*(.*?)\s*```',
            r'```\s*(db\..*?)\s*```'
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                mql_content = match.group(1).strip()
                # Look for db. pattern inside the code block
                db_matches = re.findall(r'(db\.\w+\.[a-zA-Z]+\([^)]*\)(?:\.[a-zA-Z]+\([^)]*\))*)', mql_content)
                if db_matches:
                    for db_match in db_matches:
                        if self._is_valid_mql_structure(db_match):
                            return db_match
                # Also look for direct DB operations (no collection)
                direct_ops = ['getCollectionNames', 'listCollections', 'getCollectionInfos', 'adminCommand', 'runCommand', 'stats']
                for op in direct_ops:
                    direct_pattern = rf'(db\.{op}\s*\([^)]*\))'
                    direct_matches = re.findall(direct_pattern, mql_content)
                    if direct_matches:
                        for match in direct_matches:
                            if self._is_valid_mql_structure(match):
                                return match
                # Try the whole content if it looks like MQL
                if self._is_valid_mql_structure(mql_content):
                    return mql_content

        # Try to find db.collection patterns with more flexible matching
        mql_patterns = [
            # Match complete queries with chained methods
            r'(db\.\w+\.(?:find|aggregate|countDocuments|distinct)\([^)]*\)(?:\s*\.\s*(?:sort|limit|skip|toArray)\([^)]*\))*)',
            # Match simple queries
            r'(db\.\w+\.(?:find|aggregate|countDocuments|distinct|insertOne|insertMany|updateOne|updateMany|deleteOne|deleteMany|findOne|replaceOne)\s*\([^)]*\))',
            # Match queries with nested parentheses (for aggregation pipelines)
            r'(db\.\w+\.(?:aggregate)\s*\(\s*\[[\s\S]*?\]\s*\))',
            # Match direct DB operations (no collection)
            r'(db\.(?:getCollectionNames|listCollections|getCollectionInfos|adminCommand|runCommand|stats)\s*\([^)]*\))',
        ]

        for pattern in mql_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                mql = match.strip() if isinstance(match, str) else match[0].strip()
                if self._is_valid_mql_structure(mql):
                    return mql

        # Look for any db. patterns across multiple lines
        db_pattern = r'(db\.\w+\.[a-zA-Z]+\s*\(.*?\))'
        matches = re.findall(db_pattern, text, re.DOTALL)
        for match in matches:
            if self._is_valid_mql_structure(match.strip()):
                return match.strip()

        # Look for direct DB operations as fallback
        direct_pattern = r'(db\.(?:getCollectionNames|listCollections|getCollectionInfos|adminCommand|runCommand|stats)\s*\([^)]*\))'
        direct_matches = re.findall(direct_pattern, text, re.DOTALL)
        for match in direct_matches:
            if self._is_valid_mql_structure(match.strip()):
                return match.strip()

        # Last resort: look for any line starting with db.
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('db.'):
                if self._is_valid_mql_structure(line_stripped):
                    return line_stripped

        # If we get here, extraction failed
        self.logger.warning(f"Failed to extract MQL from model output. Full text: {generated_text}")
        return ""

    def _is_valid_mql_structure(self, mql: str) -> bool:
        """Basic MQL structure validation."""
        if not mql:
            return False

        mql = mql.strip()

        # Must start with db.
        if not mql.startswith('db.'):
            return False

        # Collection operations (db.collection.operation)
        collection_operations = [
            'find', 'aggregate', 'countDocuments', 'distinct',
            'insertOne', 'insertMany', 'updateOne', 'updateMany',
            'deleteOne', 'deleteMany', 'findOne', 'replaceOne'
        ]

        # Direct DB operations (db.operation without collection)
        direct_db_operations = [
            'getCollectionNames', 'listCollections', 'getCollectionInfos',
            'adminCommand', 'runCommand', 'stats'
        ]

        # Check for collection operations: db.collection.operation(
        for op in collection_operations:
            if f'.{op}(' in mql:
                return True

        # Check for direct DB operations: db.operation(
        for op in direct_db_operations:
            # Pattern: db.operation( (with no collection name in between)
            if re.match(rf'^db\.{op}\s*\(', mql):
                return True

        return False

    def _create_system_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Create system prompt optimized for the current model's intelligence level."""
        current_model_id = self.config.config.selected_model

        if not current_model_id or current_model_id not in self.config.SUPPORTED_MODELS:
            return self._get_mongodb_intermediate_prompt(schema_info)

        model_config = self.config.SUPPORTED_MODELS[current_model_id]
        intelligence_level = self._determine_model_intelligence(current_model_id, model_config)

        self.logger.info(f"Model '{current_model_id}' classified as '{intelligence_level}' intelligence level")

        return self._get_intelligence_based_prompt(intelligence_level, schema_info)

    def _determine_model_intelligence(self, model_id: str, model_config) -> str:
        """Determine the intelligence level of a model."""
        # Similar to SQL engine but for MongoDB
        if any(model_name in model_id.lower() for model_name in ["llama", "phi", "mistral", "qwen"]):
            return "advanced"

        if "smollm" in model_id.lower():
            return "simple"

        return "intermediate"

    def _get_intelligence_based_prompt(self, intelligence_level: str, schema_info: Dict[str, Any]) -> str:
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

### CRITICAL SCHEMA CONSTRAINTS:
- **Use ONLY the fields listed in the schema below**
- **Do NOT invent or assume fields that are not explicitly listed**
- **If a field is not in the schema, the query MUST NOT use it**
- **If the question asks for data that doesn't exist in the schema, return an empty find query**
- **Choose the PRIMARY collection mentioned in the question** (e.g., "list artists" -> use Artist collection, not InvoiceLine)

Adhere to these rules:
- **Use MongoDB Query Language (MQL)** - use proper MongoDB syntax
- **Return queries in this format**: db.collection.operation({{query}})
- **Common operations**: find(), aggregate(), countDocuments(), distinct()
- **Use proper operators**: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or, $not
- **For aggregation**, use pipeline stages: $match, $group, $sort, $limit, $project, $lookup
- **Field references in aggregation**: Use "$fieldName" syntax
- **Always include the collection name** in your query
- **DO NOT add semicolons** - MQL doesn't use semicolons

### FIELD SELECTION RULES:
- **ALWAYS exclude _id field by default**: Use {{_id: 0}} in projections unless the question explicitly asks for IDs/ObjectIds
- **For find() queries**: Use db.collection.find({{query}}, {{_id: 0}}) to exclude _id
- **For aggregations**: Add final $project stage with {{_id: 0}} to exclude _id from results
- **Return only relevant fields**: Only include fields that directly answer the user's question

### MongoDB Schema:
{schema_string}

### Examples:
Question: "Find all users"
MQL: db.users.find({{}}, {{_id: 0}})

Question: "Show user names and emails"
MQL: db.users.find({{}}, {{_id: 0, name: 1, email: 1}})

Question: "Count documents where age is greater than 25"
MQL: db.users.countDocuments({{age: {{$gt: 25}}}})

Question: "Get average salary by department"
MQL: db.employees.aggregate([
  {{$group: {{_id: "$department", avgSalary: {{$avg: "$salary"}}}}}},
  {{$sort: {{avgSalary: -1}}}},
  {{$project: {{_id: 0, department: "$_id", avgSalary: 1}}}}
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

CRITICAL: Use ONLY the fields listed in the schema below. Do NOT invent fields that are not listed.

Follow these steps:
1. Understand what data the user is asking for
2. Identify the PRIMARY collection mentioned in the question (e.g., "artists" -> Artist collection)
3. Check if the required fields exist in that collection's schema
4. Choose the right MongoDB operation (find, aggregate, countDocuments, distinct)
5. Construct the query using ONLY fields from the schema

MongoDB Schema:
{schema_string}

MongoDB Query Format:
- Simple queries: db.collection.find({{field: value}}, {{_id: 0}})  // Always exclude _id
- With specific fields: db.collection.find({{}}, {{_id: 0, name: 1, age: 1}})
- Aggregations: db.collection.aggregate([{{$match: {{}}}}, {{$group: {{}}}}, {{$project: {{_id: 0}}}}])
- Counts: db.collection.countDocuments({{field: value}})
- DO NOT add semicolons
- DO NOT use fields not in the schema
- ALWAYS exclude _id unless the question asks for IDs

Example:
db.{first_collection}.find({{}}, {{_id: 0}})

Query: {{user_question}}
MQL:"""

    def _get_mongodb_simple_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Simple MongoDB prompt for smaller models."""
        # Build simple schema representation
        schema_parts = []
        for collection_name, collection_data in list(schema_info.get("collections", {}).items())[:5]:
            fields = collection_data.get("fields", [])[:10]  # Limit fields to avoid overwhelming small models
            fields_str = ", ".join(fields)
            schema_parts.append(f"{collection_name}: {fields_str}")

        schema_str = "\n".join(schema_parts) if schema_parts else "users: name, email\norders: id, date"

        return f"""Convert the question to a MongoDB query.
Use ONLY the fields listed below.
Use the PRIMARY collection mentioned in the question.
DO NOT include _id field unless asked.

Schema:
{schema_str}

Question: {{user_question}}

MongoDB query (exclude _id):"""
