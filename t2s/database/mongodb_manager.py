"""MongoDB-specific database management for T2S."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from rich.console import Console

from ..core.config import Config, DatabaseConfig


class MongoDBManager:
    """Manages MongoDB connections and query execution."""

    def __init__(self, config: Config):
        """Initialize the MongoDB manager."""
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, MongoClient] = {}

    def _create_connection_string(self, db_config: DatabaseConfig) -> str:
        """Create a MongoDB connection string."""
        if db_config.username and db_config.password:
            # Authenticated connection
            auth_part = f"{db_config.username}:{db_config.password}@"
        else:
            auth_part = ""

        host = db_config.host or "localhost"
        port = db_config.port or 27017

        return f"mongodb://{auth_part}{host}:{port}/"

    def get_connection(self, db_name: str) -> MongoClient:
        """Get a MongoDB connection."""
        if db_name not in self.config.config.databases:
            raise ValueError(f"Database {db_name} not configured")

        if db_name not in self.connections:
            db_config = self.config.config.databases[db_name]
            connection_string = self._create_connection_string(db_config)

            try:
                self.connections[db_name] = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000
                )
                # Test connection
                self.connections[db_name].admin.command('ping')
            except ConnectionFailure as e:
                self.logger.error(f"Failed to connect to MongoDB: {e}")
                raise RuntimeError(f"MongoDB connection failed: {str(e)}")

        return self.connections[db_name]

    async def test_connection(self, db_name: str) -> bool:
        """Test a MongoDB connection."""
        try:
            client = self.get_connection(db_name)
            client.admin.command('ping')
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def execute_query(self, mql: str, db_name: str) -> Dict[str, Any]:
        """Execute a MongoDB query and return results."""
        try:
            client = self.get_connection(db_name)
            db_config = self.config.config.databases[db_name]

            if not db_config.database:
                raise ValueError("MongoDB database name not specified in configuration")

            db = client[db_config.database]

            # Parse and execute the MQL query
            # MQL can be in various forms:
            # 1. db.collection.find({query})
            # 2. db.collection.aggregate([pipeline])
            # 3. Python dict format

            result = self._execute_mql(db, mql)

            return result

        except PyMongoError as e:
            self.logger.error(f"MongoDB error executing query: {e}")
            raise RuntimeError(f"MongoDB error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {e}")
            raise RuntimeError(f"Query execution error: {str(e)}")

    def _quote_unquoted_keys(self, js_str: str) -> str:
        """Convert JavaScript object notation to valid JSON by quoting unquoted keys.

        Handles:
        - Unquoted keys: {$lookup: ...} -> {"$lookup": ...}
        - Already-quoted keys: {"albums.Title": ...} -> unchanged
        - String values: {msg: "Hello: World"} -> {"msg": "Hello: World"}
        - Arrays: ["$field"] -> unchanged
        - Nested objects: recursive processing
        """
        result = []
        i = 0
        in_string = False
        string_char = None

        while i < len(js_str):
            char = js_str[i]

            # Track string boundaries (both single and double quotes)
            if char in ('"', "'") and (i == 0 or js_str[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                result.append(char)
                i += 1
                continue

            # Inside string, just copy character
            if in_string:
                result.append(char)
                i += 1
                continue

            # Outside strings: look for unquoted keys
            # Keys appear after: { , [ or whitespace, followed by identifier and :
            if char in ('{', ',', '['):
                result.append(char)
                i += 1

                # Skip whitespace after delimiter
                while i < len(js_str) and js_str[i] in (' ', '\n', '\t', '\r'):
                    result.append(js_str[i])
                    i += 1

                # Check if next token is an identifier (potential key)
                if i < len(js_str) and (js_str[i].isalpha() or js_str[i] in ('$', '_')):
                    # Collect the identifier
                    identifier_start = i
                    identifier = ''

                    # Handle $ prefix
                    if js_str[i] == '$':
                        identifier += '$'
                        i += 1

                    # Collect alphanumeric and underscore characters
                    while i < len(js_str) and (js_str[i].isalnum() or js_str[i] == '_'):
                        identifier += js_str[i]
                        i += 1

                    # Skip whitespace after identifier
                    spaces_after = ''
                    while i < len(js_str) and js_str[i] in (' ', '\n', '\t', '\r'):
                        spaces_after += js_str[i]
                        i += 1

                    # Check if followed by colon (making it a key)
                    if i < len(js_str) and js_str[i] == ':':
                        # This is an unquoted key - add quotes
                        result.append('"')
                        result.append(identifier)
                        result.append('"')
                        result.append(spaces_after)
                        # Don't increment i - the colon will be added in next iteration
                    else:
                        # Not a key, just an identifier - keep as-is
                        result.append(identifier)
                        result.append(spaces_after)
                        # Don't increment i - continue from current position
                continue

            # Copy all other characters as-is
            result.append(char)
            i += 1

        return ''.join(result)

    def _execute_mql(self, db, mql: str) -> Dict[str, Any]:
        """Execute MQL query on the database."""
        import json
        import re

        # Remove comments and clean the query
        mql = re.sub(r'//.*', '', mql)
        mql = mql.strip()

        # Remove trailing semicolons (MongoDB doesn't use them)
        mql = mql.rstrip(';').strip()

        # First, check for direct DB operations (no collection)
        direct_db_match = re.search(r'^db\.(\w+)\s*\((.*)\)', mql, re.DOTALL)

        if direct_db_match:
            operation = direct_db_match.group(1)
            params_str = direct_db_match.group(2).strip()

            # List of direct DB operations
            direct_db_operations = [
                'getCollectionNames', 'listCollections', 'getCollectionInfos',
                'adminCommand', 'runCommand', 'stats'
            ]

            if operation in direct_db_operations:
                # Handle direct DB operations
                if operation == 'getCollectionNames':
                    collection_names = db.list_collection_names()
                    df = pd.DataFrame([{"collection": name} for name in collection_names])
                    return {
                        "data": df,
                        "rows_affected": len(collection_names),
                        "query_type": "list_collections"
                    }

                elif operation == 'listCollections':
                    collections = db.list_collections()
                    collection_list = list(collections)
                    # Filter to show only relevant fields (exclude metadata like info, options, idIndex)
                    simplified_collections = [
                        {"name": c.get("name"), "type": c.get("type", "collection")}
                        for c in collection_list
                    ]
                    df = pd.DataFrame(simplified_collections)
                    return {
                        "data": df,
                        "rows_affected": len(simplified_collections),
                        "query_type": "list_collections"
                    }

                elif operation == 'getCollectionInfos':
                    collection_infos = db.list_collections()
                    infos = list(collection_infos)
                    df = pd.DataFrame(infos)
                    return {
                        "data": df,
                        "rows_affected": len(infos),
                        "query_type": "collection_info"
                    }

                elif operation == 'stats':
                    stats = db.command("dbStats")
                    df = pd.DataFrame([stats])
                    return {
                        "data": df,
                        "rows_affected": 1,
                        "query_type": "stats"
                    }

                elif operation in ['adminCommand', 'runCommand']:
                    # Parse the command from params
                    if params_str:
                        try:
                            # Try JSON first
                            command = json.loads(params_str)
                        except (json.JSONDecodeError, ValueError):
                            # Fallback to eval for JavaScript object notation
                            try:
                                # Provide namespace for common MongoDB commands
                                # This handles ES6 shorthand syntax like {listCollections} -> {listCollections: 1}
                                namespace = {
                                    'listCollections': 1,
                                    'listDatabases': 1,
                                    'dbStats': 1,
                                    'serverStatus': 1,
                                    'buildInfo': 1,
                                    'collStats': 1,
                                    'hostInfo': 1,
                                }
                                command = eval(params_str, {"__builtins__": {}}, namespace)
                            except Exception as e:
                                raise ValueError(f"Failed to parse command parameters: {e}")

                        result = db.command(command)
                        df = pd.DataFrame([result])
                        return {
                            "data": df,
                            "rows_affected": 1,
                            "query_type": operation
                        }
                    else:
                        raise ValueError(f"{operation} requires a command parameter")

        # Try to detect collection-based operations
        collection_match = re.search(r'db\.(\w+)\.(\w+)\s*\((.*)\)', mql, re.DOTALL)

        if collection_match:
            collection_name = collection_match.group(1)
            operation = collection_match.group(2)
            query_str = collection_match.group(3)

            collection = db[collection_name]

            # Parse the query parameters
            try:
                # Try to parse as JSON
                if query_str.strip():
                    # Handle both single and multiple parameters
                    query_str = query_str.strip()

                    if query_str.startswith('['):
                        # Aggregation pipeline
                        try:
                            # Convert JavaScript object notation to valid JSON
                            quoted_str = self._quote_unquoted_keys(query_str)
                            query_params = json.loads(quoted_str)
                        except (json.JSONDecodeError, ValueError) as json_err:
                            self.logger.error(f"Pipeline parsing failed: {json_err}")
                            self.logger.debug(f"Original query: {query_str[:200]}")
                            raise ValueError(f"Cannot parse aggregation pipeline. Query: {query_str[:100]}... Error: {json_err}")
                    elif query_str.startswith('{'):
                        # Single document query
                        try:
                            # Convert JavaScript object notation to valid JSON
                            quoted_str = self._quote_unquoted_keys(query_str)
                            query_params = json.loads(quoted_str)
                        except (json.JSONDecodeError, ValueError) as json_err:
                            self.logger.error(f"Query parsing failed: {json_err}")
                            self.logger.debug(f"Original query: {query_str[:200]}")
                            raise ValueError(f"Cannot parse query parameters. Query: {query_str[:100]}... Error: {json_err}")
                    else:
                        # Try to evaluate as Python
                        try:
                            js_converted = query_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                            query_params = eval(js_converted, {"__builtins__": {}}, {})
                        except Exception as eval_err:
                            self.logger.error(f"Parameter parsing failed: {eval_err}")
                            raise ValueError(f"Cannot parse query parameters: {eval_err}")
                else:
                    query_params = {}
            except ValueError:
                # Re-raise ValueError to propagate clear error messages
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error parsing query parameters: {e}")
                raise ValueError(f"Failed to parse query parameters: {e}")

            # Execute based on operation
            if operation == "find":
                if isinstance(query_params, dict):
                    cursor = collection.find(query_params)
                else:
                    cursor = collection.find({})

                results = list(cursor)
                df = pd.DataFrame(results)

                # Convert ObjectId to string for display
                if '_id' in df.columns:
                    df['_id'] = df['_id'].astype(str)

                return {
                    "data": df,
                    "rows_affected": len(results),
                    "query_type": "find"
                }

            elif operation == "aggregate":
                if isinstance(query_params, list):
                    cursor = collection.aggregate(query_params)
                else:
                    cursor = collection.aggregate([query_params])

                results = list(cursor)
                df = pd.DataFrame(results)

                # Convert ObjectId to string for display
                if '_id' in df.columns:
                    df['_id'] = df['_id'].astype(str)

                return {
                    "data": df,
                    "rows_affected": len(results),
                    "query_type": "aggregate"
                }

            elif operation == "countDocuments":
                if isinstance(query_params, dict):
                    count = collection.count_documents(query_params)
                else:
                    count = collection.count_documents({})

                df = pd.DataFrame([{"count": count}])

                return {
                    "data": df,
                    "rows_affected": 1,
                    "query_type": "count"
                }

            elif operation == "distinct":
                # distinct needs field name and optional query
                # Parse: distinct("field", {query})
                params = query_str.split(',', 1)
                field = params[0].strip().strip('"').strip("'")
                query = json.loads(params[1]) if len(params) > 1 else {}

                results = collection.distinct(field, query)
                df = pd.DataFrame([{field: val} for val in results])

                return {
                    "data": df,
                    "rows_affected": len(results),
                    "query_type": "distinct"
                }

            else:
                raise ValueError(f"Unsupported operation: {operation}")

        else:
            raise ValueError(f"Could not parse MQL query: {mql}")

    async def get_schema_info(self, db_name: str) -> Dict[str, Any]:
        """Get comprehensive schema information for a MongoDB database."""
        try:
            client = self.get_connection(db_name)
            db_config = self.config.config.databases[db_name]

            if not db_config.database:
                raise ValueError("MongoDB database name not specified")

            db = client[db_config.database]

            schema_info = {
                "collections": {},
                "database_name": db_config.database
            }

            # Get all collection names
            collection_names = db.list_collection_names()

            for collection_name in collection_names:
                collection = db[collection_name]

                # Sample documents to infer schema
                # Use up to 100 documents for better schema inference
                doc_count = collection.count_documents({})
                sample_size = min(100, doc_count)
                sample_docs = list(collection.find().limit(sample_size))

                if not sample_docs:
                    schema_info["collections"][collection_name] = {
                        "fields": [],
                        "sample_count": 0,
                        "document_count": 0,
                        "indexes": []
                    }
                    continue

                # Infer schema from samples
                fields = set()
                field_types = {}
                field_frequency = {}  # Track how often each field appears

                for doc in sample_docs:
                    for key, value in doc.items():
                        fields.add(key)
                        if key not in field_types:
                            field_types[key] = type(value).__name__
                        # Track field frequency
                        field_frequency[key] = field_frequency.get(key, 0) + 1

                # Get indexes
                indexes = []
                for index in collection.list_indexes():
                    indexes.append({
                        "name": index.get("name"),
                        "keys": list(index.get("key", {}).keys())
                    })

                # Calculate field frequency percentages
                field_frequency_pct = {
                    field: f"{count}/{sample_size} ({int(count/sample_size*100)}%)"
                    for field, count in field_frequency.items()
                }

                schema_info["collections"][collection_name] = {
                    "fields": list(fields),
                    "field_types": field_types,
                    "field_frequency": field_frequency_pct,
                    "document_count": doc_count,
                    "sample_count": sample_size,
                    "indexes": indexes
                }

            return schema_info

        except Exception as e:
            self.logger.error(f"Error getting MongoDB schema info: {e}")
            raise RuntimeError(f"Schema analysis error: {str(e)}")

    async def add_database(self, name: str, **kwargs) -> bool:
        """Add a new MongoDB database configuration."""
        try:
            db_config = DatabaseConfig(
                name=name,
                type="mongodb",
                **kwargs
            )

            # Test the connection
            connection_string = self._create_connection_string(db_config)
            test_client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000
            )
            test_client.admin.command('ping')
            test_client.close()

            self.config.add_database(name, db_config)
            self.console.print(f"[green]Successfully added MongoDB database '{name}'[/green]")
            return True

        except Exception as e:
            self.logger.error(f"Error adding MongoDB database: {e}")
            self.console.print(f"[red]Error adding database: {e}[/red]")
            return False

    def close_connection(self, db_name: str) -> None:
        """Close a MongoDB connection."""
        if db_name in self.connections:
            try:
                self.connections[db_name].close()
                del self.connections[db_name]
            except Exception as e:
                self.logger.warning(f"Error closing connection: {e}")

    def close_all_connections(self) -> None:
        """Close all MongoDB connections."""
        for db_name in list(self.connections.keys()):
            self.close_connection(db_name)
        self.console.print("[green]All MongoDB connections closed[/green]")

    def list_databases(self, host: str = "localhost", port: int = 27017, username: str = None, password: str = None) -> List[str]:
        """List all databases available on a MongoDB server."""
        try:
            # Build connection string
            if username and password:
                auth_part = f"{username}:{password}@"
            else:
                auth_part = ""

            connection_string = f"mongodb://{auth_part}{host}:{port}/"

            client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            databases = client.list_database_names()
            client.close()

            return databases

        except Exception as e:
            self.logger.error(f"Error listing databases: {e}")
            return []

    def discover_local_mongodb(self) -> List[Dict[str, Any]]:
        """Discover MongoDB instances on localhost."""
        discovered = []
        common_ports = [27017, 27018, 27019]  # Common MongoDB ports

        for port in common_ports:
            try:
                client = MongoClient(f"mongodb://localhost:{port}/", serverSelectionTimeoutMS=1000)
                client.admin.command('ping')

                # Get server info
                server_info = client.server_info()
                databases = client.list_database_names()

                discovered.append({
                    "host": "localhost",
                    "port": port,
                    "version": server_info.get("version", "unknown"),
                    "databases": databases,
                    "database_count": len(databases)
                })

                client.close()

            except Exception:
                # Port not accessible or no MongoDB running
                continue

        return discovered
