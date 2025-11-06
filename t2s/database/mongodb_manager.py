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

    def _execute_mql(self, db, mql: str) -> Dict[str, Any]:
        """Execute MQL query on the database."""
        import json
        import re

        # Remove comments and clean the query
        mql = re.sub(r'//.*', '', mql)
        mql = mql.strip()

        # Try to detect the query type and collection
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
                        query_params = json.loads(query_str)
                    elif query_str.startswith('{'):
                        # Single document query
                        query_params = json.loads(query_str)
                    else:
                        # Try to evaluate as Python
                        query_params = eval(query_str)
                else:
                    query_params = {}
            except Exception as e:
                self.logger.error(f"Failed to parse query parameters: {e}")
                query_params = {}

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
                sample_docs = list(collection.find().limit(10))

                if not sample_docs:
                    schema_info["collections"][collection_name] = {
                        "fields": [],
                        "sample_count": 0,
                        "indexes": []
                    }
                    continue

                # Infer schema from samples
                fields = set()
                field_types = {}

                for doc in sample_docs:
                    for key, value in doc.items():
                        fields.add(key)
                        if key not in field_types:
                            field_types[key] = type(value).__name__

                # Get indexes
                indexes = []
                for index in collection.list_indexes():
                    indexes.append({
                        "name": index.get("name"),
                        "keys": list(index.get("key", {}).keys())
                    })

                # Get document count
                doc_count = collection.count_documents({})

                schema_info["collections"][collection_name] = {
                    "fields": list(fields),
                    "field_types": field_types,
                    "document_count": doc_count,
                    "sample_count": len(sample_docs),
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
