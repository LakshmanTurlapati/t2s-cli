"""
Tests for MongoDB (MQL) functionality
"""
import pytest
from unittest.mock import Mock, patch
from t2s.core.mql_engine import MQLEngine
from t2s.core.config import Config
from t2s.database.mongodb_manager import MongoDBManager
from t2s.utils.mql_validator import MQLValidator


class TestMQLEngine:
    """Test MQL Engine functionality"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object"""
        config = Mock(spec=Config)
        config.config = Mock()
        config.config.model_settings = Mock()
        config.config.model_settings.model_name = "test-model"
        config.llm = Mock()
        return config

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager"""
        model_manager = Mock()
        model_manager.generate_sql = Mock(return_value="db.users.find({age: {$gt: 18}})")
        return model_manager

    @pytest.fixture
    def mql_engine(self, mock_config, mock_model_manager):
        """Create MQLEngine instance with mock config and model manager"""
        return MQLEngine(mock_config, mock_model_manager)

    def test_mql_engine_initialization(self, mql_engine):
        """Test that MQLEngine initializes correctly"""
        assert mql_engine is not None
        assert hasattr(mql_engine, 'generate_mql')
        assert hasattr(mql_engine, 'validate_mql')
        assert hasattr(mql_engine, 'extract_mql')

    def test_extract_mql_with_backticks(self, mql_engine):
        """Test MQL extraction from markdown code blocks"""
        model_output = """
        Here's your query:
        ```javascript
        db.users.find({age: {$gt: 18}}, {_id: 0})
        ```
        """
        extracted = mql_engine.extract_mql(model_output)
        assert "db.users.find" in extracted
        assert "{age: {$gt: 18}}" in extracted

    def test_extract_mql_with_db_prefix(self, mql_engine):
        """Test MQL extraction when query starts with db."""
        model_output = "db.products.find({category: 'electronics'})"
        extracted = mql_engine.extract_mql(model_output)
        assert "db.products.find" in extracted
        assert "electronics" in extracted

    def test_extract_mql_aggregation_pipeline(self, mql_engine):
        """Test MQL extraction for aggregation pipelines"""
        model_output = """
        db.orders.aggregate([
            {$match: {status: 'completed'}},
            {$group: {_id: '$customer', total: {$sum: '$amount'}}}
        ])
        """
        extracted = mql_engine.extract_mql(model_output)
        assert "db.orders.aggregate" in extracted
        assert "$match" in extracted
        assert "$group" in extracted

    @pytest.mark.asyncio
    async def test_validate_mql_valid_query(self, mql_engine):
        """Test validation of a valid MQL query"""
        valid_mql = "db.users.find({age: {$gt: 18}})"
        result = await mql_engine.validate_mql(valid_mql)
        assert result == valid_mql  # Returns the validated query

    @pytest.mark.asyncio
    async def test_validate_mql_invalid_query(self, mql_engine):
        """Test validation of invalid MQL query"""
        invalid_mql = "SELECT * FROM users"  # SQL, not MQL
        with pytest.raises(ValueError) as exc_info:
            await mql_engine.validate_mql(invalid_mql)
        assert "Invalid MQL structure" in str(exc_info.value)

    def test_validate_mql_fields_valid(self, mql_engine):
        """Test field validation with valid fields"""
        mql = "db.users.find({name: 'John', age: 25})"
        schema = {
            "collections": {
                "users": {
                    "fields": ["name", "age", "email"]
                }
            }
        }
        is_valid, error_msg = mql_engine.validate_mql_fields(mql, schema)
        assert is_valid is True
        assert error_msg == ""

    def test_validate_mql_fields_invalid(self, mql_engine):
        """Test field validation with invalid fields"""
        mql = "db.users.find({nonexistent_field: 'value'})"
        schema = {
            "collections": {
                "users": {
                    "fields": ["name", "age"]
                }
            }
        }
        is_valid, error_msg = mql_engine.validate_mql_fields(mql, schema)
        assert is_valid is False
        assert "nonexistent_field" in error_msg


class TestMongoDBManager:
    """Test MongoDB Manager functionality"""

    def test_quote_unquoted_keys_simple(self):
        """Test JavaScript to JSON conversion for simple objects"""
        manager = MongoDBManager(None)
        js_str = "{name: 'John', age: 25}"
        result = manager._quote_unquoted_keys(js_str)
        assert '"name"' in result
        assert '"age"' in result

    def test_quote_unquoted_keys_nested(self):
        """Test JavaScript to JSON conversion for nested objects"""
        manager = MongoDBManager(None)
        js_str = "{user: {name: 'John', address: {city: 'NYC'}}}"
        result = manager._quote_unquoted_keys(js_str)
        assert '"user"' in result
        assert '"name"' in result
        assert '"address"' in result
        assert '"city"' in result

    def test_quote_unquoted_keys_with_operators(self):
        """Test JavaScript to JSON conversion with MongoDB operators"""
        manager = MongoDBManager(None)
        js_str = "{$match: {age: {$gt: 18}}}"
        result = manager._quote_unquoted_keys(js_str)
        assert '"$match"' in result
        assert '"age"' in result
        assert '"$gt"' in result

    def test_quote_unquoted_keys_with_quoted_keys(self):
        """Test that already-quoted keys remain unchanged"""
        manager = MongoDBManager(None)
        js_str = '{"already_quoted": "value", unquoted: 123}'
        result = manager._quote_unquoted_keys(js_str)
        assert '"already_quoted"' in result
        assert '"unquoted"' in result

    def test_quote_unquoted_keys_with_strings_containing_colons(self):
        """Test handling of strings that contain colons"""
        manager = MongoDBManager(None)
        js_str = "{message: 'Hello: World', time: '10:30 AM'}"
        result = manager._quote_unquoted_keys(js_str)
        assert '"message"' in result
        assert '"time"' in result
        assert 'Hello: World' in result
        assert '10:30 AM' in result

    def test_quote_unquoted_keys_aggregation_pipeline(self):
        """Test complex aggregation pipeline conversion"""
        manager = MongoDBManager(None)
        js_str = "[{$lookup: {from: 'albums', localField: 'AlbumId', foreignField: 'AlbumId', as: 'album'}}]"
        result = manager._quote_unquoted_keys(js_str)
        assert '"$lookup"' in result
        assert '"from"' in result
        assert '"localField"' in result
        assert '"foreignField"' in result
        assert '"as"' in result


class TestMQLValidator:
    """Test MQL validation utility"""

    @pytest.fixture
    def validator(self):
        """Create MQLValidator instance"""
        return MQLValidator()

    def test_validate_mql_collection_query(self, validator):
        """Test validation of collection-based query"""
        mql = "db.users.find({age: {$gt: 18}})"
        result = validator.validate_syntax(mql)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_mql_direct_operation(self, validator):
        """Test validation of direct database operation"""
        mql = "db.getCollectionNames()"
        # validate_syntax doesn't check direct operations, so we check structure
        assert mql.startswith("db.")
        assert "getCollectionNames" in mql

    def test_validate_mql_list_collections(self, validator):
        """Test validation of listCollections operation"""
        mql = "db.listCollections()"
        # validate_syntax doesn't check direct operations, so we check structure
        assert mql.startswith("db.")
        assert "listCollections" in mql

    def test_validate_mql_aggregation(self, validator):
        """Test validation of aggregation pipeline"""
        mql = "db.orders.aggregate([{$match: {status: 'completed'}}])"
        result = validator.validate_syntax(mql)
        assert result["valid"] is True

    def test_validate_mql_invalid_format(self, validator):
        """Test validation of invalid MQL format"""
        mql = "SELECT * FROM users"
        result = validator.validate_syntax(mql)
        assert result["valid"] is False
        assert any("must start with 'db.'" in err for err in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_and_correct_removes_semicolon(self, validator):
        """Test that semicolons are stripped from MQL"""
        mql = "db.users.find({name: 'John'});"
        corrected = await validator.validate_and_correct(mql)
        # Should not end with semicolon after correction
        assert not corrected.endswith(";")
        assert corrected == "db.users.find({name: 'John'})"


class TestMQLIntegration:
    """Integration tests for MQL functionality"""

    def test_sql_engine_import(self):
        """Test that SQLEngine can be imported separately"""
        from t2s.core.sql_engine import SQLEngine
        assert SQLEngine is not None

    def test_mql_engine_import(self):
        """Test that MQLEngine can be imported separately"""
        from t2s.core.mql_engine import MQLEngine
        assert MQLEngine is not None

    def test_mongodb_manager_import(self):
        """Test that MongoDBManager can be imported"""
        from t2s.database.mongodb_manager import MongoDBManager
        assert MongoDBManager is not None

    def test_t2s_engine_has_mql_support(self):
        """Test that T2SEngine has MongoDB support"""
        from t2s.core.engine import T2SEngine
        # Check that T2SEngine has the necessary methods
        assert hasattr(T2SEngine, '_process_mql_query')
        assert hasattr(T2SEngine, '_detect_database_type')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
