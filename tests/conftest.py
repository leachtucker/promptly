"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from promptly.core.clients import LLMResponse, BaseLLMClient
from promptly.core.templates import PromptTemplate, PromptMetadata
from promptly.core.tracer import Tracer, TraceRecord, UsageData



@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    import sqlite3
    from pathlib import Path

    # Get the project root directory (where conftest.py is located)
    project_root = Path(__file__).parent.parent
    
    # Create temp directory within project if it doesn't exist
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Create temporary file within project directory
    with tempfile.NamedTemporaryFile(
        suffix=".db", 
        delete=False, 
        dir=temp_dir  # This ensures it's created in our temp dir
    ) as tmp:
        db_path = tmp.name
        os.environ["PROMPTLY_DB_PATH"] = db_path
    
    # Verify database can be created and is functional
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_init (id INTEGER)")
        conn.commit()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing"""
    return LLMResponse(
        content="This is a test response",
        model="gpt-3.5-turbo",
        usage=UsageData(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        metadata={"trace_id": "test-trace-123"},
    )


@pytest.fixture
def sample_prompt_template():
    """Sample prompt template for testing"""
    return PromptTemplate(
        template="Hello {{ name }}, how are you today?",
        name="greeting",
        metadata=PromptMetadata(
            name="greeting",
            description="A simple greeting template",
            tags=["greeting", "test"],
        ),
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = AsyncMock(spec=BaseLLMClient)
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content="Mock response", model="test-model", usage=UsageData(total_tokens=10)
        )
    )
    client.get_available_models = MagicMock(return_value=["test-model"])
    return client


@pytest.fixture
def tracer_with_temp_db(temp_db):
    """Tracer instance with temporary database"""
    return Tracer(db_path=temp_db)


@pytest.fixture
def sample_trace_record():
    """Sample trace record for testing"""
    return TraceRecord(
        prompt_name="test_prompt",
        prompt_template="Hello {{ name }}",
        rendered_prompt="Hello World",
        response="Hello! How can I help you?",
        model="gpt-3.5-turbo",
        duration_ms=150.5,
        usage=UsageData(total_tokens=20),
        metadata={"test": True},
    )
