"""
Tests for Tracer and TraceRecord
"""

import sqlite3
from datetime import datetime

from promptly.core.tracer import Tracer, TraceRecord, UsageData


class TestTraceRecord:
    """Test TraceRecord dataclass functionality"""

    def test_trace_record_creation(self):
        """Test basic TraceRecord creation"""
        record = TraceRecord(
            prompt_name="test_prompt",
            prompt_template="Hello {{ name }}",
            rendered_prompt="Hello World",
            response="Hello! How can I help?",
            model="gpt-3.5-turbo",
            duration_ms=150.5,
        )

        assert record.prompt_name == "test_prompt"
        assert record.prompt_template == "Hello {{ name }}"
        assert record.rendered_prompt == "Hello World"
        assert record.response == "Hello! How can I help?"
        assert record.model == "gpt-3.5-turbo"
        assert record.duration_ms == 150.5
        assert record.usage == UsageData()
        assert record.metadata == {}
        assert isinstance(record.timestamp, datetime)

    def test_trace_record_defaults(self):
        """Test TraceRecord with defaults"""
        record = TraceRecord()

        assert record.prompt_name == ""
        assert record.rendered_prompt == ""
        assert record.response == ""
        assert record.model == ""
        assert record.duration_ms == 0
        assert record.usage == UsageData()
        assert record.metadata == {}
        assert record.error is None
        assert isinstance(record.timestamp, datetime)

    def test_trace_record_with_usage(self):
        """Test TraceRecord with usage information"""
        usage = UsageData(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        record = TraceRecord(prompt_name="test", usage=usage)

        assert record.usage == usage

    def test_trace_record_with_metadata(self):
        """Test TraceRecord with metadata"""
        metadata = {"trace_id": "123", "custom": "value"}
        record = TraceRecord(prompt_name="test", metadata=metadata)

        assert record.metadata == metadata


class TestTracer:
    """Test Tracer database functionality"""

    def test_tracer_initialization(self, temp_db):
        """Test Tracer initialization with temporary database"""
        print(f"Creating tracer with temporary database at {temp_db}")
        tracer = Tracer(db_path=temp_db)

        assert tracer.db_path.exists()

        # Verify database schema was created
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='traces'")
            assert cursor.fetchone() is not None

    def test_tracer_log_record(self, tracer_with_temp_db, sample_trace_record):
        """Test logging a trace record"""
        tracer = tracer_with_temp_db

        # Log the record
        record = tracer.log(sample_trace_record)

        assert record is not None
        assert isinstance(record.id, int)

    def test_tracer_get_record(self, tracer_with_temp_db, sample_trace_record):
        """Test retrieving a trace record"""
        tracer = tracer_with_temp_db

        # Log a record
        record = tracer.log(sample_trace_record)

        # Retrieve the record
        retrieved_record = tracer.get_record(record.id)

        assert retrieved_record is not None
        assert retrieved_record.prompt_name == sample_trace_record.prompt_name
        assert retrieved_record.model == sample_trace_record.model
        assert retrieved_record.duration_ms == sample_trace_record.duration_ms

    def test_tracer_list_records(self, tracer_with_temp_db):
        """Test listing trace records"""
        tracer = tracer_with_temp_db

        # Log multiple records
        record1 = TraceRecord(prompt_name="test1", model="gpt-3.5-turbo", response="Response 1")
        record2 = TraceRecord(prompt_name="test2", model="gpt-4", response="Response 2")

        tracer.log(record1)
        tracer.log(record2)

        # List all records
        records = tracer.list_records()
        assert len(records) == 2

        # List with limit
        records_limited = tracer.list_records(limit=1)
        assert len(records_limited) == 1

    def test_tracer_list_records_with_filters(self, tracer_with_temp_db):
        """Test listing trace records with filters"""
        tracer = tracer_with_temp_db

        # Log records with different models
        record1 = TraceRecord(prompt_name="test1", model="gpt-3.5-turbo", response="Response 1")
        record2 = TraceRecord(prompt_name="test2", model="gpt-4", response="Response 2")

        tracer.log(record1)
        tracer.log(record2)

        # Filter by model
        gpt4_records = tracer.list_records(model="gpt-4")
        assert len(gpt4_records) == 1
        assert gpt4_records[0].model == "gpt-4"

        # Filter by prompt name
        test1_records = tracer.list_records(prompt_name="test1")
        assert len(test1_records) == 1
        assert test1_records[0].prompt_name == "test1"

    def test_tracer_get_stats(self, tracer_with_temp_db):
        """Test getting tracer statistics"""
        tracer = tracer_with_temp_db

        # Log some records
        record1 = TraceRecord(
            prompt_name="test1",
            model="gpt-3.5-turbo",
            duration_ms=100,
            usage=UsageData(total_tokens=10),
        )
        record2 = TraceRecord(
            prompt_name="test2",
            model="gpt-4",
            duration_ms=200,
            usage=UsageData(total_tokens=20),
        )

        tracer.log(record1)
        tracer.log(record2)

        # Get stats
        stats = tracer.get_stats()

        assert stats["total_calls"] == 2
        assert stats["total_tokens"] == 30
        assert stats["avg_duration_ms"] == 150.0
        assert "gpt-3.5-turbo" in stats["models"]
        assert "gpt-4" in stats["models"]

    def test_tracer_error_handling(self, tracer_with_temp_db):
        """Test tracer error handling"""
        tracer = tracer_with_temp_db

        # Test logging record with error
        error_record = TraceRecord(
            prompt_name="error_test", model="gpt-3.5-turbo", error="Test error message"
        )

        record = tracer.log(error_record)
        assert record is not None

        # Retrieve and verify error is stored
        retrieved = tracer.get_record(record.id)
        assert retrieved.error == "Test error message"
