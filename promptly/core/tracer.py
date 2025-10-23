import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .utils.env import get_env_var


class UsageData(BaseModel):
    """Trace statistics"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TraceRecord(BaseModel):
    """Single trace record"""

    id: Optional[int] = None
    prompt_name: str = ""
    prompt_template: str = ""
    rendered_prompt: str = ""
    response: Optional[str] = ""
    model: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float = 0
    usage: UsageData = Field(default_factory=UsageData)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "TraceRecord":
        """Create TraceRecord from database row with proper type conversion"""
        # Convert timestamp
        timestamp = datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else datetime.now()

        # Parse JSON fields safely
        usage = UsageData.model_validate_json(row["usage"]) if row["usage"] else UsageData()
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return cls(
            id=row["id"],
            prompt_name=row["prompt_name"],
            prompt_template=row["prompt_template"],
            rendered_prompt=row["rendered_prompt"],
            response=row["response"],
            model=row["model"],
            timestamp=timestamp,
            duration_ms=row["duration_ms"],
            usage=usage,
            metadata=metadata,
            error=row["error"],
        )

    def to_db_values(self) -> tuple:
        """Convert to database values tuple"""
        return (
            self.prompt_name,
            self.prompt_template,
            self.rendered_prompt,
            self.response,
            self.model,
            self.timestamp.isoformat(),
            self.duration_ms,
            self.usage.model_dump_json(),
            json.dumps(self.metadata, default=str),
            self.error,
        )


class Tracer:
    """Simple SQLite-based tracer for LLM calls"""

    def __init__(self, db_path: Optional[str] = None, enable_tracing: Optional[bool] = None):
        if db_path is None:
            db_path = get_env_var("PROMPTLY_DB_PATH", "promptly_traces.db")
            assert db_path is not None

        self.db_path = Path(db_path)
        self._init_db()

        # Enable tracing by default unless explicitly disabled via environment variable
        if enable_tracing is not None:
            self.is_tracing_enabled = enable_tracing
        else:
            # Default to True, only disable if explicitly set to "false"
            self.is_tracing_enabled = get_env_var("PROMPTLY_TRACING_ENABLED", "true") != "false"

    def _init_db(self) -> None:
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_name TEXT,
                    prompt_template TEXT,
                    rendered_prompt TEXT,
                    response TEXT,
                    model TEXT,
                    timestamp TEXT,
                    duration_ms REAL,
                    usage TEXT,  -- JSON
                    metadata TEXT,  -- JSON
                    error TEXT
                )
            """
            )
            conn.commit()

    def log(self, record: TraceRecord) -> TraceRecord:
        """Log a trace record"""

        if not self.is_tracing_enabled:
            return record

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                INSERT INTO traces (
                    prompt_name, prompt_template, rendered_prompt, response,
                    model, timestamp, duration_ms, usage, metadata, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING *
            """,
                record.to_db_values(),
            )

            row = cursor.fetchone()
            conn.commit()
            return TraceRecord.from_db_row(row)

    def list_records(
        self,
        *,
        limit: int = 100,
        model: Optional[str] = None,
        prompt_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        optimizer_only: bool = False,
    ) -> list[TraceRecord]:
        """Query trace records"""
        query = "SELECT * FROM traces WHERE 1=1"
        params = []

        if model:
            query += " AND model = ?"
            params.append(model)

        if prompt_name:
            query += " AND prompt_name = ?"
            params.append(prompt_name)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if optimizer_only:
            query += " AND json_extract(metadata, '$.optimizer_context') IS NOT NULL"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(str(limit))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        records = []
        for row in rows:
            records.append(TraceRecord.from_db_row(row))

        return records

    def list_optimizer_records(
        self,
        *,
        limit: int = 100,
        model: Optional[str] = None,
        prompt_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        optimization_id: Optional[str] = None,
        generation: Optional[int] = None,
    ) -> list[TraceRecord]:
        """Query optimizer-specific trace records"""
        query = (
            "SELECT * FROM traces WHERE json_extract(metadata, '$.optimizer_context') IS NOT NULL"
        )
        params = []

        if model:
            query += " AND model = ?"
            params.append(model)

        if prompt_name:
            query += " AND prompt_name = ?"
            params.append(prompt_name)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if optimization_id:
            query += " AND json_extract(metadata, '$.optimizer_context.optimization_id') = ?"
            params.append(optimization_id)

        if generation is not None:
            query += " AND json_extract(metadata, '$.optimizer_context.generation') = ?"
            params.append(str(generation))

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(str(limit))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        records = []
        for row in rows:
            records.append(TraceRecord.from_db_row(row))

        return records

    def get_record(self, id: int) -> Optional[TraceRecord]:
        """Get a trace record by id"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM traces WHERE id = ?", (id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return TraceRecord.from_db_row(row)

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_calls,
                    COUNT(DISTINCT model) as unique_models,
                    AVG(duration_ms) as avg_duration,
                    SUM(json_extract(usage, '$.total_tokens')) as total_tokens,
                    GROUP_CONCAT(DISTINCT model) as models
                FROM traces
                WHERE error IS NULL
            """
            )
            row = cursor.fetchone()

            return {
                "total_calls": row[0] or 0,
                "unique_models": row[1] or 0,
                "avg_duration_ms": round(row[2] or 0, 2),
                "total_tokens": row[3] or 0,
                "models": row[4] or [],
            }
