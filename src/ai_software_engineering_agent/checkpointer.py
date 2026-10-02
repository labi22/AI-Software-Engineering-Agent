"""Postgres-backed durable LangGraph Checkpointer."""
from __future__ import annotations

import logging
import random
from typing import Any, AsyncIterator, Iterator, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    WRITES_IDX_MAP,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class PostgresCheckpointSaver(BaseCheckpointSaver):
    """Durable PostgreSQL-backed checkpointer for LangGraph agent workflows."""

    def __init__(self, database_url: str) -> None:
        super().__init__()
        self.database_url = database_url
        self._initialized = False

    def _conn(self):
        import psycopg
        return psycopg.connect(self.database_url, autocommit=True)

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT NOT NULL,
                    checkpoint BYTEA NOT NULL,
                    metadata BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BYTEA NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BYTEA NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """
            )
        self._initialized = True

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self.initialize()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with self._conn() as conn, conn.cursor() as cur:
            if checkpoint_id:
                cur.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s;
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                cur.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s
                    ORDER BY checkpoint_id DESC
                    LIMIT 1;
                    """,
                    (thread_id, checkpoint_ns),
                )
            row = cur.fetchone()
            if not row:
                return None

            cid, parent_cid, c_type, c_bytes, m_bytes = row
            checkpoint = self.serde.loads_typed((c_type, bytes(c_bytes)))
            metadata = self.serde.loads_typed(("json", bytes(m_bytes)))

            # Fetch channel blobs
            values = {}
            for channel, version in checkpoint.get("channel_versions", {}).items():
                cur.execute(
                    """
                    SELECT type, blob FROM checkpoint_blobs
                    WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s AND version = %s;
                    """,
                    (thread_id, checkpoint_ns, channel, str(version)),
                )
                blob_row = cur.fetchone()
                if blob_row:
                    values[channel] = self.serde.loads_typed((blob_row[0], bytes(blob_row[1])))

            checkpoint["channel_values"] = values

            # Fetch pending writes
            cur.execute(
                """
                SELECT task_id, channel, type, blob FROM checkpoint_writes
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s;
                """,
                (thread_id, checkpoint_ns, cid),
            )
            writes = [
                (w[0], w[1], self.serde.loads_typed((w[2], bytes(w[3]))))
                for w in cur.fetchall()
            ]

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cid,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_cid,
                    }
                }
                if parent_cid
                else None
            ),
            pending_writes=writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        self.initialize()
        if not config:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        query = """
            SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = %s AND checkpoint_ns = %s
        """
        params: list[Any] = [thread_id, checkpoint_ns]
        if before and "checkpoint_id" in before.get("configurable", {}):
            query += " AND checkpoint_id < %s"
            params.append(before["configurable"]["checkpoint_id"])

        query += " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        for row in rows:
            cid, parent_cid, c_type, c_bytes, m_bytes = row
            checkpoint = self.serde.loads_typed((c_type, bytes(c_bytes)))
            metadata = self.serde.loads_typed(("json", bytes(m_bytes)))
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cid,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_cid,
                        }
                    }
                    if parent_cid
                    else None
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self.initialize()
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values", {})
        parent_cid = config["configurable"].get("checkpoint_id")
        cid = checkpoint["id"]

        with self._conn() as conn, conn.cursor() as cur:
            # Save channel blobs
            for k, v in new_versions.items():
                type_name, blob_bytes = (
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", b"")
                )
                cur.execute(
                    """
                    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO UPDATE SET
                        type = EXCLUDED.type,
                        blob = EXCLUDED.blob;
                    """,
                    (thread_id, checkpoint_ns, k, str(v), type_name, blob_bytes),
                )

            # Save checkpoint
            c_type, c_bytes = self.serde.dumps_typed(c)
            m_type, m_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
            cur.execute(
                """
                INSERT INTO checkpoints (
                    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    type, checkpoint, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                    parent_checkpoint_id = EXCLUDED.parent_checkpoint_id,
                    type = EXCLUDED.type,
                    checkpoint = EXCLUDED.checkpoint,
                    metadata = EXCLUDED.metadata;
                """,
                (thread_id, checkpoint_ns, cid, parent_cid, c_type, c_bytes, m_bytes),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cid,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.initialize()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._conn() as conn, conn.cursor() as cur:
            for idx, (c, v) in enumerate(writes):
                idx_key = WRITES_IDX_MAP.get(c, idx)
                type_name, blob_bytes = self.serde.dumps_typed(v)
                cur.execute(
                    """
                    INSERT INTO checkpoint_writes (
                        thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                        channel, type, blob, task_path
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING;
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx_key,
                        c,
                        type_name,
                        blob_bytes,
                        task_path,
                    ),
                )

    def delete_thread(self, thread_id: str) -> None:
        self.initialize()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (thread_id,))
            cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s;", (thread_id,))
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (thread_id,))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


def create_checkpointer(database_url: str | None = None) -> BaseCheckpointSaver:
    """Factory creating PostgresCheckpointSaver if database_url is provided, else MemorySaver."""
    if database_url:
        return PostgresCheckpointSaver(database_url)
    return MemorySaver()
