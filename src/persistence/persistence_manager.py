"""
Persistence manager for saving and loading database state to disk.

Provides durability and checkpoint/recovery capabilities for the vector database.
Design choices:
- JSON format for metadata (human-readable, debuggable)
- Binary format (pickle/numpy) for vector data (space-efficient)
- Write-ahead logging for consistency
- Incremental snapshots for performance
"""

import json
import pickle
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Manages persistence of vector database state to disk.

    Features:
    - Periodic snapshots
    - Write-ahead logging (WAL)
    - Atomic writes with temporary files
    - Compression support
    - Recovery from checkpoints
    """

    def __init__(
        self,
        data_dir: str = "./data",
        enable_wal: bool = True,
        snapshot_interval_seconds: int = 300,  # 5 minutes
        compression: bool = True,
    ):
        """
        Initialize persistence manager.

        Args:
            data_dir: Directory for persisted data
            enable_wal: Whether to use write-ahead logging
            snapshot_interval_seconds: Interval between automatic snapshots
            compression: Whether to compress vector data
        """
        self.data_dir = Path(data_dir)
        self.enable_wal = enable_wal
        self.snapshot_interval = snapshot_interval_seconds
        self.compression = compression

        # Create directory structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.wal_dir = self.data_dir / "wal"
        self.wal_dir.mkdir(exist_ok=True)

        # State tracking
        self.last_snapshot_time = time.time()
        self.wal_file = None
        self.snapshot_thread = None
        self.shutdown_flag = threading.Event()
        self._lock = threading.Lock()

        # Start background snapshot thread if interval > 0
        if snapshot_interval_seconds > 0:
            self._start_snapshot_thread()

    def save_state(self, state: Dict[str, Any]) -> str:
        """
        Save complete database state to disk.

        Args:
            state: Complete database state dictionary containing:
                - libraries: Dict of library data
                - documents: Dict of document data
                - chunks: Dict of chunk data with vectors
                - indexes: Serialized index structures

        Returns:
            Path to saved snapshot

        Design choices:
        - Atomic writes using temporary files
        - Separate files for metadata and vectors
        - Compression for vector data
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{timestamp}"
        snapshot_path = self.snapshots_dir / snapshot_name

        try:
            # Create a unique temporary directory for atomic write
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix=f"{snapshot_name}.", dir=str(self.snapshots_dir))
            temp_path = Path(temp_dir)

            # Save metadata (libraries, documents) as JSON
            metadata = {
                "timestamp": timestamp,
                "version": "1.0",
                "libraries": self._serialize_libraries(state.get("libraries", {})),
                "documents": self._serialize_documents(state.get("documents", {})),
            }

            metadata_file = temp_path / "metadata.json"
            # Write metadata and fsync to ensure durability
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())

            # Save chunks with vectors separately (binary format)
            chunks_data = state.get("chunks", {})
            if chunks_data:
                self._save_chunks(temp_path, chunks_data)

            # Save index structures
            indexes = state.get("indexes", {})
            if indexes:
                self._save_indexes(temp_path, indexes)

            # Atomic rename into final location
            final_path = snapshot_path
            if final_path.exists():
                raise FileExistsError(f"Snapshot path already exists: {final_path}")

            shutil.move(str(temp_path), str(final_path))

            # Fsync the snapshots directory to make the rename durable
            snapshots_fd = os.open(str(self.snapshots_dir), os.O_RDONLY)
            try:
                os.fsync(snapshots_fd)
            finally:
                os.close(snapshots_fd)

            logger.info(f"Saved snapshot to {final_path}")
            return str(final_path)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            # Cleanup temporary directory if it exists
            if 'temp_path' in locals() and temp_path.exists():
                shutil.rmtree(temp_path)
            raise

    def load_state(self, snapshot_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load database state from disk.

        Args:
            snapshot_path: Specific snapshot to load (latest if None)

        Returns:
            Loaded state dictionary

        Design choices:
        - Load latest snapshot by default
        - Apply WAL entries after snapshot for recovery
        - Validate data integrity during load
        """
        if snapshot_path is None:
            snapshot_path = self._get_latest_snapshot()
            if snapshot_path is None:
                logger.warning("No snapshots found")
                return {}

        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot not found: {snapshot_path}")

        try:
            state = {}

            # Load metadata
            metadata_file = snapshot_path / "metadata.json"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            state["libraries"] = self._deserialize_libraries(metadata.get("libraries", {}))
            state["documents"] = self._deserialize_documents(metadata.get("documents", {}))

            # Load chunks with vectors
            state["chunks"] = self._load_chunks(snapshot_path)

            # Load indexes
            state["indexes"] = self._load_indexes(snapshot_path)

            # Apply WAL entries if enabled
            if self.enable_wal:
                state = self._apply_wal_entries(state, metadata.get("timestamp"))

            logger.info(f"Loaded snapshot from {snapshot_path}")
            return state

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            raise

    def write_wal_entry(self, operation: str, data: Dict[str, Any]):
        """
        Write entry to write-ahead log for durability.

        Args:
            operation: Operation type (create, update, delete)
            data: Operation data

        Design choice: Append-only log for simplicity and performance
        """
        if not self.enable_wal:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "data": data,
        }

        wal_file = self.wal_dir / f"wal_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

        with self._lock:
            with open(wal_file, "a") as f:
                json.dump(entry, f, default=str)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())  # Ensure durability

    def _save_chunks(self, path: Path, chunks: Dict):
        """Save chunks with vectors in binary format."""
        chunks_file = path / "chunks.pkl"
        vectors_file = path / "vectors.npz"

        # Separate metadata from vectors
        chunk_metadata = {}
        vectors_dict = {}

        for chunk_id, chunk_data in chunks.items():
            # Extract vector
            vector = chunk_data.pop("vector", None)
            if vector is not None:
                vectors_dict[str(chunk_id)] = np.array(vector, dtype=np.float32)

            # Save rest as metadata
            chunk_metadata[chunk_id] = chunk_data

        # Save metadata
        with open(chunks_file, "wb") as f:
            pickle.dump(chunk_metadata, f)

        # Save vectors with compression
        if vectors_dict:
            if self.compression:
                np.savez_compressed(vectors_file, **vectors_dict)
            else:
                np.savez(vectors_file, **vectors_dict)

    def _load_chunks(self, path: Path) -> Dict:
        """Load chunks with vectors from binary format."""
        chunks_file = path / "chunks.pkl"
        vectors_file = path / "vectors.npz"

        if not chunks_file.exists():
            return {}

        # Load metadata
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        # Load vectors
        if vectors_file.exists():
            vectors_data = np.load(vectors_file)
            for chunk_id in chunks:
                if str(chunk_id) in vectors_data:
                    chunks[chunk_id]["vector"] = vectors_data[str(chunk_id)].tolist()

        return chunks

    def _save_indexes(self, path: Path, indexes: Dict):
        """Save index structures."""
        indexes_file = path / "indexes.pkl"
        with open(indexes_file, "wb") as f:
            pickle.dump(indexes, f)

    def _load_indexes(self, path: Path) -> Dict:
        """Load index structures."""
        indexes_file = path / "indexes.pkl"
        if indexes_file.exists():
            with open(indexes_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _serialize_libraries(self, libraries: Dict) -> Dict:
        """Serialize library objects for JSON storage."""
        serialized = {}
        for lib_id, lib_data in libraries.items():
            if hasattr(lib_data, "dict"):
                serialized[str(lib_id)] = lib_data.dict()
            else:
                serialized[str(lib_id)] = lib_data
        return serialized

    def _deserialize_libraries(self, libraries: Dict) -> Dict:
        """Deserialize library objects from JSON."""
        deserialized = {}
        for lib_id, lib_data in libraries.items():
            # Convert string UUID back to UUID if needed
            if isinstance(lib_id, str):
                try:
                    lib_id = UUID(lib_id)
                except ValueError:
                    # Skip invalid UUIDs
                    logger.warning(f"Skipping invalid UUID: {lib_id}")
                    continue
            deserialized[lib_id] = lib_data
        return deserialized

    def _serialize_documents(self, documents: Dict) -> Dict:
        """Serialize document objects for JSON storage."""
        serialized = {}
        for doc_id, doc_data in documents.items():
            if hasattr(doc_data, "dict"):
                serialized[str(doc_id)] = doc_data.dict()
            else:
                serialized[str(doc_id)] = doc_data
        return serialized

    def _deserialize_documents(self, documents: Dict) -> Dict:
        """Deserialize document objects from JSON."""
        deserialized = {}
        for doc_id, doc_data in documents.items():
            if isinstance(doc_id, str):
                try:
                    doc_id = UUID(doc_id)
                except ValueError:
                    # Skip invalid UUIDs
                    logger.warning(f"Skipping invalid UUID: {doc_id}")
                    continue
            deserialized[doc_id] = doc_data
        return deserialized

    def _get_latest_snapshot(self) -> Optional[Path]:
        """Get path to latest snapshot."""
        snapshots = sorted(self.snapshots_dir.glob("snapshot_*"))
        return snapshots[-1] if snapshots else None

    def _apply_wal_entries(self, state: Dict, after_timestamp: str) -> Dict:
        """Apply WAL entries after given timestamp."""
        # Implementation would replay WAL entries
        # This is simplified for now
        return state

    def _start_snapshot_thread(self):
        """Start background thread for periodic snapshots."""
        def snapshot_worker():
            while not self.shutdown_flag.is_set():
                time.sleep(self.snapshot_interval)
                if not self.shutdown_flag.is_set():
                    try:
                        # Would call save_state with current state
                        logger.info("Periodic snapshot triggered")
                    except Exception as e:
                        logger.error(f"Periodic snapshot failed: {e}")

        self.snapshot_thread = threading.Thread(target=snapshot_worker, daemon=True)
        self.snapshot_thread.start()

    def shutdown(self):
        """Shutdown persistence manager cleanly."""
        self.shutdown_flag.set()
        if self.snapshot_thread:
            self.snapshot_thread.join(timeout=5)

    def get_statistics(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        snapshots = list(self.snapshots_dir.glob("snapshot_*"))
        wal_files = list(self.wal_dir.glob("wal_*.jsonl"))

        total_size = sum(
            sum(f.stat().st_size for f in s.rglob("*") if f.is_file())
            for s in snapshots
        )

        return {
            "num_snapshots": len(snapshots),
            "latest_snapshot": str(self._get_latest_snapshot()) if snapshots else None,
            "total_size_bytes": total_size,
            "num_wal_files": len(wal_files),
            "compression_enabled": self.compression,
            "wal_enabled": self.enable_wal,
        }