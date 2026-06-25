from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import math


# ============================================================
# Memory Chunk
# ============================================================

@dataclass
class MemoryChunk:
    content: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Short-Term Memory
# ============================================================

class ShortTermMemory:
    """
    Stores recent context window.
    """

    def __init__(self, max_chunks: int = 20):
        self.max_chunks = max_chunks
        self.chunks: List[MemoryChunk] = []

    def add(self, chunk: MemoryChunk):
        self.chunks.append(chunk)

        if len(self.chunks) > self.max_chunks:
            self.chunks.pop(0)

    def get_context(self) -> str:
        return "\n".join(chunk.content for chunk in self.chunks)

    def clear(self):
        self.chunks = []


# ============================================================
# Episodic Memory
# ============================================================

class EpisodicMemory:
    """
    Stores full execution history.
    """

    def __init__(self):
        self.events: List[MemoryChunk] = []

    def add_event(self, chunk: MemoryChunk):
        self.events.append(chunk)

    def retrieve_recent(self, n: int = 5) -> List[MemoryChunk]:
        return self.events[-n:]

    def get_all(self) -> List[MemoryChunk]:
        return self.events


# ============================================================
# Long-Term Memory
# ============================================================

class LongTermMemory:
    """
    Stores persistent knowledge across tasks.
    """

    def __init__(self):
        self.store: Dict[str, Any] = {}

    def save(self, key: str, value: Any):
        self.store[key] = value

    def load(self, key: str):
        return self.store.get(key)

    def keys(self):
        return list(self.store.keys())


# ============================================================
# Memory Manager
# ============================================================

class MemoryManager:
    """
    Unified interface over all memory systems.
    """

    def __init__(self):
        self.stm = ShortTermMemory()
        self.episodic = EpisodicMemory()
        self.ltm = LongTermMemory()

    def add_to_short_term(
        self,
        content: str,
        source: str,
        importance: float = 1.0
    ):
        chunk = MemoryChunk(
            content=content,
            source=source,
            importance=importance
        )
        self.stm.add(chunk)

    def add_to_episodic(
        self,
        content: str,
        source: str,
        importance: float = 1.0
    ):
        chunk = MemoryChunk(
            content=content,
            source=source,
            importance=importance
        )
        self.episodic.add_event(chunk)

    def context_window(self) -> str:
        return self.stm.get_context()

    def prune_memory(
        self,
        importance_threshold: float = 0.5
    ):
        """
        Remove low-importance chunks.
        """
        filtered = []

        for chunk in self.stm.chunks:
            if chunk.importance >= importance_threshold:
                filtered.append(chunk)

        self.stm.chunks = filtered

    def context_pollution_ratio(self) -> float:
        """
        Approximate irrelevant context ratio.
        """
        if not self.stm.chunks:
            return 0.0

        irrelevant = 0

        for chunk in self.stm.chunks:
            if chunk.importance < 0.5:
                irrelevant += 1

        return irrelevant / len(self.stm.chunks)

    def hard_reset(self):
        """
        Used during immune intervention.
        """
        self.stm.clear()