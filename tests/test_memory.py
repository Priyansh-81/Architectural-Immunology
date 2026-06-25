from src.memory import MemoryManager

memory = MemoryManager()

memory.add_to_short_term(
    content="Revenue = 100",
    source="retriever",
    importance=0.9
)

memory.add_to_short_term(
    content="Noise chunk",
    source="tool",
    importance=0.2
)

print(memory.context_window())
print(memory.context_pollution_ratio())