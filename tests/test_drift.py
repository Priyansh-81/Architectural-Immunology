from src.schemas import *
from src.memory import *
from src.drift import *

task = GaiaTask(
    task_id="1",
    question="What is 2+2?",
    difficulty="easy"
)

state = AgentState(task=task)

state.planner_output = AgentOutput(
    agent_name="planner",
    content="Need to add two numbers"
)

memory = MemoryManager()
monitor = DriftMonitor(memory)

metrics = monitor.compute(state)

print(metrics)