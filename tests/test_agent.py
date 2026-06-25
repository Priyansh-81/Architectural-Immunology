from src.schemas import *
from src.memory import *
from src.agents import *
from src.mock_models import load_models

small_model, reasoning_model = load_models()

memory = MemoryManager()

planner = PlannerAgent(
    "planner",
    small_model,
    memory
)

task = GaiaTask(
    task_id="1",
    question="What is 2+2?",
    difficulty="easy"
)

state = AgentState(task=task)

result = planner.run(state)

print(result)