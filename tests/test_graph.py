from src.schemas import *
from src.memory import *
from src.agents import *
from src.graph import *
from src.drift import *
from src.immune import *
from src.mock_models import load_models

small_model, reasoning_model = load_models()

memory = MemoryManager()

planner = PlannerAgent(
    "planner",
    small_model,
    memory
)

retriever = RetrieverAgent(
    "retriever",
    small_model,
    memory
)

reasoner = ReasonerAgent(
    "reasoner",
    reasoning_model,
    memory
)

validator = ValidatorAgent(
    "validator",
    small_model,
    memory
)

drift = DriftMonitor(memory)
immune = ImmuneSystem(memory)

graph = GaiaAgentGraph(
    planner,
    retriever,
    reasoner,
    validator,
    drift,
    immune
)

task = GaiaTask(
    task_id="1",
    question="What is 2+2?",
    difficulty="easy",
    expected_answer="4"
)

state = AgentState(task=task)

final_state = graph.execute(state)

print(final_state.validator_output)