from src.schemas import GaiaTask, AgentState

task = GaiaTask(
    task_id="1",
    question="What is 2+2?",
    difficulty="easy",
    expected_answer="4"
)

state = AgentState(task=task)

print(task)
print(state)