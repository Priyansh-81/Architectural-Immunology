from src.evaluation import EvaluationRunner
from src.schemas import GaiaTask
from tests.test_graph import graph

runner = EvaluationRunner(graph)

tasks = [
    GaiaTask(
        task_id="1",
        question="What is 2+2?",
        difficulty="easy",
        expected_answer="4"
    )
]

runner.run_benchmark(tasks)

metrics = runner.compute_metrics()

print(metrics)