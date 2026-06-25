import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from src.schemas import GaiaTask
from src.config import CONFIG


class GaiaLoader:
    """
    Loads GAIA benchmark tasks and converts them into GaiaTask objects.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or CONFIG.paths.data_dir

    def load_json(self, filename: str) -> List[GaiaTask]:
        """
        Load GAIA dataset from JSON file.
        """
        filepath = self.data_dir / filename

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        tasks = [self._parse_task(item) for item in raw_data]
        return tasks

    def load_jsonl(self, filename: str) -> List[GaiaTask]:
        """
        Load GAIA dataset from JSONL file.
        """
        filepath = self.data_dir / filename
        tasks = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                tasks.append(self._parse_task(raw))

        return tasks

    def _parse_task(self, raw: Dict) -> GaiaTask:
        """
        Normalize raw GAIA task fields.
        """

        task_id = str(
            raw.get("task_id")
            or raw.get("id")
            or raw.get("TaskID")
            or "unknown"
        )

        question = (
            raw.get("question")
            or raw.get("Question")
            or ""
        )

        difficulty = (
            raw.get("difficulty")
            or raw.get("level")
            or raw.get("Level")
            or "unknown"
        )

        expected_answer = (
            raw.get("answer")
            or raw.get("expected_answer")
            or raw.get("Final answer")
        )

        metadata = {
            k: v for k, v in raw.items()
            if k not in {
                "task_id",
                "id",
                "TaskID",
                "question",
                "Question",
                "difficulty",
                "level",
                "Level",
                "answer",
                "expected_answer",
                "Final answer"
            }
        }

        return GaiaTask(
            task_id=task_id,
            question=question,
            difficulty=difficulty,
            expected_answer=expected_answer,
            metadata=metadata
        )

    def filter_by_difficulty(
        self,
        tasks: List[GaiaTask],
        difficulty: str
    ) -> List[GaiaTask]:
        """
        Return tasks matching difficulty.
        """
        return [
            task for task in tasks
            if task.difficulty.lower() == difficulty.lower()
        ]

    def sample_tasks(
        self,
        tasks: List[GaiaTask],
        n: int,
        seed: int = 42
    ) -> List[GaiaTask]:
        """
        Randomly sample tasks.
        """
        random.seed(seed)

        if n >= len(tasks):
            return tasks

        return random.sample(tasks, n)

    def stratified_sample(
        self,
        tasks: List[GaiaTask],
        easy_n: int,
        medium_n: int,
        hard_n: int
    ) -> List[GaiaTask]:
        """
        Balanced sampling across difficulty levels.
        """

        easy = self.filter_by_difficulty(tasks, "easy")
        medium = self.filter_by_difficulty(tasks, "medium")
        hard = self.filter_by_difficulty(tasks, "hard")

        sampled = []
        sampled += self.sample_tasks(easy, easy_n)
        sampled += self.sample_tasks(medium, medium_n)
        sampled += self.sample_tasks(hard, hard_n)

        random.shuffle(sampled)
        return sampled


def load_default_gaia_dataset(filename: str):
    """
    Convenience loader.
    """
    loader = GaiaLoader()
    return loader.load_json(filename)