from abc import ABC, abstractmethod
from typing import Optional
import re

from src.schemas import AgentState, AgentOutput
from src.memory import MemoryManager
from src.models import BaseLLM


# ============================================================
# Base Agent
# ============================================================

class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    def __init__(
        self,
        name: str,
        model: BaseLLM,
        memory: MemoryManager
    ):
        self.name = name
        self.model = model
        self.memory = memory

    @abstractmethod
    def run(self, state: AgentState) -> AgentOutput:
        pass

    def _store_output(self, output: AgentOutput):
        self.memory.add_to_short_term(
            content=output.content,
            source=self.name,
            importance=output.confidence
        )

        self.memory.add_to_episodic(
            content=output.content,
            source=self.name,
            importance=output.confidence
        )


# ============================================================
# Planner Agent
# ============================================================

class PlannerAgent(BaseAgent):
    """
    Break task into executable steps.
    """

    def run(self, state: AgentState) -> AgentOutput:
        prompt = f"""
You are a planning agent.

Task:
{state.task.question}

Break this task into step-by-step executable subtasks.
Return concise numbered steps.
"""

        response = self.model.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.3
        )

        output = AgentOutput(
            agent_name=self.name,
            content=response,
            confidence=0.9
        )

        self._store_output(output)
        return output


# ============================================================
# Retriever Agent
# ============================================================

class RetrieverAgent(BaseAgent):
    """
    Retrieve relevant information.
    (Mock retrieval for now)
    """

    def run(self, state: AgentState) -> AgentOutput:
        plan = ""
        if state.planner_output:
            plan = state.planner_output.content

        context = self.memory.context_window()

        prompt = f"""
You are a retrieval agent.

Task:
{state.task.question}

Plan:
{plan}

Context:
{context}

Extract useful factual information needed to solve the task.
"""

        response = self.model.generate(
            prompt,
            max_new_tokens=250,
            temperature=0.2
        )

        output = AgentOutput(
            agent_name=self.name,
            content=response,
            confidence=0.85,
            tool_calls=["retrieval"]
        )

        self._store_output(output)
        return output


# ============================================================
# Reasoner Agent
# ============================================================

class ReasonerAgent(BaseAgent):
    """
    Performs reasoning over retrieved information.
    """

    def run(self, state: AgentState) -> AgentOutput:
        retrieval = ""

        if state.retriever_output:
            retrieval = state.retriever_output.content

        prompt = f"""
You are a reasoning agent.

Task:
{state.task.question}

Retrieved Information:
{retrieval}

Solve the task carefully.
Show reasoning and final answer.
"""

        response = self.model.generate(
            prompt,
            max_new_tokens=300,
            temperature=0.5
        )

        output = AgentOutput(
            agent_name=self.name,
            content=response,
            confidence=0.88
        )

        self._store_output(output)
        return output


# ============================================================
# Validator Agent
# ============================================================

class ValidatorAgent(BaseAgent):
    """
    Checks final answer quality.
    """

    def extract_final_answer(self, text: str) -> str:
        match = re.search(
            r"(final answer\s*:?\s*)(.*)",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(2).strip()

        return text.strip()

    def run(self, state: AgentState) -> AgentOutput:
        reasoning = ""

        if state.reasoner_output:
            reasoning = state.reasoner_output.content

        prompt = f"""
You are a validator.

Task:
{state.task.question}

Candidate Answer:
{reasoning}

Check if the answer is logical, consistent,
and properly formatted.
"""

        response = self.model.generate(
            prompt,
            max_new_tokens=150,
            temperature=0.2
        )

        final_answer = self.extract_final_answer(reasoning)

        output = AgentOutput(
            agent_name=self.name,
            content=response + f"\nFINAL_ANSWER: {final_answer}",
            confidence=0.92
        )

        self._store_output(output)
        return output