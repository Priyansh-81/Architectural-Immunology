from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ============================================================
# GAIA Benchmark Task
# ============================================================

@dataclass
class GaiaTask:
    """
    Represents a single GAIA benchmark task.
    """
    task_id: str
    question: str
    difficulty: str
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Agent Output
# ============================================================

@dataclass
class AgentOutput:
    agent_name: str
    content: str

    confidence: float = 1.0
    tool_calls: List[str] = field(default_factory=list)

    model_used: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================
# Drift Metrics
# ============================================================

@dataclass
class DriftMetrics:
    """
    Stores semantic drift indicators.
    """
    entropy: float = 0.0
    perplexity: float = 0.0

    pid_redundancy: float = 0.0
    pid_uniqueness: float = 0.0

    tool_oscillation_score: float = 0.0
    context_pollution_ratio: float = 0.0

    overall_drift_score: float = 0.0


# ============================================================
# Execution Trace
# ============================================================

@dataclass
class ExecutionTrace:
    """
    Stores full execution trace for one GAIA task.
    Useful for evaluation and debugging.
    """
    task_id: str

    steps: List[str] = field(default_factory=list)
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    drift_history: List[DriftMetrics] = field(default_factory=list)

    final_answer: Optional[str] = None
    success: Optional[bool] = None

    # Evaluation Metrics
    execution_time: float = 0.0
    step_count: int = 0

    # Immune System Metadata
    intervention_triggered: bool = False
    intervention_type: Optional[str] = None


# ============================================================
# Global Agent State
# ============================================================

@dataclass
class AgentState:
    """
    Shared state passed across all agents.
    """
    task: GaiaTask

    current_step: int = 0

    # Shared working memory
    memory: Dict[str, Any] = field(default_factory=dict)

    # Agent outputs
    planner_output: Optional[AgentOutput] = None
    retriever_output: Optional[AgentOutput] = None
    reasoner_output: Optional[AgentOutput] = None
    validator_output: Optional[AgentOutput] = None

    # Drift monitoring
    drift_metrics: Optional[DriftMetrics] = None

    # Immune intervention
    intervention_triggered: bool = False
    intervention_type: Optional[str] = None