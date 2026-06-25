from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# Model Configuration
# ============================================================

@dataclass
class ModelConfig:
    """
    Two-model architecture.

    Small model:
        - Planner
        - Retriever
        - Validator

    Reasoning model:
        - Reasoner
    """

    # Small shared model
    small_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Heavy reasoning model
    reasoning_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Runtime settings
    quantized: bool = True
    use_flash_attention: bool = False
    device: str = "auto"

    # Generation defaults
    max_new_tokens: int = 256
    temperature: float = 0.7


# ============================================================
# Drift Configuration
# ============================================================

@dataclass
class DriftConfig:
    """
    Thresholds used by semantic drift monitor.
    """

    entropy_threshold: float = 3.0
    perplexity_threshold: float = 50.0
    tool_oscillation_threshold: float = 0.4
    context_pollution_threshold: float = 0.35
    intervention_threshold: float = 0.7


# ============================================================
# Evaluation Configuration
# ============================================================

@dataclass
class EvaluationConfig:
    """
    Benchmark / GAIA evaluation configuration.
    """

    max_steps_per_task: int = 25
    rolling_window: int = 10
    save_logs: bool = True


# ============================================================
# Paths
# ============================================================

@dataclass
class Paths:
    """
    Project filesystem paths.
    """

    root: Path = Path(".")
    data_dir: Path = Path("./data")
    logs_dir: Path = Path("./outputs/logs")
    plots_dir: Path = Path("./outputs/plots")


# ============================================================
# App Configuration
# ============================================================

@dataclass
class AppConfig:
    """
    Global application configuration.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: Paths = field(default_factory=Paths)


# Global config object
CONFIG = AppConfig()