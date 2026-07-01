    from abc import ABC, abstractmethod
    from typing import Dict, Optional, Tuple

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None

    from src.config import CONFIG


    # ============================================================
    # Quantization Helper
    # ============================================================

    def build_quant_config():
        """
        Build 4-bit quantization config if supported.
        Returns None on unsupported systems.
        """

        if not CONFIG.model.quantized:
            return None

        if BitsAndBytesConfig is None:
            return None

        if not torch.cuda.is_available():
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )


    # ============================================================
    # Base Interface
    # ============================================================

    class BaseLLM(ABC):
        """
        Abstract interface for all LLM backends.
        """

        @abstractmethod
        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.7
        ) -> str:
            pass


    # ============================================================
    # HuggingFace Backend
    # ============================================================

    class HuggingFaceLLM(BaseLLM):

        def __init__(
            self,
            model_name: str,
            device: Optional[str] = None
        ):
            self.model_name = model_name

            # Device detection
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    device = "mps"
                else:
                    device = "cpu"

            self.device = device

            print(f"[INFO] Loading model: {model_name}")
            print(f"[INFO] Using device: {device}")

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = (
                    self.tokenizer.eos_token
                )

            # Quantization config
            quant_config = build_quant_config()

            model_kwargs = {
                "trust_remote_code": True
            }

            if quant_config is not None:
                print("[INFO] Using 4-bit quantization")

                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"

            else:
                dtype = (
                    torch.float16
                    if self.device in ["cuda", "mps"]
                    else torch.float32
                )

                model_kwargs["dtype"] = dtype

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

            # Manual move only if NOT quantized
            if (
                quant_config is None
                and self.device != "cpu"
            ):
                self.model = self.model.to(self.device)

            self.model.eval()

        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.7
        ) -> str:

            print(f"[{self.model_name}] generating...")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            # Move inputs safely
            if self.device != "cpu":
                if hasattr(self.model, "device"):
                    target_device = self.model.device
                else:
                    target_device = self.device

                inputs = {
                    key: value.to(target_device)
                    for key, value in inputs.items()
                }

            with torch.no_grad():
                outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
                )

            decoded = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].strip()

            return decoded


    # ============================================================
    # Model Registry
    # ============================================================

    class ModelRegistry:
        """
        Ensures models are loaded only once.
        """

        def __init__(self):
            self.loaded_models: Dict[str, BaseLLM] = {}

        def get_model(
            self,
            model_name: str
        ) -> BaseLLM:

            if model_name not in self.loaded_models:
                model = HuggingFaceLLM(model_name)
                self.loaded_models[model_name] = model

            return self.loaded_models[model_name]


    registry = ModelRegistry()


    # ============================================================
    # Two-Model Loader
    # ============================================================

    def load_models() -> Tuple[BaseLLM, BaseLLM]:
        """
        Returns:
            small_model
            reasoning_model
        """

        small_model = registry.get_model(
            CONFIG.model.small_model
        )

        reasoning_model = registry.get_model(
            CONFIG.model.reasoning_model
        )

        return small_model, reasoning_model