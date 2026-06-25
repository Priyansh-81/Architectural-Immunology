class MockLLM:
    def generate(
        self,
        prompt: str,
        max_new_tokens=256,
        temperature=0.7
    ):
        prompt = prompt.lower()

        if "planning agent" in prompt:
            return """1. Identify relevant data
2. Solve problem
3. Return final answer"""

        elif "retrieval agent" in prompt:
            return "Retrieved information: 2 + 2 equals 4"

        elif "reasoning agent" in prompt:
            return "Reasoning complete. Final Answer: 4"

        elif "validator" in prompt:
            return "Answer is logically correct."

        return "Mock response"


def load_models():
    small_model = MockLLM()
    reasoning_model = MockLLM()

    return small_model, reasoning_model