from src.models import HuggingFaceLLM

model = HuggingFaceLLM(
    "sshleifer/tiny-gpt2"
)

output = model.generate(
    "Hello, my name is",
    max_new_tokens=20
)

print(output)