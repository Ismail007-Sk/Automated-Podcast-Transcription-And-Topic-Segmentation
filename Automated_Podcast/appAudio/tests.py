from django.test import TestCase

# Create your tests here.



from llama_cpp import Llama
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

# llm = Llama(
#     model_path=model_path,
#     n_gpu_layers=8,
#     n_ctx=2048
# )

llm = Llama(
    model_path=model_path,
    n_gpu_layers=8,
    verbose=True
)





