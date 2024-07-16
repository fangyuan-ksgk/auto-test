# Configuration for Interacting with Base & Finetuned llama3 model

# Some of the served model with OpenAI compatible structure
BASE_MODEL_URL = "http://ec2-13-229-24-160.ap-southeast-1.compute.amazonaws.com:8000/v1"
BASE_MODEL_NAME = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ"
FINETUNE_MODEL_URL = "http://43.218.77.178:8000/v1"
FINETUNE_MODEL_NAME = "Ksgk-fy/maria_v113_gptq_prompt" # v0.1.3 quantised model

# Some API Keys used to simulate & evaluate above served model
ANTHROPIC_API_KEY = "YOUR_API_KEY"
OPENAI_API_KEY = "YOUR_API_KEY"
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
HF_TOKEN = "YOUR_TOKEN_HERE"