# from synwrite.prompt_fwd import attack_prompt, customerPromptLlama3_instruct_v6
from .utils import adversial_attack_ooc, collect_ooc_response
from .utils import maria_prompt, alex_prompt
import json 
import random

# Customer & Agent Prompt
eCoach_prompt = maria_prompt
eAgent_prompt = alex_prompt

# Model names
model_names = ["google/gemini-flash-1.5", "openai/gpt-4o", "qwen/qwen-110b-chat", "google/gemini-pro-1.5", "cohere/command-r-plus", "mistralai/mistral-large", "mistralai/mixtral-8x22b-instruct"]


def run_detection(queries, detection_issues):
    idx = random.randint(0, len(queries)-1)
    agent_response = queries[idx]
    dir = "data/prompt-benchmark/adversial/"
    max_rounds = 10

    for agent_response in queries:
        issue_history, conversation_history = adversial_attack_ooc(eCoach_prompt, eAgent_prompt, detection_issues, agent_response, model_names, idx, max_rounds, dir)
    
    return issue_history, conversation_history

if __name__ == "__main__":
    
    issue_path = "data/detect/issues.json"
    with open(issue_path, 'r') as file:
        detection_issues = json.load(file)
        
    query_path = "data/detect/queries.json"
    with open(query_path, 'r') as file:
        queries = json.load(file)["queries"]

    run_detection(queries, detection_issues)
    collect_ooc_response()