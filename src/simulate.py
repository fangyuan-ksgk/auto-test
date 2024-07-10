import json 
import os
import random
from .utils import VLLM_MODEL, OpenRouter_Model, Agent
from .config import BASE_MODEL_NAME, BASE_MODEL_URL, FINETUNE_MODEL_NAME, FINETUNE_MODEL_URL
from transformers import AutoTokenizer
from .model import get_claude_response
from tqdm import tqdm as tqdm 
import time
from typing import Optional

# OpenRouter Model Names
model_names = ["google/gemini-flash-1.5", "openai/gpt-4o", "qwen/qwen-110b-chat", "google/gemini-pro-1.5", "cohere/command-r-plus", "mistralai/mistral-large", "mistralai/mixtral-8x22b-instruct"]

# Simulate Conversation
class Simulator:
    
    def __init__(self, initial_query, p1_agent, p2_agent, max_rounds, output_dir):
        """ 
        Simulator: 2-player conversation
        - Automate testing on served models, with different prompts
        """
        self.p1_agent = p1_agent
        self.p2_agent = p2_agent
        self.max_rounds = max_rounds
        self.conversation_history = [{"role": "p2", "content": initial_query}] # Agent begins with specific query
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    @classmethod
    def make(cls, 
             use_customer_base: bool,  
             sales_model_name: str, 
             customer_prompt: str,
             sales_prompt: str, 
             initial_query: str,
             max_round: Optional[int] = 10,
             tokenizer_name: str= "meta-llama/Meta-Llama-3-8B-Instruct"):
        if use_customer_base:
            customer_model = VLLM_MODEL(BASE_MODEL_NAME, BASE_MODEL_URL)
        else:
            customer_model = VLLM_MODEL(FINETUNE_MODEL_NAME, FINETUNE_MODEL_URL)
     
        # Initialize Sales Agent
        sales_model = OpenRouter_Model(sales_model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or "meta-llama/Meta-Llama-3-8B-Instruct")
        sales_agent = Agent(sales_model, tokenizer, sales_prompt)
        
        # Initialize Customer Agent
        customer_agent = Agent(customer_model, tokenizer, customer_prompt)
        
        # Set up other parameters
        max_rounds = max_round if max_round is not None else 10
        output_dir = "data/simulate/"
        
        return cls(initial_query, customer_agent, sales_agent, max_rounds, output_dir)

    def p1_act(self):
        # Get p1 response
        p1_response = self.p1_agent.get_response(self.conversation_history[-1]["content"] if self.conversation_history else "")
        self.conversation_history.append({"role": "p1", "content": p1_response})
        return p1_response
        
    def p2_act(self):
        # Get p2 response
        p2_response = self.p2_agent.get_response(self.conversation_history[-1]["content"])
        self.conversation_history.append({"role": "p2", "content": p2_response})
        return p2_response
    
    @property
    def mapped_conversation(self):
        mapped_conversation = [
            {"role": "Maria" if msg["role"] == "p1" else "Alex", "content": msg["content"]} # Temporary Hack
            for msg in self.conversation_history
        ]
        return mapped_conversation
        
    def run(self):
        # Run simulation
        for _ in tqdm(range(self.max_rounds)):
            self.p1_act()
            self.p2_act()
        
        # Store the simulated conversation
        timestamp = int(time.time())
        filename = f"simulated_conversation_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.mapped_conversation, f, indent=2)
        
        print(f"Simulated conversation stored in: {filepath}")
        
        return self.mapped_conversation
    
    def save_conversation(self, output_file):
        """Save the conversation to a JSON file."""
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(self.mapped_conversation, f, indent=2)
        print(f"Simulated conversation stored in: {output_path}")
    
    
if __name__ == "__main__":
    
    import argparse
    import random
    from .prompt import maria_prompt, alex_prompt

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run conversation simulation")
    parser.add_argument('-m', type=int, choices=[0, 1], default=0, help='0: use fine-tuned model, 1: use base model')
    parser.add_argument('-v', type=str, default='Jul-10', help='Prompt version')
    parser.add_argument('-o', type=str, default='simulated_conversation', help='Output file name prefix')
    args = parser.parse_args()

    # Determine which model to use
    use_base_model = bool(args.m)
    model_type = "base" if use_base_model else "fine-tuned"

    # Set up prompts
    customer_prompt = maria_prompt
    sales_prompt = alex_prompt

    # Loading a variety of queries
    with open("data/detect/queries.json", 'r') as file:
        queries = json.load(file)["queries"]    
    
    for initial_query in queries:
        # Diverse model provides diverse chatting experience
        sales_model_name = random.choice(model_names)

        # Create the simulator
        simulator = Simulator.make(
            use_customer_base=use_base_model,
            sales_model_name=sales_model_name,
            customer_prompt=customer_prompt,
            sales_prompt=sales_prompt,
            initial_query=initial_query,
        )

        # Run the simulation
        conversation = simulator.run()

        # Construct the output file name
        output_file = f"{args.o}_{model_type}_model_{args.v}_{sales_model_name.split('/')[-1]}_{initial_query[:20]}.json"
        
        # Store Conversation
        simulator.save_conversation(output_file)