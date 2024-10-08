import os
import re
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from .model import get_claude_response
from .utils import VLLM_MODEL, OpenRouter_Model, Agent, strip_reflection
from .config import BASE_MODEL_NAME, BASE_MODEL_URL, FINETUNE_MODEL_NAME, FINETUNE_MODEL_URL


# OpenRouter Model Names
model_names = ["google/gemini-flash-1.5", "openai/gpt-4o", "qwen/qwen-110b-chat", "google/gemini-pro-1.5", "cohere/command-r-plus", "mistralai/mistral-large", "mistralai/mixtral-8x22b-instruct"]


class Detector:
    def __init__(self, initial_query, detection_issues, p1_agent, p2_agent, max_rounds, dir):
        """ 
        Detector Class: 2-player conversation && Issue detector
        - Claude Sonnet adopted for Issue detection with JSON output
        - Mutate on roles are required during the conversation
        """
        self.detection_issues = detection_issues
        self.p1_agent = p1_agent
        self.p2_agent = p2_agent
        self.max_rounds = max_rounds
        self.dir = dir
        self.conversation_history = [{"role": "p2", "content": initial_query}] # Agent begins with specific query
        self.issue_history = []
        self.issues = 0
        
    @classmethod
    def make(cls, 
             use_customer_base: bool,  
             sales_model_name: str, 
             customer_prompt: str,
             sales_prompt: str, 
             initial_query: str,
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
        
        # Load detection issues
        with open("data/detect/issues.json", 'r') as file:
            detection_issues = json.load(file)
        
        # Set up other parameters
        max_rounds = 10
        dir = "data/issues/"
        
        return cls(initial_query, detection_issues, customer_agent, sales_agent, max_rounds, dir)

        
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
    

    def extract_stripped_message(self, last_two_messages):
        """ 
        Extra functional to extract stripped message w/o reflection tag
        - Report issues if reflection tags are not properly closed or opened
        """
        stripped_messages = [strip_reflection(msg["content"]) for msg in last_two_messages]
        for i, (original, stripped) in enumerate(zip(last_two_messages, stripped_messages)):
            if not stripped:
                reflect_issue = {
                    "is_ooc": True,
                    "reflect_issue": True,
                    "issue_detected": "Reflection Tag Issue",
                    "rationale": f"Reflection tags are not properly closed or opened in message {i+1}: {original['content']}"
                }
                self.issues += 1
                self.issue_history.append(reflect_issue)
                self.store_detected_issue(reflect_issue)
                return False # Could not extract conversation properly due to reflection tag issues
        
        # Convert to stripped messages following original format 
        stripped_messages = [
            {"role": msg["role"], "content": stripped}
            for msg, stripped in zip(last_two_messages, stripped_messages)
        ]
        return stripped_messages
    
        
    def detect_issue(self):
        # use sonnet to detect issues
        last_two_messages = self.mapped_conversation[-2:]
        # Strip reflection tag
        stripped_messages = self.extract_stripped_message(last_two_messages)

        if not stripped_messages:
            return False 
        
        claude_prompt = f"""
        Analyze the following conversation for out-of-character behavior or other issues:

        {json.dumps(stripped_messages, indent=2)}
                
        Detect any of the following issues: {', '.join([f"Name: {issue['name']}, Description: {issue['description']}" for issue in self.detection_issues])}
        
        Respond with a JSON object in the following format:
        {{
            "is_ooc": boolean,
            "issue_detected": string (name of the issue detected, or null if no issue),
            "rationale": string (explanation of why the issue was detected)
        }}
        """
        
        claude_response = get_claude_response(claude_prompt)
        try:
            detection_result = json.loads(claude_response)
            if detection_result['is_ooc']:
                self.issues += 1
                self.issue_history.append(detection_result)
                self.store_detected_issue(detection_result)
            return detection_result
        except json.JSONDecodeError:
            print("Error parsing Claude response. Skipping this round.")
            return None
        
    def store_detected_issue(self, detection_result, file_name: str = ""):
        if file_name == "":
            file_name = f"issue_response_{len(self.issue_history)}.json"
        file_path = os.path.join(self.dir, file_name)
        os.makedirs(self.dir, exist_ok=True)
        
        # Map roles for conversation history
        with open(file_path, "w") as f:
            json.dump({
                "conversation": self.mapped_conversation,
                "detection_result": detection_result
            }, f, indent=2)
        
    def run(self):
        # Run detector
        for _ in tqdm(range(self.max_rounds)):
            self.p1_act()
            self.p2_act()
            detection_result = self.detect_issue()
            if detection_result and detection_result['is_ooc']:
                print("### OOC Detected ###")
                print(f"Issue: {detection_result['issue_detected']}")
                print(f"Rationale: {detection_result['rationale']}")
                print(f"Issue Query: {self.mapped_conversation[-2]['content']}")
                print(f"Issue Response: {self.mapped_conversation[-1]['content']}")
                print("####################")
                
        self.store_detected_issue(detection_result)
        
        return self.issue_history, self.mapped_conversation
    
    
    
if __name__ == "__main__":
    
    import argparse
    import random
    import json
    from .prompt import maria_prompt_v018e as maria_prompt
    from .prompt import alex_prompt, alex_incoherent_prompt

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run conversation detection")
    parser.add_argument('-m', type=int, choices=[0, 1], default=0, help='0: use fine-tuned model, 1: use base model')
    parser.add_argument('-v', type=str, default='Jul-10', help='Prompt version')
    parser.add_argument('-o', type=str, default='detected_conversation', help='Output file name prefix')
    args = parser.parse_args()

    # Determine which model to use
    use_base_model = bool(args.m)
    model_type = "base" if use_base_model else "fine-tuned"


    # Loading a variety of queries
    with open("data/detect/queries.json", 'r') as file:
        queries = json.load(file)

    # Loading a variety of agent prompts
    with open("data/detect/prompts.json", 'r') as file:
        prompts = json.load(file)

        
    # Loop through all the conversation phases
    max_rep = 5
    for k in queries:
        for i in range(max_rep):
            # Randomize prompt for sale, as well as initial query from the sale
            sales_prompt = random.choice(prompts.get(k))
            customer_prompt = maria_prompt
            initial_query = random.choice(queries.get(k))
      
            # Diverse model provides diverse chatting experience
            sales_model_name = random.choice(model_names)

            # Create the detector
            detector = Detector.make(
                use_customer_base=use_base_model,
                sales_model_name=sales_model_name,
                customer_prompt=customer_prompt,
                sales_prompt=sales_prompt,
                initial_query=initial_query,
            )

            # Run the detection
            issue_history, conversation_history = detector.run()

            # Construct the output file name
            output_file = f"{args.o}_{model_type}_model_{args.v}_{k}_{sales_model_name.split('/')[-1]}_{initial_query[:20]}.json"
            
            # Store Detection Results
            detector.store_detected_issue({
                "issue_history": issue_history,
                "conversation_history": conversation_history
            }, file_name=output_file)