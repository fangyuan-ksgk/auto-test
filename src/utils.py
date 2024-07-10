import json
from openai import OpenAI
import os
import json
import glob
import random
from tqdm import tqdm
from .model import get_oai_response
from os import getenv 
from typing import Optional, Union


def load_requirements(directory="data/attribute"):
    attribute_files = glob.glob(f"{directory}/*.json")
    loaded_requirements = []

    for file_path in attribute_files:
        with open(file_path, 'r') as file:
            attribute_data = json.load(file)
            loaded_requirements.append({
                "name": attribute_data["name"],
                "desc": attribute_data["desc"]
            })

    return loaded_requirements


def load_conversations(folder_dir="data/conversation"):
    """ 
    Load conversation under certain directory
    """
    conversation_files = glob.glob(f"{folder_dir}/*.json")
    for file_path in conversation_files:
        with open(file_path, 'r') as file:
            conversation_data = json.load(file)
            conversations = conversation_data["conversations"]
    return conversations


# Comment: This template is super weird -- why do we put example input & output as if they are part of the conversation (?)
LlAMA3_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example_output}<|eot_id|><|start_header_id|>user<|end_header_id|>"""


class VLLM_MODEL: 
    """
    vLLM served model class
    """
    def __init__(self, model_name, base_url):
        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)

    def get_completion(self, prompt, max_tokens=512, temperature=0.0, stop="<|eot_id|>"):
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=False,
            extra_body={
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "min_tokens": 0,
            },
        )
        return response.choices[0].text.strip()

    def get_streaming_completion(self, prompt, max_tokens=512, temperature=0.0, stop="<|eot_id|>"):
        stream = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
            extra_body={
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "min_tokens": 0,
            },
        )
        return stream
    
    
class OpenRouter_Model:
    """
    OpenRouter served model class
    """
    def __init__(self, model_name="google/gemini-flash-1.5"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv("OPENROUTER_API_KEY"),
        )

    def get_completion(self, prompt, max_tokens=512, temperature=0.0, stop=["<|eot_id|>"]):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return response.choices[0].message.content
        except:
            return get_oai_response(prompt=prompt)

    def get_streaming_completion(self, prompt, max_tokens=512, temperature=0.0, stop=["<|eot_id|>"]):
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
            )
            return stream
        except:
            return get_oai_response(prompt=prompt, stream=True)
    

class Agent:
    """ 
    vLLM served Agent LLM object
    """
    def __init__(self, 
                 model: Optional[Union[OpenRouter_Model, VLLM_MODEL]],
                 tokenizer,
                 system_prompt):
        
        self.model = model 
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.reset_conversation()
        
    @property 
    def query_prompt(self):
        """
        Format the prompt for the LLM using the conversation history and system prompt
        """
        completion = "####Dummy-Answer"
        messages = []
        messages.extend(self.conversation_history)
        messages.append({"role": "assistant", "content": completion})
        format_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        query_prompt = format_prompt.split(completion)[0]
        return query_prompt 
    
    def get_response(self, user_input):
        """
        Get a response from the LLM based on the user input and system prompt
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.model.get_completion(self.query_prompt)
        self.conversation_history.append({"role": "assistant", "content": response})
        # Truncate conversation history
        self.conversation_history = self.conversation_history[-8:]
        return response
    
    def reset_conversation(self):
        """
        Reset the conversation history
        """
        self.conversation_history = [{"role":"system", "content":self.system_prompt}]