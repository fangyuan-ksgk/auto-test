# Evaluator Model 
import anthropic
from anthropic import Anthropic 
from os import getenv 
import torch
import transformers
from huggingface_hub import login
from openai import OpenAI
import os

HF_TOKEN = getenv("HF_TOKEN")
ANTHROPIC_API_KEY = "YOUR_API_KEY"
OPENAI_API_KEY = "YOUR_API_KEY"
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"

client = Anthropic(api_key=getenv("ANTHROPIC_API_KEY"))
oai_client = OpenAI(
    api_key=getenv('OPENAI_API_KEY'))

def get_claude_response(prompt):
    """ 
    Anthropic Claude Sonnet-3.5 Model
    - Powerful, Slow, Costly
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.content[0].text

from anthropic import AsyncAnthropic
async_client = AsyncAnthropic(api_key=getenv("ANTHROPIC_API_KEY"))

async def get_claude_response_async(prompt):
    """ 
    Anthropic Claude Sonnet-3.5 Model
    - Powerful, Slow, Costly
    """
    response = await async_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.content[0].text


def get_oai_response(prompt, system_prompt = "You are a helpful assistant that always closely follows instructions."):
    """ 
    Get response from OAI Model
    """
    completion = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
    )
    response = completion.choices[0].message.content
    return response


# Load HF-FineTuned CheckPoint model and serve with vLLM 
try:
    from vllm import SamplingParams, LLM
    login(token=HF_TOKEN)

    class HFModel:
        def __init__(self, model_name="Ksgk-fy/ecoach_philippine_v2_merge"):
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=256,
                stop=["<|eot_id|>"]
            )
            self.llm = LLM(model_name, params=self.sampling_params)

        def generate_responses(self, prompts):
            outputs = self.llm.generate(
                prompts,
                self.sampling_params,
            )
            return outputs
except ImportError:
    print("vllm package not found. HFModel will not be available.")
    HFModel = None


# Load REFT checkpoint and run inference 
try:
    import pyreft
    from pyreft import ReftModel
    login(token=HF_TOKEN)

    class ReftInferenceModel:
            
        def __init__(self, model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_max_length = 2048
            self.model_name_or_path = model_name_or_path
            
            self._load_model()
            self._load_tokenizer()
            self._load_reft_adaptor()
            
        def _load_model(self):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, torch_dtype=torch.bfloat16, device_map=self.device)
            
        def _load_tokenizer(self):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name_or_path, model_max_length=self.model_max_length, 
                padding_side="right", use_fast=False)
            if "Meta-Llama-3-" in self.model_name_or_path:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
        def _load_reft_adaptor(self):
            self.reft_model = ReftModel.load("Ksgk-fy/Zalinger02_reft_llama3", self.model, from_huggingface_hub=True)
            self.reft_model.set_device("cuda")
            
        def generate_response(self, prompt, system_prompt="Follow the instruction closely and provide your answer."):
            # tokenize and prepare the input
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], 
                tokenize=False)
            tokenized_prompt = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            # get reft model configuration
            reft_config = pyreft.ReftConfig(representations=[{
                "layer": l, "component": "block_output",
                "low_rank_dimension": 2,
                "intervention": pyreft.LoreftIntervention(embed_dim=self.model.config.hidden_size,
                low_rank_dimension=2)} for l in [8, 16, 24]])
            share_weights = True
            positions = "f1+l1"
            first_n, last_n = pyreft.parse_positions(positions)

            unit_locations = torch.IntTensor([pyreft.get_intervention_locations(
                last_position=tokenized_prompt["input_ids"].shape[-1], 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="last",
                num_interventions=len(reft_config.representations),
                share_weights=share_weights
            )]).permute(1, 0, 2).tolist()

            _, reft_response = self.reft_model.generate(
                tokenized_prompt, unit_locations={"sources->base": (None, unit_locations)},
                intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
                eos_token_id=self.terminators, early_stopping=True
            )
            return self.tokenizer.decode(reft_response[0])
except ImportError:
    print("pyreft package not found. ReftInferenceModel will not be available.")