import json
from openai import OpenAI
import os
import json
# import pandas as pd
import glob
import random
from tqdm import tqdm
from groq import Groq

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],)

def get_oai_response(prompt, system_prompt = "You are a helpful assistant that always closely follows instructions."):
    completion = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
    )
    response = completion.choices[0].message.content
    return response

from openai import OpenAI
from os import getenv

# Convert Conversation History: Agent Side 
def mutate_role_for_agent(c):
    if c["role"] == "Agent":
        c["role"] = "assistant"
    elif c["role"] == "You":
        c["role"] = "user"
    c["text"] = c["text"].strip("\n")
    return c

# Convert Conversation History: Customer Side
def revert_role_for_customer(c):
    if c["role"] == "assistant":
        c["role"] = "Agent"
    elif c["role"] == "user":
        c["role"] = "You"
    c["text"] = c["text"].strip("\n")
    return c    

# gets API Key from environment variable OPENAI_API_KEY
def get_agent_response(utterance, sys_prompt, model="google/gemini-flash-1.5"):
    try:
        client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_API_KEY"),
        )
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,

            },
            {
            "role": "user",
            "content": utterance,
            },
        ],
        )
        return completion.choices[0].message.content
    except:
        response = get_oai_response(prompt = utterance, system_prompt = sys_prompt)
        return response
    
def get_agent_reply(utterance, attack_prompt, conversation_history, model_name):

    [mutate_role_for_agent(c) for c in conversation_history] # In-Place Role Switch
    
    messages = [{"role": "system", "content": attack_prompt}] + [{"role": c["role"], "content": c["text"]} for c in conversation_history]
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_API_KEY"),
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    agent_response = completion.choices[0].message.content
    # Add Agent Response and revert to customer side
    # conversation_history.append({"role": "assistant", "text": agent_response})
    [revert_role_for_customer(c) for c in conversation_history]

    return agent_response, conversation_history

        


# Quality Checker Factory
def quality_checker(realistic: bool, reason: str) -> str:
    f"""
    Use this function to check whether the customer response is realistic. Unrealistic response is super weird and usually repeat sale's response or act like a sale person.
    Note that being rude is completely fine, customer tends to have a variety of personalities. 
    Args: 
        realistic (bool): Whether the response is realistic or not.
        reason (str): The reason for the response being realistic or not.

    Returns:
        dict: A dictionary containing the quality of the response and the reason for the quality.
    """
    try:
        return json.dumps({"realistic": realistic, "reason": reason})
    except:
        return json.dumps({"realistic": False, "reason": "No quality checker found"})
    

def get_response(assistant):
    content = assistant.memory.chat_history[-1].content
    response = []
    lines = content.split("\n")
    patterns = ["1.", "2.", "3.", "4.", "5."]
    for line in lines:
        for pattern in patterns:
            if pattern in line:
                r = line.split(pattern)[1].strip()
                response.append(r)
    return response


def parse_messages(text):
    lines = text.split("\n")
    # Few pattern templates for parsing the generated responses
    pattern_templates = ["{i}. **[RESPONSE{i}]**", "{i}. [RESPONSE{i}]", "{i}. [RESPONSE]", "{i}."]
    messages = []
    for l in lines:
        for i in range(1,6):
            for pattern_template in pattern_templates:
                pattern = pattern_template.format(i=i)
                if pattern in l:
                    messages.append(l.split(pattern)[-1].strip())
                    break
    return messages


# Comment: This template is super weird -- why do we put example input & output as if they are part of the conversation (?)
LlAMA3_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example_output}<|eot_id|><|start_header_id|>user<|end_header_id|>"""



from openai import OpenAI
import json

def get_instruct_conversation_continue(utter_dict, prompt, experiment=False):
    # dictionary input for conversation parsing
    if experiment:
        client = OpenAI(api_key="EMPTY", base_url="http://43.218.240.61:8000/v1")
    else:    
        client = OpenAI(api_key="EMPTY", base_url="http://ec2-13-229-24-160.ap-southeast-1.compute.amazonaws.com:8000/v1")
    conversation_history = []
    for key, item in utter_dict.items():
        if key == "Customer":
            conversation_history.append({"role": "You", "text": item})
        else:
            conversation_history.append({"role": "Agent", "text": item})
    conversation = "\n".join([f"{x['role']}: {x['text'].strip()}" for x in conversation_history])
    conversation += "\nYou:"
    input_prompt = prompt + f"{conversation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    agent_response = []
    stream = client.completions.create(
        model="MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ",
        prompt=input_prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"],
        stream=True,
        extra_body={
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "min_tokens": 0,
        },
    )
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        agent_response.append(txt)

    agent_response_text = "".join(agent_response)
    return agent_response_text


def get_finetune_response(utterance, prompt):
    client = OpenAI(api_key="EMPTY", base_url="http://43.218.240.61:8000/v1")  # Jarkata Server Fine-tuned llama3 + Lora adapted model response

    conversation_history = []
    user_input = utterance
    conversation_history.append({"role": "Agent", "text": user_input})
    conversation = "\n".join([f"{x['role']}: {x['text'].strip()}" for x in conversation_history])
    conversation += "\nYou:"
    input_prompt = prompt + f"{conversation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    agent_response = []
    stream = client.completions.create(
        model = "Ksgk-fy/ecoach_philippine_v2_merge",
        prompt=input_prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"],
        stream=True,
        extra_body={
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "min_tokens": 0,
        },
    )
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        agent_response.append(txt)

    agent_response_text = "".join(agent_response)
    conversation_history.append({"role": "You", "text": agent_response_text})
    return agent_response_text


def get_instruct_response(utterance, prompt, experiment = False):
    if experiment:
        client = OpenAI(api_key="EMPTY", base_url="http://43.218.240.61:8000/v1")
    else:
        client = OpenAI(api_key="EMPTY", base_url="http://ec2-13-229-24-160.ap-southeast-1.compute.amazonaws.com:8000/v1")

    conversation_history = []
    user_input = utterance
    conversation_history.append({"role": "Agent", "text": user_input})
    conversation = "\n".join([f"{x['role']}: {x['text'].strip()}" for x in conversation_history])
    conversation += "\nYou:"
    input_prompt = prompt + f"{conversation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    agent_response = []
    stream = client.completions.create(
        model="MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ",
        prompt=input_prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"],
        stream=True,
        extra_body={
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "min_tokens": 0,
        },
    )
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        agent_response.append(txt)

    agent_response_text = "".join(agent_response)
    conversation_history.append({"role": "You", "text": agent_response_text})
    return agent_response_text


def get_instruct_response_history(utterance, prompt, conversation_history: list = []):
    client = OpenAI(api_key="EMPTY", base_url="http://ec2-13-229-24-160.ap-southeast-1.compute.amazonaws.com:8000/v1")

    user_input = utterance
    conversation_history.append({"role": "Agent", "text": user_input})
    conversation = "\n".join([f"{x['role']}: {x['text'].strip()}" for x in conversation_history])
    conversation += "\nYou:"
    input_prompt = prompt + f"{conversation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    agent_response = []
    stream = client.completions.create(
        model="MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ",
        prompt=input_prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"],
        stream=True,
    )
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        agent_response.append(txt)

    agent_response_text = "".join(agent_response)
    conversation_history.append({"role": "You", "text": agent_response_text})
    return agent_response_text, conversation_history


def prepare_prompt_llama3(instructions):

    example_input = """So FWD Set For Life is a life insurance policy that helps you earn some returns by investing your premiums"""
    
    example_output = """Like, how does the investment part work? What kind of returns should I be looking at, pa-reh?"""

    return LlAMA3_PROMPT_TEMPLATE.format(system_prompt=instructions, example_input=example_input, example_output=example_output)


def get_llama_experiment_response(utterance, sys_prompt):
    prompt = prepare_prompt_llama3(sys_prompt)
    return get_finetune_response(utterance, prompt)

def get_llama_prod_response(utterance, sys_prompt, experiment=False):
    prompt = prepare_prompt_llama3(sys_prompt)
    if isinstance(utterance, str):
        return get_instruct_response(utterance, prompt, experiment=experiment)
    else:
        return get_instruct_conversation_continue(utterance, prompt, experiment=experiment)
    
def get_llama_chat(utterance, sys_prompt, conversation_history: list = []):
    # prompt = prepare_prompt_llama3(sys_prompt) # V5 Prompt is already folded up 
    prompt = sys_prompt
    return get_instruct_response_history(utterance, prompt, conversation_history)
    

#####################
# Scenario Specific #
#####################

# (IV) objection: customer begin | agent reply | customer push back
# Out Of Character Checks (OOCC)



def parse_conversation(conv_dict):
    conversation_str = ""
    for entry in conv_dict:
        if entry['role'] == 'Agent':
            conversation_str += "Stan: " + entry['text'] + "\n"
        elif entry['role'] == 'You':
            conversation_str += "Maria: " + entry['text'] + "\n"
    return conversation_str

def groq_OOC_check(conv_dict):
    """
    Use Groq Agent to check if the response from Maria is out of character from the provided conversation, we need Maria to be a customer
    """
    
    parsed_conversation = parse_conversation(conv_dict)

    client = Groq(api_key = os.getenv('GROQ_API_KEY'))
    MODEL = 'llama3-70b-8192'

    conv_history = """Here are the conversation between Maria, the customer and Stan, the insurance Agent: {conversation_history}"""
    conv_history = conv_history.format(conversation_history=parsed_conversation)

    messages=[
        {
            "role": "system",
            "content": "You are a function calling LLM which checks if the response from Maria is out of character from the provided conversation, we need Maria to be a customer. Provide your rationale for making the out-of-character judgement.",
        },
        {
            "role": "user",
            "content": conv_history,
        }
    ]
    tools = [
            {
                "type": "function",
                "function": {
                    "name": "is_out_of_character",
                    "description": "Check if the response is out of character for the given agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "out_of_character": {
                                "type": "boolean",
                                "description": "Whether the response is out of character",
                            },
                            "out_of_character_rationale": {
                                "type": "string",
                                "description": "The reason the response is out of character",
                            }
                        },
                        "required": ["out_of_character", "out_of_character_rationale"],
                    },
                },
            }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    available_functions = {
        "is_out_of_character": True,
    }  # only one function in this example, but you can have multiple
    out_of_character = False
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            out_of_character = function_args['out_of_character']
            rationale = function_args["out_of_character_rationale"]

    return out_of_character, rationale



def groq_behavior_check(conv_dict):
    """
    Use Groq Agent to check if the response from Maria is behaving in the wrong way
    -- Check whether response from Maria is asking back questions about insurance
    """
    
    parsed_conversation = parse_conversation(conv_dict)

    client = Groq(api_key = os.getenv('GROQ_API_KEY'))
    MODEL = 'llama3-70b-8192'

    conv_history = """Here are the conversation between Maria, the customer and Stan, the insurance Agent: {conversation_history}"""
    conv_history = conv_history.format(conversation_history=parsed_conversation)

    messages=[
        {
            "role": "system",
            "content": "You are a function calling LLM which checks if the response from Maria is asking back questions about insurance.",
        },
        {
            "role": "user",
            "content": conv_history,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_back_insurance",
                "description": "Check if the response is asking back questions about insurance product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ask_back_insurance": {
                            "type": "boolean",
                            "description": "Whether the response is asking back questions about insurance product"
                        },
                        "ask_back_insurance_rationale": {
                            "type": "string",
                            "description": "The reason the response is asking back questions about insurance product"
                        }
                    },
                    "required": ["ask_back_insurance", "ask_back_insurance_rationale"]
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    available_functions = {
        "ask_back_insurance": True,
    }  # only one function in this example, but you can have multiple
    out_of_character = False
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            ask_back_insurance = function_args['ask_back_insurance']
            rationale = function_args["ask_back_insurance_rationale"]

    # return out_of_character, rationale
    try:
        return ask_back_insurance, rationale
    except:
        return False, ""

# Adversial Attack


def adversial_attack(eCoach_prompt, dir, attack_prompt, model_name, id=1, agent_response = "Hello Maria", max_rounds: int = 100):
    issues = 0
    issue_history = []
    conversation_history = []
    for i in tqdm.tqdm(range(max_rounds)):
        try:
            response, conversation_history = get_llama_chat(utterance = agent_response, sys_prompt = eCoach_prompt)
        except Exception as e:
            print("Rounds of conversation accumulated so far: ", len(conversation_history)//2)
            print("Error in getting Ecustomer response : ", e)
            # Slice out only last 10 rounds of conversations
            conversation_history = conversation_history[-10:]
            response, conversation_history = get_llama_chat(utterance = agent_response, sys_prompt = eCoach_prompt)

        model_name = model_name.replace("/", "-")
        file_name = f"llama_v6_response_{model_name}_{id}_{i}.json"
        file_path = os.path.join(dir, file_name)
        os.makedirs(dir, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(conversation_history, f)

        try:
            agent_response = get_agent_response(utterance=response, sys_prompt=attack_prompt, model=model_name)
        except:
            import time
            time.sleep(5)
            agent_response = get_agent_response(utterance=response, sys_prompt=attack_prompt, model=model_name)
    
    if issues == 0:
        out_str = f"\nRun id {id} | Model: {model_name} | No Issues Detected"
        print(out_str)
    else:
        out_str = f"\nRun id {id} | Model: {model_name} | Issues Detected: {issues}"
        print(out_str)

    return issue_history, conversation_history


def adversial_attack_ooc(eCoach_prompt, attack_prompt, agent_response, model_names, id, max_rounds, dir):
    """ 
    V2 Refinement of the Adversial Attack Functional 
    - Focus on OutOfCharacter Issues (Which we could also potentially fix with fine-tuning)
    - FWD Production GPTQ llama3 adopted for customer roleplay
    - OpenRouter with a hoard of LLMs to roleplay as Insurance agent 
    - Groq 70B llama3 for OOC detection (Function calling agent)
    """

    issues = 0
    issue_history = []
    conversation_history = []
    model_name = random.choice(model_names)
    for i in tqdm(range(max_rounds)):
        try:
            # How come the previous conversation history is never included into the input ?
            response, conversation_history = get_llama_chat(utterance = agent_response, sys_prompt = eCoach_prompt, conversation_history=conversation_history)
        except Exception as e:
            print("Rounds of conversation accumulated so far: ", len(conversation_history)//2)
            print("Error in getting Ecustomer response : ", e)
            # Slice out only last 10 rounds of conversations
            conversation_history = conversation_history[-10:]
            response, conversation_history = get_llama_chat(utterance = agent_response, sys_prompt = eCoach_prompt, conversation_history=conversation_history)

        # Issue Detection for the Current Round
        # is_ooc, rationale = groq_OOC_check(conversation_history[-2:])
        is_ooc, rationale = groq_behavior_check(conversation_history[-2:])
        if is_ooc:
            print("### OOC Detected ###")
            print("Rationale: ", rationale)
            issue_query = conversation_history[-2]["text"]
            issue_response_maria = conversation_history[-1]["text"]
            print("Issue Query: ", issue_query)
            print("Issue Response from Maria: ", issue_response_maria)
            print("####################")
            issues += 1
            issue_history.append(rationale)

            # Store Detected Issue Script ||  Last Utterance Has Issue
            model_name = model_name.replace("/", "-")
            file_name = f"llama_v6_issue_response_{model_name}_{id}_{i}.json"
            file_path = os.path.join(dir, file_name)
            os.makedirs(dir, exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(conversation_history, f)

        # Agent respond without knowing any of the conversation history, which is also very wrong ..
        agent_response_success = False
        while not agent_response_success:
            try:
                agent_response, conversation_history = get_agent_reply(utterance=response, attack_prompt=attack_prompt, conversation_history=conversation_history, model_name=model_name)
                agent_response_success = True
            except Exception as e:
                # print("Error in getting agent response : ", e)
                model_name = random.choice(model_names)

    return issue_history, conversation_history