import dataclasses
from typing import List, Dict, Optional
import os
import json

@dataclasses.dataclass
class Bucket:
    """ 
    Bucket is a collection of responses
    """
    responses: List[str]    

# Attribute Class Object
@dataclasses.dataclass
class Attribute:
    """ 
    Importantly, buckets host responses which human evaluation is collected
    - These evaluation result will be hosted in the bucket for aligning the evaluator
    - Alignment happens both in prompt (few-shot), RAG (adaptive prompt), hidden representation (REFT), or in weights (FineTune)
    """
    name: str
    desc: str
    buckets: Dict[str, Bucket] = {"Unacceptable": Bucket(responses=[]), 
                                  "Ok": Bucket(responses=[]), 
                                  "Acceptable": Bucket(responses=[])}

    @classmethod
    def make(cls, data: dict[str, str], folder_dir: Optional[str] = None):
        """ 
        Make an attribute from dictionary
        """
        name = data.get('name', '')
        desc = data.get('desc', '')
        
        if folder_dir:
            try:
                return cls.load(folder_dir, name)
            except FileNotFoundError:
                pass
        
        return cls(name=name, desc=desc)
    
    def update_bucket(cls, bucket_name: str, response: str):
        """
        Update a bucket with a response
        """
        cls.buckets[bucket_name].responses.append(response)
        
    @classmethod
    def save(cls, folder_dir: str):
        """ 
        Save attribute and bucket
        """
        # Ensure the directory exists
        os.makedirs(folder_dir, exist_ok=True)
        # Create a dictionary to store the attribute data
        attribute_data = {
            "name": cls.name,
            "desc": cls.desc,
            "buckets": {
                bucket_name: [response for response in bucket.responses]
                for bucket_name, bucket in cls.buckets.items()
            }
        }
        # Save the attribute data as a JSON file
        file_path = os.path.join(folder_dir, f"{cls.name}.json")
        with open(file_path, 'w') as f:
            json.dump(attribute_data, f, indent=4)
            
    @classmethod
    def load(cls, folder_dir: str, attribute_name: str):
        """ 
        Load attribute and bucket from a JSON file
        """
        file_path = os.path.join(folder_dir, f"{attribute_name}.json")
        with open(file_path, 'r') as f:
            attribute_data = json.load(f)
        
        attribute = cls(name=attribute_data['name'], desc=attribute_data['desc'])
        for bucket_name, responses in attribute_data['buckets'].items():
            attribute.buckets[bucket_name] = Bucket(responses=responses)
        
        return attribute
    
# Formulation of Comparison Prompt: ToBeChanged
def construct_judge_prompt_from_scenarios_v3(compare_scenarios):
    if isinstance(compare_scenarios, dict):
        compare_scenarios = [compare_scenarios]
    prompt_parts = []
    for scenario in compare_scenarios:
        prompt_part = f"""
Attribute: {scenario['name']}
Description: {scenario['desc']}
Good Response Example: {scenario['good_response']}
Bad Response Example: {scenario['bad_response']}
"""
        if 'reflection' in scenario:
           prompt_part += f"Reflection: {scenario['reflection']}"
        prompt_parts.append(prompt_part)
    return ("".join(prompt_parts)).strip()
    
    
    
# Requirement Class Object
@dataclasses.dataclass
class Requirement:
    """ 
    Requirement is a collection of attributes
    """
    attributes: List[Attribute]
    
    @classmethod
    def make(cls, data: list[dict[str, str]]):
        attributes = [Attribute.make(item) for item in data]
        return cls(attributes=attributes)
    
    def add_scenario(self, scenario: dict[str, str]):
        self.attributes.append(Attribute(**scenario))
        
    def get_scenario_index(self, scenario_name: str):
        for idx in range(len(self.attributes)):
            if self.attributes[idx].name == scenario_name:
                return idx
        return -1

    def mutate_scenario(self, scenario: dict[str, str]) -> bool:
        index = self.get_scenario_index(scenario["name"])
        if index >= 0:
            self.attributes[index] = Attribute(**scenario)
            return True
        else:
            return False
        
    @property
    def attributes(self):
        attributes_str = "\n".join([f"{attribute.name}\n- Scenario: {attribute.scenario_desc}\n- Good Response: {attribute.good_response}\n- Bad Response: {attribute.bad_response}\n" for attribute in self.attributes])
        return attributes_str
    
    