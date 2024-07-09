import dataclasses
from typing import List, Dict, Optional
import os
import json
import random

ATTRIBUTE_FOLDER = "data/attribute"

@dataclasses.dataclass
class Bucket:
    """ 
    Bucket is a collection of responses
    """
    responses: List[str]  
    
    def get_random_response(self):
        return random.choice(self.responses)

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
    buckets: Dict[str, Bucket] = dataclasses.field(default_factory=lambda: {
        "Unacceptable": Bucket(responses=[]), 
        "Ok": Bucket(responses=[]), 
        "Acceptable": Bucket(responses=[])
    })

    @classmethod
    def make(cls, data: dict[str, str], folder_dir: Optional[str] = ATTRIBUTE_FOLDER):
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
    
    def update_bucket(self, bucket_name: str, response: str):
        """
        Update a bucket with a response
        """
        self.buckets[bucket_name].responses.append(response)
        
    def save(self, folder_dir: str = ATTRIBUTE_FOLDER):
        """ 
        Save attribute and bucket
        """
        # Ensure the directory exists
        os.makedirs(folder_dir, exist_ok=True)
        # Create a dictionary to store the attribute data
        attribute_data = {
            "name": self.name,
            "desc": self.desc,
            "buckets": {
                bucket_name: [response for response in bucket.responses]
                for bucket_name, bucket in self.buckets.items()
            }
        }
        # Save the attribute data as a JSON file
        file_path = os.path.join(folder_dir, f"{self.name}.json")
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
    
    @property 
    def info(self):
        """ 
        Get judge prompt
        - Use buckets to formulate a judge prompt
        - Evaluator takes the information for each attribute and make a sound evaluation
        """
        prompt = f"Attribute: {self.name}\nDescription: {self.desc}\n\n"
        for bucket_name, bucket in self.buckets.items():
            prompt += f"{bucket_name} Response Examples:\n"
            for response in bucket.responses[:3]:  # Limit to 3 examples per bucket
                prompt += f"- {response}\n"
            prompt += "\n"
        return prompt.strip()
    
    
# Requirement Class Object
@dataclasses.dataclass
class Requirement:
    """ 
    Requirement is a collection of attributes
    """
    _attributes: List[Attribute]
    
    def __repr__(self):
        attribute_details = []
        for attr in self._attributes:
            bucket_info = ", ".join([f"{name}: {len(bucket.responses)} responses" for name, bucket in attr.buckets.items()])
            attribute_details.append(f"{attr.name} ({bucket_info})")
        return f"Requirement(attributes=[{', '.join(attribute_details)}])"
    
    @classmethod
    def make(cls, data: list[dict[str, str]]):
        attributes = [Attribute.make(item) for item in data]
        return cls(_attributes=attributes)
    
    def add_scenario(self, scenario: dict[str, str]):
        self._attributes.append(Attribute(**scenario))
        
    def get_scenario_index(self, scenario_name: str):
        for idx in range(len(self._attributes)):
            if self._attributes[idx].name == scenario_name:
                return idx
        return -1

    def mutate_scenario(self, scenario: dict[str, str]) -> bool:
        index = self.get_scenario_index(scenario["name"])
        if index >= 0:
            self._attributes[index] = Attribute(**scenario)
            return True
        else:
            return False
        
    @property
    def attributes(self):
        return self._attributes

    @property
    def attributes_str(self):
        return "\n".join([f"{attribute.name}\n- Scenario: {attribute.scenario_desc}\n- Good Response: {attribute.good_response}\n- Bad Response: {attribute.bad_response}\n" for attribute in self._attributes])