from .attribute import Requirement, Attribute
from .model import get_claude_response
from typing import Callable, Union

# Two Approach to Evaluation 
# - 1. Direct Bucket Selection | Direct & Cheap
# - 2. Comparative Scoring with all availabel buckets and vote accordingly | More expensive & accuracy

direct_bucket_prompt_template = """Based on the attribute described as:
{info}

Evaluate the following conversation:
{conversation}

Provide your rationale for the evaluation, considering the attribute's description and the conversation content. Then, decide which bucket the conversation falls into.

Your response should be structured as follows:
1. Rationale: [Your detailed reasoning here]
2. Decision: [Chosen bucket]

Available buckets: Unacceptable / Ok / Acceptable

Example response format:
Rationale: [Your reasoning]
Decision: Ok
"""

compare_bucket_prompt_template = """Based on the attribute described as:
{info}

Compare the following two conversations:

Conversation A:
{conversation}

Conversation B:
{bucket_conversation}

Provide your rationale for the comparison, considering the attribute's description and the content of both conversations. Then, decide which conversation better aligns with the attribute's criteria.

Your response should be structured as follows:
1. Rationale: [Your detailed reasoning here]
2. Decision: [A or B]

Example response format:
Rationale: [Your reasoning]
Decision: A
"""


def parse_direct_bucket_response(response: str) -> Union[str, bool]:
    """ 
    Parse Rationale and Decision
    - Missing Decision / Rationale returns False 
    """
    if "Decision:" not in response or "Rationale:" not in response:
        return False
    try:
        rationale, decision = response.split("Decision:")
        rationale = rationale.strip()
        if "Rationale:" not in rationale:
            return False
        rationale = rationale.split("Rationale:")[1].strip()
        decision = decision.strip()
        
        if not rationale or not decision:
            return False
        
        if decision not in ["Unacceptable", "Ok", "Acceptable"]:
            return False
        
        return decision
    except ValueError:
        return False
    
    
def parse_compare_bucket_response(response: str) -> Union[str, bool]:
    """ 
    Parse Rationale and Decision
    - Missing Decision / Rationale returns False 
    """
    if "Decision:" not in response or "Rationale:" not in response:
        return False
    try:
        rationale, decision = response.split("Decision:")
        rationale = rationale.strip()
        if "Rationale:" not in rationale:
            return False
        rationale = rationale.split("Rationale:")[1].strip()
        decision = decision.strip()
        
        if not rationale or not decision:
            return False
        
        if decision not in ["A", "B"]:
            return False
        
        return decision
    except ValueError:
        return False
    

def direct_bucket_eval(conversation: str, attribute: Attribute) -> str:
    """ 
    Direct request for Absolute Scroing
    - Output is a bucket: Unacceptable / Ok / Acceptable
    """
    prompt = direct_bucket_prompt_template.format(
        info=attribute.info,
        conversation=conversation
    )
    response = get_claude_response(prompt)
    
    bucket = parse_direct_bucket_response(response.content[0].text)
    return bucket


def compare_bucket_eval(conversation: str, attribute: Attribute) -> str:
    """ 
    Compare request for Relative Scoring
    - Compared with available response in some bucket
    - We decide how many bucket conversation will be used, by default we use binary search
    - Output is a bucket: Unacceptable / Ok / Acceptable
    """
    # First, compare with Ok bucket
    ok_bucket = attribute.buckets["Ok"]
    prompt = compare_bucket_prompt_template.format(
        info=attribute.info,
        conversation=conversation,
        bucket_conversation=ok_bucket.get_random_response()
    )
    response = get_claude_response(prompt)
    choice = parse_compare_bucket_response(response.content[0].text)
    
    if choice == "A":  # If better than Ok
        # Compare with Acceptable bucket
        acceptable_bucket = attribute.buckets["Acceptable"]
        prompt = compare_bucket_prompt_template.format(
            info=attribute.info,
            conversation=conversation,
            bucket_conversation=acceptable_bucket.get_random_response()
        )
        response = get_claude_response(prompt)
        choice = parse_compare_bucket_response(response.content[0].text)
        return "Acceptable" if choice == "A" else "Ok"
    else:  # If not better than Ok
        # Compare with Unacceptable bucket
        unacceptable_bucket = attribute.buckets["Unacceptable"]
        prompt = compare_bucket_prompt_template.format(
            info=attribute.info,
            conversation=conversation,
            bucket_conversation=unacceptable_bucket.get_random_response()
        )
        response = get_claude_response(prompt)
        choice = parse_compare_bucket_response(response.content[0].text)
        return "Unacceptable" if choice == "B" else "Ok"


class AOEval:
    def __init__(self, requirement: Requirement):
        """ 
        Wrapper class which enables 
        - Human supervision
        - Bucket Update
        """
        self.requirement = requirement

    def direct_eval(self, conversation: str, attribute: Attribute) -> str:
        bucket = direct_bucket_eval(conversation, attribute)
        return bucket 
    
    def compare_eval(self, conversation: str, attribute: Attribute) -> str:
        try:
            bucket = compare_bucket_eval(conversation, attribute)
            return bucket
        except:
            # Likely having empty buckets up to now
            return self.direct_eval(conversation, attribute)

    def parse_bucket(self, response: str) -> str:
        for bucket in ["Unacceptable", "Ok", "Acceptable"]:
            if bucket.lower() in response.lower():
                return bucket
        return "Unknown"

    def human_annotation(self, conversation: str, attribute: Attribute, suggested_bucket: str) -> str:
        print(f"Attribute: {attribute.name}")
        print(f"Conversation:\n{conversation}")
        print(f"Suggested bucket: {suggested_bucket}")
        while True:
            human_choice = input("Accept (a) or change (c) the bucket? ").lower()
            if human_choice == 'a':
                return suggested_bucket
            elif human_choice == 'c':
                new_bucket = input("Enter the correct bucket (Unacceptable/Ok/Acceptable): ").capitalize()
                if new_bucket in ["Unacceptable", "Ok", "Acceptable"]:
                    return new_bucket
                else:
                    print("Invalid bucket. Please try again.")
            else:
                print("Invalid choice. Please enter 'a' or 'c'.")

    def evaluate_and_annotate(self, conversation: str):
        for attribute in self.requirement.attributes:
            suggested_bucket = self.compare_eval(conversation, attribute)
            final_bucket = self.human_annotation(conversation, attribute, suggested_bucket)
            attribute.update_bucket(final_bucket, conversation)

    def save(self):
        for attribute in self.requirement.attributes:
            attribute.save()