from .pairmatch import pairmatch_baseline
from .batch_pairmatch import pairmatch_batch, pairmatch_base, build_compare_tools, build_compare_tools_claude3, pairmatch_claude
from .dataset import generate_hash, concat_conversation, deconcat_conversation, get_id_pairs
from typing import List, Tuple, Dict, Any, Optional, Callable
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from .preference import Requirement
from .decode import load_hf_model, pairmatch_decode

class MultiAttributeCompare:

    def __init__(self, conversations_list, attributes: List[str], pair_compare_fn: Optional[Callable] = None):
        self.hash_dict = {generate_hash(conversation): concat_conversation(conversation) for conversation in conversations_list}
        self.hash_ids = list(self.hash_dict.keys())
        self.cached_memory = pd.DataFrame(columns=['hash_id_a', 'hash_id_b', 'score', 'rationale', 'attribute'])
        self.pair_compare_fn = pair_compare_fn
        self.attributes = attributes
        self.ranking = {}

    def _get_conversation(self, hash_id):
        return deconcat_conversation(self.hash_dict[hash_id])

    def _get_pair(self, id1, id2):
        return (self._get_conversation(self.hash_ids[id1]), self._get_conversation(self.hash_ids[id2]))
    
    def _check_compare_result_exist_in_cached_memory(self, id1, id2):
        hash_id_a, hash_id_b = self.hash_ids[id1], self.hash_ids[id2]
        memory = self.cached_memory[self.cached_memory.apply(lambda row: (row['hash_id_a'] == hash_id_a) & (row['hash_id_b'] == hash_id_b), axis=1)]
        if len(memory) > 0:
            return memory
        else:
            return None
        
    def _store_compare_result_in_cached_memory(self, id1, id2, score, attributes, reason: str = "NA"):
        hash_id_a, hash_id_b = self.hash_ids[id1], self.hash_ids[id2]
        self.cached_memory.loc[len(self.cached_memory)] = [hash_id_a, hash_id_b, score, reason, attributes]

    def _get_compare_result_from_cached_memory(self, id1, id2):
        hash_id_a, hash_id_b = self.hash_ids[id1], self.hash_ids[id2]
        memory_df = self.cached_memory[self.cached_memory.apply(lambda row: (row['hash_id_a'] == hash_id_a) & (row['hash_id_b'] == hash_id_b), axis=1)]
        scores = {}
        for attribute in self.attributes:
            scores[attribute] = memory_df[memory_df['attribute'] == attribute]['score'].values[0]
        return scores
    
    def _save_cached_memory(self, path: str):
        self.cached_memory.to_csv(path, index=False)

    def _load_cached_memory(self, path: str):
        self.cached_memory = pd.read_csv(path)
    
    def __call__(self, id1, id2):
        memory = self._check_compare_result_exist_in_cached_memory(id1, id2)
        if memory is not None:
            scores = self._get_compare_result_from_cached_memory(id1, id2)
        else:
            scores = self._compare_pair(id1, id2)
        return scores
        
    def get_cached_memory(self):
        return self.cached_memory
    
    def _compare_pair(self, id1, id2):
        conversations_pair = self._get_pair(id1, id2)
        try:
            scores, reasons = self.pair_compare_fn(conversations_pair[0], conversations_pair[1])
        except:
            scores = self.pair_compare_fn(conversations_pair[0], conversations_pair[1])
            
        for attribute, score in scores.items():
            rationale = reasons[attribute]
            self._store_compare_result_in_cached_memory(id1, id2, score, attribute, rationale)
        return scores
    
    def _compare_pair_attribute_A_less(self, id1, id2, attribute_name: str):
        scores = self.__call__(id1, id2)
        s = scores[attribute_name]
        return s[0] <= s[1]
    
    def _merge_attribute(self, arr, l, m, r, attribute_name: str = None):
        n1 = m - l + 1
        n2 = r - m
        
        L = [0] * n1
        R = [0] * n2

        for i in range(n1):
            L[i] = arr[l + i]

        for j in range(n2):
            R[j] = arr[m + 1 + j]

        i = 0
        j = 0
        k = l

        while i < n1 and j < n2:
            if self._compare_pair_attribute_A_less(L[i], R[j], attribute_name):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
    
        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1

    def _merge_sort_attribute(self, arr, l, r, attribute_name: str):
        if l < r:
            m = l + (r - l) // 2
            self._merge_sort_attribute(arr, l, m, attribute_name)
            self._merge_sort_attribute(arr, m + 1, r, attribute_name)
            self._merge_attribute(arr, l, m, r, attribute_name)

    def compare_specific(self, id_pairs: List[Tuple[int, int]]):
        for id1, id2 in tqdm(id_pairs, desc="Comparing Specific Pairs"):
            attempts = 0
            success = False
            while attempts < 3 and not success:
                try:
                    self._compare_pair(id1, id2)
                    success = True
                except Exception as e:
                    print(f"Attempt {attempts + 1} failed: {e}")
                    time.sleep(1)
                    attempts += 1
            if not success:
                print(f"Failed to compare pair after {attempts} attempts.")

        return self.cached_memory.copy().reset_index(drop=True)
    
    def compare_specific_quiet(self, id_pairs: List[Tuple[int, int]]):
        for id1, id2 in (id_pairs):
            attempts = 0
            success = False
            while attempts < 3 and not success:
                try:
                    self._compare_pair(id1, id2)
                    success = True
                except Exception as e:
                    print(f"Attempt {attempts + 1} failed: {e}")
                    time.sleep(1)
                    attempts += 1
            if not success:
                print(f"Failed to compare pair after {attempts} attempts.")

        return self.cached_memory.copy().reset_index(drop=True)
    
    def pairwise_compare(self):
        num_conversations = len(self.hash_ids)
        id_pairs = get_id_pairs(num_conversations)
        pred_result = self.compare_specific(id_pairs)
        return pred_result

    def sort(self, attributes: Optional[List[str]] = None):
        if attributes is None:
            attributes = self.attributes
        for attribute in attributes:
            indices_arr = [i for i in range(len(self.hash_ids))]
            self._merge_sort_attribute(indices_arr, 0, len(indices_arr) - 1, attribute)
            self.ranking[attribute] = indices_arr
            indices_arr = np.array(indices_arr) + 0.2
            norm_score = indices_arr / indices_arr.sum()
            self.ranking[attribute] = norm_score.tolist()
        return self.ranking

class AOEval(MultiAttributeCompare):
    def __init__(self, conversations_list, requirements: Requirement, model_name: str = 'gpt-4',
                 model: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.requirements = requirements
        self.attributes = self.requirements.to_attribute_name_list()
        self.judge_prompts = self.requirements.to_judge_prompts()
        super().__init__(conversations_list, self.attributes)
        self.model_name = model_name
        if model_name == 'gpt-4':
            self.pair_compare_fn = self._pair_compare_fn_with_gpt4
        elif model_name == 'claude3':
            self.pair_compare_fn = self._pair_compare_fn_with_claude3
        else:
            if model is None or tokenizer is None:
                self.model, self.tokenizer = load_hf_model(model_name)
            else:
                self.model, self.tokenizer = model, tokenizer
            self.pair_compare_fn = self._pair_compare_fn_with_local_llm

    @property
    def compare_tools(self):
        if self.model_name == 'gpt-4':
            return build_compare_tools(self.attributes, self.judge_prompts)
        if self.model_name == 'claude3':
            return build_compare_tools_claude3(self.attributes, self.judge_prompts)
    
    def _pair_compare_fn_with_claude3(self, cA, cB):
        return pairmatch_claude(conversation_pairs=(cA, cB), attributes=self.attributes, compare_tools=self.compare_tools)

    def _pair_compare_fn_with_gpt4(self, cA, cB):
        print("Calling pairmatch base")
        return pairmatch_base(conversation_pairs=(cA, cB), attributes=self.attributes, compare_tools=self.compare_tools, judge_prompts = self.judge_prompts)
    
    def _pair_compare_fn_with_local_llm(self, cA, cB):
        return pairmatch_decode(conversation_pairs=(cA, cB), requirements=self.requirements, model=self.model, tokenizer=self.tokenizer)
    
    @classmethod
    def error_analysis(cls, pred_result, annotated_dataset, gt_attribute_to_pred_attribute, model_name: str = 'gpt-4'):
        preference_dict = {
            (row['hash_id_a'], row['hash_id_b'], gt_attribute_to_pred_attribute[row['attribute']]): row['preference']
            for index, row in annotated_dataset.iterrows()
        }
        pred_result['preference'] = pred_result.apply(
            lambda row: preference_dict.get((row['hash_id_a'], row['hash_id_b'], row['attribute'])), axis=1
        )
        if model_name != 'gpt-4':
            convert_score = lambda score: [np.round(s.item(),4) if isinstance(s, torch.Tensor) else np.round(s,4) for s in score]
            pred_result['score'] = pred_result.apply(lambda x: convert_score(x['score']), axis=1)
        return pred_result
    
    def get_case_study(self, idx, pred_result = None):
        if pred_result is None:
            pred_result = self.cached_memory.copy()
        conversation_a = self._get_conversation(pred_result.iloc[idx].hash_id_a)
        conversation_b = self._get_conversation(pred_result.iloc[idx].hash_id_b)
        rationale = pred_result.iloc[idx].rationale
        attribute = pred_result.iloc[idx].attribute
        score = pred_result.iloc[idx].score
        judge_map = {0.0: 'V2 Better', 0.5: 'Tie', 1.0: 'V1 Better'}
        judgement = judge_map[score[0]]
        return conversation_a, conversation_b, rationale, judgement, attribute