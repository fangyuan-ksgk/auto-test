from functools import partial
from typing import List, Tuple, Callable, Dict, Optional, Union
import pandas as pd
import os, itertools, json, glob, hashlib
from .attribute import AttributeTree, SimpleTree

concat_conversation = lambda conversation: ('||').join(conversation)
deconcat_conversation = lambda conversation: conversation.split('||')

# def generate_hash(conversation):
#     conversation = concat_conversation(conversation)
#     # Convert the conversation to a string representation if it's not already
#     conv_str = str(conversation)
#     # Create a hash of the conversation
#     return hashlib.sha256(conv_str.encode()).hexdigest()

def generate_hash(conversation):
    """
    Generate a unique hash for a given conversation by using the length of the conversation
    as a prefix to the concatenated conversation text. This approach aims to provide a distinctive
    ID to each conversation, ensuring different conversations have different hash IDs.

    Parameters:
    conversation (List[str]): A list of strings representing the conversation.

    Returns:
    str: A unique hash string representing the conversation.
    """
    # Concatenate the conversation into a single string
    conversation_str = '||'.join(conversation)
    # Use the length of the conversation as a prefix to create a unique identifier
    unique_identifier = f"{len(conversation_str)}-{conversation_str[::10][4:14]}-{conversation_str[::-1][1:5]}".replace(' ','').replace("'","").replace('"',"")
    # Generate the hash using hashlib.sha256
    return unique_identifier



def expand_unannotated(unannotated: pd.DataFrame):
    unannotated_combinations = []
    # Iterate through each row and column
    for attribute in unannotated.columns:    
        for index, row in unannotated.iterrows():   
            if not row[attribute]: 
                item = []
                for id in index:
                    item.append(id)
                item.append(attribute)
                unannotated_combinations.append(tuple(item))
    return unannotated_combinations

def create_temp_anno_record(N, num_comb=2):
    return {i : [0]*num_comb for i in range(N)}

import random

def parse_preference_conversations(folder_path: str = './data/preference/copan/synthetic_conversation/') -> List[str]:
    conversation_files = glob.glob(f'{folder_path}synthetic_scenario_*.json')
    conversations = []
    for conversation_file in conversation_files:
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
            conversation = [s for s in conversation.split('\n') if s != '']
            conversations.append(conversation)
    return random.choices(conversations, k=10)    

def parse_conversations_attribute(folder_path: str = './data/conversation/', attribute: str = 'specific') -> List[str]:
    conversation_files = glob.glob(f'{folder_path}{attribute}*.json')
    conversations = []
    for conversation_file in conversation_files:
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
            conversations.append(conversation)
    n = len(conversation_files)
    return conversations
    # return random.choices(conversations, k=min(10, n))


# Parse whatever dataset of conversations into list of conversations
# This one works if your conversatons is stored in a folder of json files
def parse_conversations(folder_path: str = './data/conversation/') -> List[str]:
    conversation_files = glob.glob(f'{folder_path}conversation_*.json')
    conversations = []
    for conversation_file in conversation_files:
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
            conversations.append(conversation)
    return random.choices(conversations, k=10)
    # return conversations


def remove_duplicate_in_hash_dict(hash_dict, storage_ids):
    # List to keep track of hash_ids to be removed
    hash_ids_to_remove = []

    # Check HashID against duplications
    for hash_id in hash_dict:
        if hash_id in storage_ids:
            # Add hash_id to the list for removal
            hash_ids_to_remove.append(hash_id)

    # Remove the identified hash_ids from hash_dict
    for hash_id in hash_ids_to_remove:
        hash_dict.pop(hash_id)
    return hash_dict

parse_new_conversations = partial(parse_conversations, folder_path='./data/conversation/')
parse_new_conversations_attribute = partial(parse_conversations_attribute, folder_path='./data/fwd-customer/')


class PoeBaseDataset:

    def __init__(self, conversations: pd.DataFrame, 
                 annotations: pd.DataFrame, 
                 annotator_info: str, 
                 attribute_info: Dict,
                 store_dir: str,
                 prefix: str,
                 num_comb: int = 2):
        # Storage directory
        self.store_dir = store_dir
        # Conversations
        self.conversations = conversations
        # Annotations
        self.annotations = annotations
        # Annotator Info
        self.annotator_info = annotator_info
        # Attribute Info
        self.attribute_info = attribute_info
        # Prefix
        self.prefix = prefix
        # number of combinations
        self.num_comb = num_comb
        # Prepare for annotation -- only on pair-wise comparison not done yet
        unannotated = self.get_unannotated_pairs_attributes()
        self._prepare_anno()

    def get_unannotated_pairs_attributes(self) -> pd.DataFrame:
        # get all the pairs of conversations
        hash_id_pairs = list(itertools.combinations(list(self.conversations['hash_id']), 2))
        # get all the attributes
        attributes = list(self.attribute_info.keys())
        # Create a MultiIndex
        multi_index = pd.MultiIndex.from_tuples(hash_id_pairs, names=['hash_id_a', 'hash_id_b'])
        # Initialize an empty DataFrame with this MultiIndex and attributes as columns
        self.anno_info = pd.DataFrame(index=multi_index, columns=attributes, dtype=bool, data=False)

        # Not all the conversation pairs require annotation, so this anno_info dataframe should be changed. 
        # The self.annotations on the other hand, is directly loaded from a csv file, so it should be kept as is.

        # get all the pairs of conversations that have not been annotated for each attribute, default value to False
        # loop through current annotations to get the annotated pairs
        for idx, row in self.annotations.iterrows():
            hash_id_a = row['hash_id_a']
            hash_id_b = row['hash_id_b']
            # get the pair of conversations
            pair = (hash_id_a, hash_id_b)
            # get the attribute
            attribute = row['attribute']
            
            # mark the pair of conversations as annotated for the attribute
            self.anno_info.loc[pair, attribute] = True



        # get the un-annotated pairs of conversations || those entry with False value
        self.unannotated = self.anno_info[~self.anno_info.any(axis=1)]
        # self.unannotated = self.anno_info[self.anno_info.isfalse().any(axis=1)]
        return self.unannotated
        

    @classmethod
    def load_conversations(cls, store_dir: str) -> pd.DataFrame:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversation_file_path = f'{store_dir}conversations.csv'
        if not os.path.exists(conversation_file_path):
            conversations = pd.DataFrame(columns=['hash_id', 'conversation'])
        else:
            conversations = pd.read_csv(conversation_file_path)
        return conversations
    
    @classmethod
    def load_specific_conversations(cls, store_dir: str, prefix: str) -> pd.DataFrame:
        """
        Load conversations from files that start with a specific prefix.
        
        Args:
            store_dir (str): The directory where the conversation files are stored.
            prefix (str): The prefix of the file names to load.
        
        Returns:
            pd.DataFrame: A dataframe containing the conversations with columns ['hash_id', 'conversation'].
        """
        # Initialize an empty DataFrame to store conversations
        conversations = pd.DataFrame(columns=['hash_id', 'conversation'])
        # List all files in the directory
        for file_name in os.listdir(store_dir):
            # Check if the file name starts with the given prefix
            if file_name.startswith(prefix) and file_name.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(store_dir, file_name)
                # Load the conversations from the file
                current_conversations = pd.read_csv(file_path)
                # Append the loaded conversations to the main DataFrame
                conversations = pd.concat([conversations, current_conversations], ignore_index=True)
        return conversations

    
    @classmethod
    def preprocess(cls, store_dir: str, prefix: str = '') -> list[str]:
        """
        Preprocess the conversations (CSV stored) and return a list of conversations
        """
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        if prefix == '':
            df = cls.load_conversations(store_dir)
        else:
            print('Store dir: ', store_dir)
            print('Prefix: ', prefix)
            df = cls.load_specific_conversations(store_dir, prefix)
            
        conversations = [deconcat_conversation(concated_conversation) for concated_conversation in list(df.conversation)]
        return conversations
    
    @classmethod
    def save_conversations(cls, conversations: pd.DataFrame, store_dir: str, prefix: str = "conversations") -> None:
        # Save the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversation_file_path = f'{store_dir}/{prefix}.csv'
        conversations.to_csv(conversation_file_path, index=False)

    @classmethod
    def load_annotations(cls, store_dir: str, annotator_info: str, prefix: str = '', num_comb: int = 2) -> pd.DataFrame:
        # Load the annotations as pandas dataframe || (hash_id, hash_id), preference
        """
        Load the annotations if exist, initialize an empty dataframe otherwise.
        """
        # print('Loaded Number Comb: ', num_comb)
        anno_file_path = f'{store_dir}{prefix}annotation_{annotator_info}.csv'
        i_map = {0: 'a', 1: 'b', 2:'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}
        column_names = ['hash_id_'+i_map[i] for i in range(num_comb)] + ['preference', 'attribute']
        # print('load annotations: ', column_names)
        if not os.path.exists(anno_file_path):
            # print('Annotation file does not exist.')
            annotations = pd.DataFrame(columns=column_names)
        else:
            annotations = pd.read_csv(anno_file_path)
        return annotations
    
    def save_annotations(self) -> None:
        # Save the annotations as pandas dataframe || (hash_id, hash_id), preference
        anno_file_path = f'{self.store_dir}{self.prefix}annotation_{self.annotator_info}.csv'
        self.annotations.to_csv(anno_file_path, index=False)

    @classmethod
    def load(cls, store_dir: str, annotator_info: str, attribute_info: Dict, prefix: str = 'conversations', load_dir: str = './data/fwd-customer/', num_comb: int = 2) -> 'POEDataset':
        # Load the conversations
        conversations = cls.load_specific_conversations(load_dir, prefix)
        # Load the annotations
        # print('Num combinations: ', num_comb)
        annotations = cls.load_annotations(store_dir, annotator_info, num_comb=num_comb)
        # Create the dataset
        return cls(conversations, annotations, annotator_info, attribute_info, store_dir, prefix, num_comb)
    
    
    @classmethod
    def load_storage_ids(cls, store_dir: str = './data/annotation/') -> List[str]:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversations = cls.load_conversations(store_dir)
        if len(conversations) > 0:
            return conversations['hash_id'].tolist()
        else:
            return []
    
    @classmethod
    def merge_conversations(cls, store_dir: str, hash_dict: Dict[str, List[str]], prefix: str = "conversations", return_df: bool = False) -> Optional[pd.DataFrame]:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversations = cls.load_specific_conversations(store_dir, prefix)
        # Convert the hash_dict to a dataframe
        new_conversations = pd.DataFrame(hash_dict.items(), columns=['hash_id', 'conversation'])
        # Merge the new conversations into the storage
        conversations = pd.concat([conversations, new_conversations], ignore_index=True)
        # Save the new conversations
        cls.save_conversations(conversations, store_dir, prefix)
        if return_df:
            return conversations
        
    # @classmethod
    # def make_with_conversations(cls, parse_new_conversations: Callable,
    #            scenaiors: List[dict[str, str]]):

    @classmethod
    def make_raw(cls, parse_new_conversation: Callable,
                 scenarios: List[dict],
                 store_dir: str = './data/annotation/'):
        raw_conversations = parse_new_conversations()
        print('Total number of parsed conversations: ', len(raw_conversations))
        hash_dict = {generate_hash(conversation): concat_conversation(conversation) for conversation in raw_conversations}
        # Merge the new conversations into the storage
        cls.merge_conversations(store_dir, hash_dict)

    @classmethod
    def parse_conversation(cls, conversations, store_dict: str = './data/annotation/', prefix: str ="conversations"):
        hash_dict = {generate_hash(conversation): concat_conversation(conversation) for conversation in conversations}
        cls.merge_conversations(store_dict, hash_dict, prefix)
                

    @classmethod
    def make(cls, parse_new_conversations: Union[Callable, None],
             attribute_tree: Union[AttributeTree, SimpleTree], 
             annotator_info: str, 
             store_dir: str = './data/annotation/',
             prefix: str = "conversations"):
        
        # Load in-storage ids
        storage_ids: List[str] = cls.load_storage_ids(store_dir)

        # Hash the new-conversations (potentially not in storage)
        if parse_new_conversations is None:
            raw_conversations = parse_new_conversations()
            print('Total number of parsed conversations: ', len(raw_conversations))
            hash_dict = {generate_hash(conversation): concat_conversation(conversation) for conversation in raw_conversations}
            
            # Check HashID against duplications
            hash_dict = remove_duplicate_in_hash_dict(hash_dict, storage_ids)

            # Merge the new conversations into the storage
            cls.merge_conversations(store_dir, hash_dict)


        # get the attribute leaf nodes | currently the subjective score is not utilized
        attribute_leaf_nodes = attribute_tree.get_leaf_nodes()
        attribute_info = {node.value: node.subjective_score for node in attribute_leaf_nodes}

        # Load the dataset
        return cls.load(store_dir, annotator_info, attribute_info, prefix)

    def retrieve_conversation(self, hash_id):
        conversation = self.conversations[self.conversations['hash_id']==hash_id]['conversation'].iloc[0]
        return deconcat_conversation(conversation)
    
    # So, During Annotation, we evaluate each attribute independently 
    # -- loop through attribute, subloop through pairs of conversation

    def _prepare_anno(self) -> None:
        self.update_annotation_summary()
        self.unannotated_combinations = expand_unannotated(self.unannotated)
        num_comb = self.num_comb
        self.temp_anno_record = create_temp_anno_record(len(self.unannotated_combinations), num_comb)

    def _idx_to_hash_id_pair(self, idx: int) -> Tuple[int, int]:
        tuples = self.unannotated_combinations[idx]
        return tuples[:-1]

    # def _pair_idx_to_idx(self, pair_idx: Tuple[int, int]) -> int:
        # return self.indices.index(pair_idx)

    def __len__(self) -> int:
        return len(self.unannotated_combinations)
    
    def annotate(self, idx: int, preference) -> None:
        # annotation goes into temporay buffer to record one-time selection (change is possible)
        self.temp_anno_record[idx] = preference

    def _cache_anno(self) -> None:
        # Cache the annotation
        for idx, preference in self.temp_anno_record.items():
            print("Line 366 -- preference: ", preference)
            if (preference[0] + preference[1] + preference[2]) == 0:
                continue
            tuple = self.unannotated_combinations[idx]
            anno = list(tuple[:-1]) + [preference] + [tuple[-1]]
            self.annotations.loc[len(self.annotations)] = anno
        # Save the annotation
        self.save_annotations()
        # Update the unannotated pairs
        self.unannotated = self.get_unannotated_pairs_attributes()
        # Prepare for annotation
        self._prepare_anno()

    def save(self) -> None:
        # Cache the annotation
        self._cache_anno()

    def update_annotation_summary(self):
        # Update the annotation summary
        # self.annotation_summary = self.annotations.groupby(['hash_id_a', 'hash_id_b']).agg({'preference': 'sum'}).reset_index()
        # self.annotation_summary['preference'] = self.annotation_summary['preference'].apply(lambda x: 1 if x > 0 else -1)
        # self.annotation_summary = self.annotation_summary.rename(columns={'preference': 'preference_sum'})
        pass

 



def combine_df(annotation, conversation):
    # Combine the annotation and conversation dataframes and select the required columns
    combined_df = annotation.merge(conversation, left_on='hash_id_a', right_on='hash_id', how='left')
    combined_df.rename(columns={'conversation': 'conversation_a'}, inplace=True)
    combined_df = combined_df.merge(conversation, left_on='hash_id_b', right_on='hash_id', how='left')
    combined_df.rename(columns={'conversation': 'conversation_b'}, inplace=True)
    combined_df = combined_df[['conversation_a', 'conversation_b', 'hash_id_a', 'hash_id_b', 'preference', 'attribute']]
    return combined_df

class POEDataset(PoeBaseDataset):
    def __getitem__(self, idx):
        if idx < len(self.unannotated_combinations):
            hash_id_a, hash_id_b, attribute = self.unannotated_combinations[idx]
            return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), attribute
        else:
            raise IndexError("Index out of range")
    def __iter__(self):
        self._iter_idx = 0
        return self
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        self._iter_idx += 1
        hash_id_a, hash_id_b, attribute = self.unannotated_combinations[self._iter_idx]
        return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), attribute
    
    @classmethod
    def load_annotated_dataset(cls, store_dir: str, annotater_info: str) -> pd.DataFrame:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        annotation = cls.load_annotations(store_dir, annotater_info)
        conversation = cls.load_conversations(store_dir)
        return combine_df(annotation, conversation)
    
    @classmethod 
    def lazy_load_annotated_dataset(cls, store_dir: str, anno_path: str) -> pd.DataFrame:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        assert anno_path.endswith('.csv'), 'The annotation file should be a csv file'
        prefix, annotator_info = anno_path[:-4].split('annotation_')
        annotation = cls.load_annotations(store_dir=store_dir, annotator_info=annotator_info, prefix=prefix)
        conversation = cls.load_conversations(store_dir)

    
def filter_indices(grouped_indices, n_gen_response=3):
    filtered_indices = []
    for key, value in grouped_indices.items():
        if len(value) == n_gen_response:
            filtered_indices.append(value)
        else:
            continue
    return filtered_indices

def get_annotation_indices(df, n_gen_response=3, n_compare=2):
    df['query'] = df.apply(lambda x: x['conversation'].split('||')[0], axis=1)
    grouped_indices = df.groupby('query').apply(lambda x: list(x.index)).to_dict()
    annotation_indices = []
    indices = filter_indices(grouped_indices, n_gen_response=3)
    for triplet in indices:
        annotation_indices.extend(list(itertools.combinations(triplet, n_compare)))
    return annotation_indices


def get_annotation_hash_id_pairs(df, n_gen_response=3, n_compare=2):
    annotated_indices = get_annotation_indices(df, n_gen_response=n_gen_response, n_compare=n_compare)
    # get hash id pairs 
    hash_id_pairs = []
    for pair in annotated_indices:
        ids = []
        for idx in pair:
            ids.append(df.loc[idx, 'hash_id'])
        hash_id_pairs.append(ids)
    return hash_id_pairs


class PoeBaseDataset_Zero(PoeBaseDataset):
    def get_unannotated_pairs_attributes(self) -> pd.DataFrame:
        # get all the pairs of query-uniform conversations
        hash_id_pairs = get_annotation_hash_id_pairs(self.conversations, n_gen_response=3, n_compare=3)
        # get all the attributes
        attributes = list(self.attribute_info.keys())
        # Create a MultiIndex
        multi_index = pd.MultiIndex.from_tuples(hash_id_pairs, names=['hash_id_a', 'hash_id_b', 'hash_id_c'])
        # Initialize an empty DataFrame with this MultiIndex and attributes as columns
        self.anno_info = pd.DataFrame(index=multi_index, columns=attributes, dtype=bool, data=False)

        # get all the pairs of conversations that have not been annotated for each attribute, default value to False
        # loop through current annotations to get the annotated pairs
        for idx, row in self.annotations.iterrows():
            hash_id_a = row['hash_id_a']
            hash_id_b = row['hash_id_b']
            hash_id_c = row['hash_id_c']
            # get the pair of conversations
            pair = (hash_id_a, hash_id_b, hash_id_c)
            # get the attribute
            attribute = row['attribute']

            # mark the pair of conversations as annotated for the attribute
            self.anno_info.loc[pair, attribute] = True
        # get the un-annotated pairs of conversations || those entry with False value
        self.unannotated = self.anno_info[~self.anno_info.any(axis=1)]
        return self.unannotated
    
    def __init__(self, conversations: pd.DataFrame, 
                 annotations: pd.DataFrame, 
                 annotator_info: str, 
                 attribute_info: Dict,
                 store_dir: str,
                 prefix: str,
                 num_comb: int = 2):
        # Storage directory
        self.store_dir = store_dir
        # Conversations
        self.conversations = conversations
        # Annotations
        self.annotations = annotations
        # Annotator Info
        self.annotator_info = annotator_info
        # Attribute Info
        self.attribute_info = attribute_info
        # Prefix
        self.prefix = prefix
        # number of combinations
        self.num_comb = num_comb
        # Prepare for annotation -- only on pair-wise comparison not done yet
        unannotated = self.get_unannotated_pairs_attributes()
        self._prepare_anno()
    
    def __len__(self) -> int:
        return len(self.unannotated_combinations)


class PoeBaseDataset_v2(PoeBaseDataset):

    def get_unannotated_pairs_attributes(self) -> pd.DataFrame:
        # get all the pairs of query-uniform conversations
        hash_id_pairs = get_annotation_hash_id_pairs(self.conversations, n_gen_response=3)
        # get all the attributes
        attributes = list(self.attribute_info.keys())
        # Create a MultiIndex
        multi_index = pd.MultiIndex.from_tuples(hash_id_pairs, names=['hash_id_a', 'hash_id_b'])
        # Initialize an empty DataFrame with this MultiIndex and attributes as columns
        self.anno_info = pd.DataFrame(index=multi_index, columns=attributes, dtype=bool, data=False)

        # get all the pairs of conversations that have not been annotated for each attribute, default value to False
        # loop through current annotations to get the annotated pairs
        for idx, row in self.annotations.iterrows():
            hash_id_a = row['hash_id_a']
            hash_id_b = row['hash_id_b']
            # get the pair of conversations
            pair = (hash_id_a, hash_id_b)
            # get the attribute
            attribute = row['attribute']
            
            # mark the pair of conversations as annotated for the attribute
            self.anno_info.loc[pair, attribute] = True

        # get the un-annotated pairs of conversations || those entry with False value
        self.unannotated = self.anno_info[~self.anno_info.any(axis=1)]
        # self.unannotated = self.anno_info[self.anno_info.isfalse().any(axis=1)]
        return self.unannotated
    

class POEDataset_Zero(PoeBaseDataset_Zero):

    def __len__(self):
        return len(self.unannotated)

    def __getitem__(self, idx):
        if idx < len(self.unannotated_combinations):
            hash_id_a, hash_id_b, hash_id_c, attribute = self.unannotated_combinations[idx]
            return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), self.retrieve_conversation(hash_id_c), attribute
        else:
            raise IndexError("Index out of range")
        
    def __iter__(self):
        self._iter_idx = 0
        return self
    
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        self._iter_idx += 1
        hash_id_a, hash_id_b, hash_id_c, attribute = self.unannotated_combinations[self._iter_idx]
        return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), self.retrieve_conversation(hash_id_c), attribute
    
    @classmethod
    def load_annotated_dataset(cls, store_dir: str, annotater_info: str) -> pd.DataFrame:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        annotation = cls.load_annotations(store_dir, annotater_info)
        conversation = cls.load_conversations(store_dir)
        return combine_df(annotation, conversation)
    
    # Convert to Dictionary
    # 6 evaluation results -> 
    def prepare_poe_dict(self, idx):
        attribute = self.prefix
        conv1, conv2, conv3, annotation_question = self[idx]
        poe_dict = {
            "attribute": attribute.replace("_", " "),
            "question": annotation_question,
            "sale_query": conv1[0],
            "evaluation": [
                {
                    "id": 1,
                    "response": conv1[1],
                    "status": "not_evaluated"
                },
                {
                    "id": 2,
                    "response": conv2[1],
                    "status": "not_evaluated"
                },
                {
                    "id": 3,
                    "response": conv3[1],
                    "status": "not_evaluated"
                }
            ]
        }
        return poe_dict

    # Append all the PoE dictionary into a list 
    def prepare_poe_list(self):
        poe_list = []
        for idx in range(len(self)):
            poe_dict = self.prepare_poe_dict(idx)
            # poe_dict["attribute"] = 
            poe_dict["id"] = idx
            poe_list.append(poe_dict)
        return poe_list


class POEDataset_v2(PoeBaseDataset_v2):

    def __getitem__(self, idx):
        if idx < len(self.unannotated_combinations):
            hash_id_a, hash_id_b, attribute = self.unannotated_combinations[idx]
            return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), attribute
        else:
            raise IndexError("Index out of range")
    def __iter__(self):
        self._iter_idx = 0
        return self
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        self._iter_idx += 1
        hash_id_a, hash_id_b, attribute = self.unannotated_combinations[self._iter_idx]
        return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), attribute
    
    @classmethod
    def load_annotated_dataset(cls, store_dir: str, annotater_info: str) -> pd.DataFrame:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        annotation = cls.load_annotations(store_dir, annotater_info)
        conversation = cls.load_conversations(store_dir)
        return combine_df(annotation, conversation)

    

def parse_conversation_into_name_and_messages(conversation):
    try:
        name, message = conversation.split(':')
    except:
        if 'Sale:' in conversation:
            name = 'Sale'
            message = conversation.split('Sale:')[1]
        elif 'Customer:' in conversation:
            name = 'Customer'
            message = conversation.split('Customer:')[1]
        else:
            print('Issuing Conversation: ', conversation)
    return name, message

def get_id_pairs(conversations_length):
    """
    Return non-repeating pairs of conversation ids
    """
    from itertools import combinations
    return list(combinations(range(conversations_length), 2))