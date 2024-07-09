import dataclasses
import json
import re, glob
from .dataset import POEDataset
import ast
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist



@dataclasses.dataclass
class Attribute:
   name: str
   desc: str
   good_response: str
   bad_response: str
   scenario_desc: str
   weight: float = 1.0
   auto_compare_query: str = None # Use in auto-evaluaotor comparison
   anno_compare_query: str = None # Use in annotator comparison

   @classmethod
   def make(cls, data: dict[str, str]):
      if 'name' in data:
         name = data['name']
      if 'desc' in data:
         desc = data['desc']
      if 'good_response' in data:
         good_response = data['good_response']
      if 'bad_response' in data:
         bad_response = data['bad_response']
      if 'scenario_desc' in data:
         scenario_desc = data['scenario_desc']
      if 'weight' in data:
         weight = float(data['weight'])
      else:
         weight = 1.0
      if 'auto_compare_query' in data:
         auto_compare_query = data['auto_compare_query']
      if 'anno_compare_query' in data:
         anno_compare_query = data['anno_compare_query']
      return cls(name=name, desc=desc, good_response=good_response, bad_response=bad_response, scenario_desc=scenario_desc, weight=weight, auto_compare_query=auto_compare_query, anno_compare_query=anno_compare_query)

# This one is probably not that good :: I would neeed a better one here
def construct_judge_prompt_from_scenarios(compare_scenarios):
    if isinstance(compare_scenarios, dict):
        compare_scenarios = [compare_scenarios]
    prompt_parts = []
    for scenario in compare_scenarios:
        prompt_part = f"""
Attribute: {scenario['name']}
Detail: {scenario['desc']}
Good Response Example: {scenario['good_response']}
Bad Response Example: {scenario['bad_response']}
"""
        if 'reflection' in scenario:
           prompt_part += f"Reflection: {scenario['reflection']}"
        prompt_parts.append(prompt_part)
    return ("".join(prompt_parts)).strip()

def construct_judge_prompt_from_scenarios_v2(compare_scenarios):
    """
    This one is still too long for GPT-4 (New version)
    """
    if isinstance(compare_scenarios, dict):
        compare_scenarios = [compare_scenarios]
    prompt_parts = []
    for scenario in compare_scenarios:
        prompt_part = f"""
Attribute: {scenario['name']}
Description: {scenario['desc']}\n
"""
        if 'reflection' in scenario:
           prompt_part += f"""{scenario['reflection']}"""
        prompt_parts.append(prompt_part)
    return ("".join(prompt_parts)).strip()


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


@dataclasses.dataclass
class Requirement:
   attributes: list[Attribute] # this is a concise name for the attribute
   
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
      
   def get_anno_compare_queries(self):
      return [attribute.anno_compare_query for attribute in self.attributes]
   
   def get_auto_compare_queries(self):
      return [attribute.auto_compare_query for attribute in self.attributes]
   
   def get_attribute_names(self):
      return [attribute.name for attribute in self.attributes]
   
   def to_attribute_name_list(self):
      return [attribute.name for attribute in self.attributes]
   
   def to_attribute_dict(self):
      return [{"name": a.name, "desc": a.desc, "scenario_desc": a.scenario_desc, "good_response": a.good_response, "bad_response": a.bad_response} for a in self.attributes]
   
   def to_judge_prompts(self):
      return [construct_judge_prompt_from_scenarios_v3(a) for a in self.to_attribute_dict()]
   
   @property
   def show_attributes(self):
      show_attributes = "\n".join([f"{attribute.name}\n- Scenario: {attribute.scenario_desc}\n- Good Response: {attribute.good_response}\n- Bad Response: {attribute.bad_response}\n" for attribute in self.attributes])
      return show_attributes
   
   def form_compare_template(self):
      
      template = """
      Please compare two conversations (conversation A, conversation B) and judge customer responses based on the following attributes:

      For each attribute described, identify whether the customer in Conversation A or Conversation B aligns more closely with the 'good response' or 'bad response'. 
      Your comparison response on each attributes should be a paragraph, and do not use new-line within the paragraph.

      {show_attributes}

      Based on these scenarios, evaluate and determine which customer (customer A or customer B) demonstrates more alignment with either 'good responses' or 'bad responses' in each attribute.

      Conversation A: {conversation_A}

      Conversation B: {conversation_B}
      """
      return template
   
   def form_compare_prompt(self, conversation_A, conversation_B):
      return self.form_compare_template().format(show_attributes=self.show_attributes, conversation_A=conversation_A, conversation_B=conversation_B)
   
   def save(self, filename: str):
      scenarios = [{"name": a.name, "scenario_desc": a.scenario_desc, "good_response": a.good_response, "bad_response": a.bad_response} for a in self.attributes]
      # store dict into json file
      with open(filename, 'w') as file:
          json.dump(scenarios, file, indent=4)
          
   @classmethod
   def load(cls, filename: str):
      # load dict from json file
      with open(filename, 'r') as file:
          scenarios = json.load(file)
      return cls.make(scenarios)
   
   def anno_compare_to_name(self, anno_compare_query: str):
      for attribute in self.attributes:
         if attribute.anno_compare_query == anno_compare_query:
            return attribute.name
      return None
   
   def auto_compare_to_name(self, auto_compare_query: str):
      for attribute in self.attributes:
         if attribute.auto_compare_query == auto_compare_query:
            return attribute.name
      return None
   
   def name_to_anno_compare(self, name: str):
      for attribute in self.attributes:
         if attribute.name == name:
            return attribute.anno_compare_query
      return None
   
   def name_to_auto_compare(self, name: str):
      for attribute in self.attributes:
         if attribute.name == name:
            return attribute.auto_compare_query
      return None
   
   def form_anno_compare_to_name_dict(self):
      return {attribute.anno_compare_query: attribute.name for attribute in self.attributes}
   
   def form_auto_compare_to_name_dict(self):
      return {attribute.auto_compare_query: attribute.name for attribute in self.attributes}
   
   def form_name_to_anno_compare_dict(self):
      return {attribute.name: attribute.anno_compare_query for attribute in self.attributes}
   
   def form_name_to_auto_compare_dict(self):
      return {attribute.name: attribute.auto_compare_query for attribute in self.attributes}



def parse_annotation_info(file_name):
    # This function extracts the annotation info from a given file name string
    # It assumes the file name format is 'annotation_<info>.csv'
    pattern = r'annotation_(.*).csv'
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        return None
    
get_comparative_rate = lambda x: ast.literal_eval(x)[0]
get_pred_rate = lambda x: x[0]

def get_preference_vector(anno_set: pd.DataFrame, mode='anno') -> np.array:
    if mode == 'anno':
      preference_vector = np.array(anno_set['preference'].apply(get_comparative_rate))
    elif mode == 'pred':
      preference_vector = np.array(anno_set['score'].apply(get_pred_rate))
    else:
      raise ValueError("mode should be either 'anno' or 'pred'")
    return preference_vector

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def pairwise_annotator_agreement_analysis(preference_vectors_array, annotators, draw=False):

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = cosine_similarity(preference_vectors_array)

    # Assuming 'cosine_similarity_matrix' is the similarity matrix to be visualized
    # and is already computed in the previous cells.

    # Create a heatmap to visualize the similarity matrix
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Pairwise Annotator Agreement Heatmap')
    plt.xlabel('Annotator')
    plt.ylabel('Annotator')

    plt.xticks(ticks=np.arange(len(annotators)), labels=annotators, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(annotators)), labels=annotators)


    # Highlight the outliers in the similarity matrix
    # Calculate the mean similarity score for each vector
    mean_similarity_scores = np.mean(cosine_similarity_matrix, axis=1)
    # Define a threshold for outliers, e.g., mean score +/- 2 standard deviations
    std_dev = np.std(mean_similarity_scores)
    threshold_upper = np.mean(mean_similarity_scores) + 2 * std_dev
    threshold_lower = np.mean(mean_similarity_scores) - 2 * std_dev
    outliers = np.where((mean_similarity_scores > threshold_upper) | 
                        (mean_similarity_scores < threshold_lower))

    # Annotate the outliers on the heatmap
    for outlier_index in outliers[0]:
        plt.axhline(outlier_index + 0.5, color='red', linestyle='--')
        plt.axvline(outlier_index + 0.5, color='red', linestyle='--')

    plt.show(fig)  
    if not draw:
        plt.close()
        # print("Trying to close the figure")
        plt.close(fig)
    
    return fig2img(fig)


def plot_embedding_deviation_1d(embeddings, annotations = None, title: str = "Annotator Preference & Deviation", draw = True):
    # Improve the aesthetics of the PCA scatter plot
    pca_result = PCA(n_components=1).fit_transform(embeddings)
    x = pca_result[:, 0]


def plot_embedding_deviations(embeddings, annotators = None, title:str="Annotator Preference & Deviation", draw=True):
    # Improve the aesthetics of the PCA scatter plot

    # Dimension reduction with PCA
    pca_result = PCA(n_components=2).fit_transform(embeddings)
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    # Compute the L2 distance between each pair of row vectors
    distances = cdist(pca_result, pca_result, 'euclidean')

    # Calculate average disagreement level
    # Calculate the mean difference between each row and the other rows
    mean_differences = distances.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and an axes object
    scatter = ax.scatter(x, y, c=mean_differences, cmap='YlOrRd', alpha=0.9, edgecolors='w', s=120)
    # scatter = ax.scatter(x, y, c=mean_differences, cmap='red', alpha=0.6, edgecolors='w', s=120)

    if annotators is not None:
        for i, label in enumerate(annotators):
            label += '|' + str(round(mean_differences[i], 2))
            ax.annotate(label, (x[i], y[i] - 0.053), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    fig.colorbar(scatter, ax=ax, label='Avgerage Deviation')
    ax.grid(True)  # Add grid for better readability
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    # Annotate the points with their index
    for i, txt in enumerate(range(len(embeddings))):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    
    # Calculate pair-wise distances between points
    distances = squareform(pdist(pca_result))
    
    # Draw lines and labels for pair-wise distances
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            ax.plot([x[i], x[j]], [y[i], y[j]], color='gray', linestyle='--', linewidth=0.5)
            
            # Calculate the midpoint coordinates for label placement
            mid_x = (x[i] + x[j]) / 2
            mid_y = (y[i] + y[j]) / 2
            
            # Format the distance label
            distance_label = f"{distances[i, j]:.2f}"
            
            # Place the distance label beside the line
            ax.annotate(distance_label, (mid_x, mid_y), textcoords="offset points", xytext=(0, 5), ha='center')
    
    if not draw:
        # print("Trying to close the figure")
        plt.close(fig)
    
    return fig2img(fig)


# Calculate row-vector variance, with dot-product as the distance metric, so the variance is a scaler
# Used to compute the variance among the pool of preference vectors
def calculate_variance(preference_vectors_array):
    n = preference_vectors_array.shape[0]  # Number of preference vectors
    
    # Calculate the squared norm of each preference vector
    squared_norms = np.sum(preference_vectors_array**2, axis=1)
    
    # Calculate the mean preference vector
    mean_vector = np.mean(preference_vectors_array, axis=0)
    
    # Calculate the squared norm of the mean vector
    mean_squared_norm = np.sum(mean_vector**2)
    
    # Calculate the variance
    variance = (1/n) * np.sum(squared_norms) - mean_squared_norm
    
    return variance


def get_preference_infos(annotation_file_info = 'data/annotation/annotation_*.csv'):
    anno_files = [name.split('/')[-1] for name in glob.glob(annotation_file_info)]

    # anno_file = parse_annotation_info(anno_files[0])
    preference_vectors = []
    annotators = []
    for i, anno_path in enumerate(anno_files):
        anno_file = parse_annotation_info(anno_path)
        annotated_dataset = POEDataset.load_annotated_dataset('data/annotation/', anno_file)
        preference_vector = get_preference_vector(annotated_dataset)
        preference_vectors.append(preference_vector)
        annotators.append(anno_file.split('_')[0])
    # Convert the list of preference vectors to a 2D array
    preference_vectors_array = np.array(preference_vectors)
    return preference_vectors_array, annotators

get_pred_rate = lambda x: x[0]

def get_preference_vector(anno_set: pd.DataFrame, mode='anno') -> np.array:
    if mode == 'anno':
      preference_vector = np.array(anno_set['preference'].apply(get_comparative_rate))
    elif mode == 'pred':
      preference_vector = np.array(anno_set['score'].apply(get_pred_rate))
    else:
      raise ValueError("mode should be either 'anno' or 'pred'")
    return preference_vector


# annotated_dataset
def get_unique_pairs(annotated_dataset):
    unique_pairs = set()
    for index, row in annotated_dataset.iterrows():
        pair = (row['hash_id_a'], row['hash_id_b'])
        unique_pairs.add(pair)
    return unique_pairs


def get_preference_anno_infos(annotation_path = 'data/fwd-customer-v3.3/annotation/*annotation_*.csv'):
    store_path = ('/').join(annotation_path.split('/')[:-1])+'/'
    anno_files = [name.split('/')[-1] for name in glob.glob(annotation_path)]

    unique_pairs = {}
    for i, anno_file in enumerate(anno_files):
        prefix, annotator_info = anno_file[:-4].split('annotation_')
        annotated_dataset = POEDataset.load_annotations(store_dir=store_path, annotator_info=annotator_info, prefix=prefix)
        if unique_pairs == {}:
            unique_pairs = get_unique_pairs(annotated_dataset)
        else:
            unique_pairs = unique_pairs.intersection(get_unique_pairs(annotated_dataset))
    hash_pairs = list(unique_pairs)

    preference_vectors = []
    annotators = []
    for i, anno_file in enumerate(anno_files):
        prefix, annotator_info = anno_file[:-4].split('annotation_')
        annotated_dataset = POEDataset.load_annotations(store_dir=store_path, annotator_info=annotator_info, prefix=prefix)

        # order hash pairs 
        ordered_pairs = []
        for pair in hash_pairs:
            pair_data = annotated_dataset.loc[(annotated_dataset['hash_id_a'] == pair[0]) & (annotated_dataset['hash_id_b'] == pair[1])]
            if not pair_data.empty:
                ordered_pairs.append(pair_data)
        filtered_dataset = pd.concat(ordered_pairs)

        # filtered_dataset = annotated_dataset[annotated_dataset.apply(lambda row: (row['hash_id_a'], row['hash_id_b']) in unique_pairs, axis=1)]
        preference_vector = get_preference_vector(filtered_dataset)
        preference_vectors.append(preference_vector)
        annotator = anno_file[:-4].split('annotation_')[1].split('_')[0]
        annotators.append(annotator)

    return preference_vectors, annotators, filtered_dataset, hash_pairs

# def get_preference_anno_infos(annotation_path = 'data/fwd-customer-v3.3/annotation/*annotation_*.csv'):
#     store_path = ('/').join(annotation_path.split('/')[:-1])+'/'
#     anno_files = [name.split('/')[-1] for name in glob.glob(annotation_path)]

#     unique_pairs = {}
#     for i, anno_file in enumerate(anno_files):
#         prefix, annotator_info = anno_file[:-4].split('annotation_')
#         annotated_dataset = POEDataset.load_annotations(store_dir=store_path, annotator_info=annotator_info, prefix=prefix)
#         if unique_pairs == {}:
#             unique_pairs = get_unique_pairs(annotated_dataset)
#         else:
#             unique_pairs = unique_pairs.intersection(get_unique_pairs(annotated_dataset))

#     preference_vectors = []
#     annotators = []
#     for i, anno_file in enumerate(anno_files):
#         prefix, annotator_info = anno_file[:-4].split('annotation_')
#         annotated_dataset = POEDataset.load_annotations(store_dir=store_path, annotator_info=annotator_info, prefix=prefix)
#         filtered_dataset = annotated_dataset[annotated_dataset.apply(lambda row: (row['hash_id_a'], row['hash_id_b']) in unique_pairs, axis=1)]
#         preference_vector = get_preference_vector(filtered_dataset)
#         preference_vectors.append(preference_vector)
#         annotator = anno_file[:-4].split('annotation_')[1].split('_')[0]
#         annotators.append(annotator)

#     return preference_vectors, annotators, filtered_dataset


import matplotlib.pyplot as plt
import seaborn

def case_study(idx, preference_vectors_array, filtered_dataset, conversation_df):
    hash_id_a, hash_id_b = filtered_dataset.iloc[idx][['hash_id_a', 'hash_id_b']]
    conversation_a = conversation_df[conversation_df['hash_id'] == hash_id_a]['conversation'].iloc[0].split("||")
    conversation_b = conversation_df[conversation_df['hash_id'] == hash_id_b]['conversation'].iloc[0].split("||")

    customer_a, customer_b = conversation_a[1], conversation_b[1]
    # Ensure values are [0., 0.5, 1.0] and counts are counted accordingly
    expected_values = np.array([1.0, 0.5, 0.0])
    values, counts = np.unique(preference_vectors_array[:, idx], return_counts=True)
    counts_dict = dict(zip(values, counts))
    counts_corrected = [counts_dict.get(value, 0) for value in expected_values]
    percentages = (np.array(counts_corrected) / np.sum(counts_corrected)) * 100

    # percentages
    keys = ["A", "Same", "B"]
    data, key = [], []
    for d, k in zip(percentages, keys):
        if d > 0:
            data.append(d)
            key.append(k)

    # declaring exploding pie 
    explode = [0.1] + [0] * (len(data)-1) 
    # define Seaborn color palette to use 
    palette_color = seaborn.color_palette('bright') 
    # plotting data on chart 
    plt.pie(data, labels=key, colors=palette_color, 
            explode=explode, autopct='%.0f%%') 
    plt.title("Which customer is more realistic? Annotator Votes")
    
    # displaying chart 
    plt.show() 

    print("Customer A: ", customer_a.split('Customer: ')[1])
    print("Customer B: ", customer_b.split('Customer: ')[1])


