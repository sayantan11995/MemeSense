import json
import os
import random
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from lmm_icl_interface import LMMPromptManager

from load_ds_utils import load_okvqa_ds, load_vqav2_ds

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from annoy import AnnoyIndex
from PIL import Image
import pandas as pd

prompt_template = "As an AI assistant tasked with social media content moderation, your role is to prevent harmful, offensive, hateful, vulgar, misogynistic, or unethical content from being posted on public platforms.\n\nYour Task: A toxic meme has the description below along with few commonsense parameters which assess whether the meme has the potential to be perceived as vulgar, harmful, or unethical. Write an intervention for the this toxic meme to discourage user posting such memes based on provided knwoledge.\n\n"

model_name = 'clip-ViT-B-32'
# model_name='all-MiniLM-L6-v2'

def build_annoy_index(dataset, model_name=model_name, index_file='annoy_index.ann', image_based=True, dimension=384):
    """
    Build an Annoy index for the dataset descriptions.
    
    Args:
        dataset (Dataset): Hugging Face dataset containing the descriptions.
        model_name (str): Pretrained Sentence Transformer model to use.
        index_file (str): File path to save the Annoy index.
        dimension (int): Dimension of embeddings (default is 384 for MiniLM).

    Returns:
        AnnoyIndex: The built Annoy index.
    """
    # Load the model
    model = SentenceTransformer(model_name).to("cuda")
    
    if image_based:
        # dataset["image_id"]
        train_image_directory = "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/ours/mscoco2014/train2014/"
        img_names = [os.path.join(train_image_directory, f"{img_id}.jpg") for img_id in dataset["image_id"]]
        print(img_names)
        embeddings = model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
        dimension = 512
    else:
        # Encode dataset descriptions
        descriptions = dataset["question"]
        descriptions = [d.replace(prompt_template, "") for d in descriptions]
        embeddings = model.encode(descriptions)
    
    # Initialize Annoy index
    annoy_index = AnnoyIndex(dimension, 'angular')
    
    # Add embeddings to the index
    for idx, embedding in enumerate(embeddings):
        annoy_index.add_item(idx, embedding)
    
    # Build the index
    annoy_index.build(10)  # Use 10 trees (adjust based on speed/accuracy trade-off)
    
    # Save the index
    annoy_index.save(index_file)
    
    return annoy_index

def load_annoy_index(index_file, dimension=384):
    """
    Load a saved Annoy index.
    
    Args:
        index_file (str): File path of the saved Annoy index.
        dimension (int): Dimension of the embeddings.

    Returns:
        AnnoyIndex: The loaded Annoy index.
    """
    annoy_index = AnnoyIndex(dimension, 'angular')
    annoy_index.load(index_file)
    return annoy_index

def get_most_relevant_indexes_annoy(annoy_index, input_text=None, input_image_path=None, model_name=model_name, n=5):
    """
    Retrieve the most relevant n indexes using the Annoy index.

    Args:
        annoy_index (AnnoyIndex): Pre-built Annoy index.
        input_text (str): Text to compare against dataset descriptions.
        model_name (str): Pretrained Sentence Transformer model to use.
        n (int): Number of most similar descriptions to retrieve.

    Returns:
        List[int]: Indexes of the most relevant descriptions.
    """
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Encode the input text
    # input_embedding = model.encode(input_text)

    image = Image.open(input_image_path)
    input_embedding = model.encode(image)
    
    # Retrieve top n indexes
    most_relevant_indexes = annoy_index.get_nns_by_vector(input_embedding, n)
    
    return most_relevant_indexes

# Example Usage:
train_image_path = "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/coco/mscoco2014/train2014"
train_ds = load_vqav2_ds(
                "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/memeguard_train_annotations",
                train_image_path,
                "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/coco/mscoco2014/val2014",
                split="train",
                val_ann_file="v2_mscoco_val2014_annotations_subdata.json",
            )

validation_image_path="/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/coco/mscoco2014/val2014"
validation_ds = load_vqav2_ds(
                "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/memeguard_validation",
                train_image_path,
                validation_image_path,
                split="validation",
                val_ann_file="v2_mscoco_val2014_annotations_subdata.json",
            )

# print(train_ds[0])
# validation_ds = load_vqav2_ds(
#                 "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/fhm",
#                 "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/facebook-hateful-meme-dataset/data/mscoco2014/train2014",
#                 "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/facebook-hateful-meme-dataset/data/mscoco2014/val2014",
#                 split="validation",
#                 val_ann_file="v2_mscoco_val2014_annotations_subdata.json",
#             )

# Build the Annoy index
# annoy_index = build_annoy_index(train_ds)

train_mapping_id = {}

# ## Getting relevant index based on image commonsense embedding similarity
# for idx in validation_ds["idx"]:
#     # print(idx)
#     # query = validation_ds[idx]["question"]

#     print(validation_ds[idx])
#     query = os.path.join(validation_image_path, f"{validation_ds[idx]['image_id']}")

#     # print(train_ds[idx])
#     print(query)
#     relevant_indexes = get_most_relevant_indexes_annoy(annoy_index, input_image_path=query)
#     # print(idx, relevant_indexes)
    
#     if idx in relevant_indexes:
#         relevant_indexes.remove(idx)

#     relevant_indexes = relevant_indexes[:4]

#     # train_mapping_id[idx] = [relevant_indexes[1]]
#     train_mapping_id[idx] = relevant_indexes
#     print(idx, relevant_indexes)
#     # break


# # with open("training_intervention_images_relevant_mapping_idx_image_anchored.json", "w") as content:
# #     json.dump(train_mapping_id, content)

# with open("validation_with_text_intervention_images_relevant_mapping_idx_image_anchored.json", "w") as content:
#     json.dump(train_mapping_id, content)


# import sys
# sys.exit()

## Get a mapping of commonsense_category to ds_id of the images
# Create a mapping of image_name to its index
image_name_to_index = {train_ds[i]['image_id']: i for i in range(len(train_ds))}

# Given dictionary of features
with open("commonsense_to_5_images_new.json", "r") as content:
    commonsense_to_5_images_dict = json.load(content)
# Convert image names to dataset indices
broad_commonsense_ds_index_dict = {
    feature.lower(): [image_name_to_index[img] for img in images if img in image_name_to_index]
    for feature, images in commonsense_to_5_images_dict.items()
}

# print(feature_index_dict)

def get_related_image_indices(original_image_index, features, broad_commonsense_ds_index_dict, max_length=4):
    selected_indices = set()  # To store unique indices
    feature_queues = {feature: [idx for idx in broad_commonsense_ds_index_dict.get(feature, []) if idx != original_image_index] 
                      for feature in features}
    
    # Shuffle each feature queue to ensure randomness
    for queue in feature_queues.values():
        random.shuffle(queue)
    
    # Iteratively add indices while following constraints
    while len(selected_indices) < max_length:
        added = False
        for feature in features:
            if feature_queues[feature]:  # If there are remaining indices
                idx = feature_queues[feature].pop(0)  # Take the first available index
                if idx not in selected_indices:  # Ensure uniqueness
                    selected_indices.add(idx)
                    added = True
                if len(selected_indices) == max_length:
                    break  # Stop if max limit reached
        
        if not added:  # If no new index was added in this iteration, break
            break
    
    return list(selected_indices)

## Getting relevant index based on broad commonsense category
with open("memeguard_validation_broad_commonsense_mapping_new.json", "r") as content:
    broad_commonsense_mapping = json.load(content)
for idx in validation_ds["idx"]:

    """
        img_id
        commonsense tags from img_id

        for tag in tags:

            1st image wrt tag
            get the ds_id of the image in the train_ds: requires a mapping
            if ds_id already in train_mapping_id:
                pass
            if len(train_mapping_id) >= 3:
                break 

    """
    image_id = validation_ds[idx]["image_id"]
    # broad_commonsense_tags = broad_commonsense_mapping[f"{image_id}.jpeg"] ## List of tags for the input image
    broad_commonsense_tags = broad_commonsense_mapping[image_id]

    train_mapping_id[idx] = get_related_image_indices(idx, broad_commonsense_tags, broad_commonsense_ds_index_dict)
    print(broad_commonsense_ds_index_dict)

    # i = 0
    # for tag in broad_commonsense_tags*5:
        
    #     print(broad_commonsense_ds_index_dict[tag])
    #     if i < len(broad_commonsense_ds_index_dict[tag]):
    #         selected_idx = broad_commonsense_ds_index_dict[tag][i]

    #         if selected_idx!=idx:
    #             if idx not in train_mapping_id.keys():
    #                 train_mapping_id[idx] = [selected_idx]
    #             else:
    #                 if (selected_idx not in train_mapping_id[idx]):
    #                     train_mapping_id[idx].append(selected_idx)

    #             print(train_mapping_id)

    #             if len(train_mapping_id[idx]) >= 4:
    #                 break
    #     print("*"*20)
    #     i+=1


    # if idx == 2:
    #     break


print(train_mapping_id)

with open("memeguard_validation_intervention_images_relevant_mapping_idx_commonsense_anchored.json", "w") as content:
    json.dump(train_mapping_id, content)

# with open("validation_intervention_images_relevant_mapping_idx.json", "w") as content:
#     json.dump(train_mapping_id, content)

# Retrieve most relevant indexes
# input_text = "This is a sample description to compare."
# n = 5
# relevant_indexes = get_most_relevant_indexes_annoy(annoy_index, input_text)
# print("Most relevant indexes:", relevant_indexes)


# def get_embeddings(dataset, model_name='all-MiniLM-L6-v2'):
#     """
#     Select the most relevant n indexes from the dataset based on similarity to the input text.

#     Args:
#         dataset (Dataset): Hugging Face dataset containing the descriptions.
#         text_column (str): Column in the dataset containing the descriptions.
#         input_text (str): Text to compare against dataset descriptions.
#         n (int): Number of most similar descriptions to retrieve.
#         model_name (str): Pretrained Sentence Transformer model to use.

#     Returns:
#         List[int]: Indexes of the most relevant descriptions.
#     """
#     # Load the model
#     model = SentenceTransformer(model_name)
    
#     # Encode the dataset descriptions
#     descriptions = dataset["question"]
#     description_embeddings = model.encode(descriptions, convert_to_tensor=True)

#     return description_embeddings
    
#     # # Encode the input text
#     # input_embedding = model.encode(input_text, convert_to_tensor=True)
    
#     # # Compute cosine similarities
#     # cosine_similarities = (description_embeddings @ input_embedding.T).cpu().numpy()
    
#     # # Get the top n indexes
#     # most_relevant_indexes = np.argsort(-cosine_similarities, axis=0)[:n]
    
#     # return most_relevant_indexes.tolist()

# def get_most_relevant_indexes(all_embeddings, input_text, model_name='all-MiniLM-L6-v2', n=5):
#     """
#     Retrieve the most relevant n indexes using the Annoy index.

#     Args:
#         annoy_index (AnnoyIndex): Pre-built Annoy index.
#         input_text (str): Text to compare against dataset descriptions.
#         model_name (str): Pretrained Sentence Transformer model to use.
#         n (int): Number of most similar descriptions to retrieve.

#     Returns:
#         List[int]: Indexes of the most relevant descriptions.
#     """
#     # Load the model
#     model = SentenceTransformer(model_name)
    
#     # Encode the input text
#     input_embedding = model.encode(input_text, convert_to_tensor=True)
    
#     # Compute cosine similarities
#     cosine_similarities = (all_embeddings @ input_embedding.T).cpu().numpy()
    
#     # Get the top n indexes
#     most_relevant_indexes = np.argsort(-cosine_similarities, axis=0)[:n]

#     print(most_relevant_indexes)

#     return most_relevant_indexes.tolist()
    
