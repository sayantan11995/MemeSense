import json
import os
import random
import numpy as np


with open("training_intervention_images_relevant_mapping_idx_commonsense_anchored.json", "r") as content:
    commonsense_anchored = json.load(content)

with open("training_intervention_images_relevant_mapping_idx_image_anchored.json", "r") as content:
    image_anchored = json.load(content)

image_commonsense_anchored = {}
for idx, mapping_idx in commonsense_anchored.items():
    image_commonsense_anchored[idx] = mapping_idx[:2] + image_anchored[idx][:2]

    # print(mapping_idx)
    # print(image_commonsense_anchored[idx])

    random.shuffle(image_commonsense_anchored[idx])
    # print(image_commonsense_anchored[idx])

    # break


with open("training_intervention_images_relevant_mapping_idx_image_and_commonsense_anchored.json", "w") as content:
    json.dump(image_commonsense_anchored, content)