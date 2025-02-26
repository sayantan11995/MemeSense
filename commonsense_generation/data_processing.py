from datasets import Dataset, DatasetDict, load_dataset
from datasets import Image as image_ecoder 
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import os 
import shutil

# data_path = "../final_images/"

# Load your CSV file
# csv_file_path = "../image_data/images_with_annotations/annotations.csv"  # Update with your CSV file path
csv_file_path = "../images_with_intervention.csv"
image_folder = Path("/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/coco/mscoco2014/final_images/final_images")  # Update with your images folder path

train_csv_file_path = "/home/student/sayantan/safety_llm/train_memes_with_commonsense_new.csv"
val_csv_file_path = "/home/student/sayantan/safety_llm/validation_memes_with_commonsense.csv"


# dataset  = load_dataset("imagefolder", data_dir="/home/student/heisenberg/safety_llm/image_data/hg_datasets_formatted_data/" )


# annotated_file = pd.read_csv(csv_file_path)

train_data = pd.read_csv(train_csv_file_path)
val_data = pd.read_csv(val_csv_file_path)

######### Randomly copy 80% images to train and 20% to validation ###########
# images = os.listdir(data_path)
# images = list(annotated_file["images"])
# print(images)

# num_to_select = int(len(images) * 0.8)
# selected_images = random.sample(images, num_to_select)
# remaining_images = list(set(images) - set(selected_images))

commonsense_labelled_data_path = "../commonsense_labelled_data_new/"

try:
    os.mkdir(commonsense_labelled_data_path)
except:
    pass

# Copy selected images to the 80% directory
for image in list(train_data["image_id"]):
    source_path = os.path.join(image_folder, f"{image}.jpg")
    target_path = os.path.join(commonsense_labelled_data_path, "train", f"{image}.jpg")
    shutil.copy(source_path, target_path)
    print(f"Copied to 80% directory: {image}")

# Copy remaining images to the 20% directory
for image in list(val_data["image_id"]):
    source_path = os.path.join(image_folder, f"{image}.jpg")
    target_path = os.path.join(commonsense_labelled_data_path, "validation", f"{image}.jpg")
    shutil.copy(source_path, target_path)
    print(f"Copied to 20% directory: {image}")


########### Save metadata.csv wih file_name and text ###############
# file_names = annotated_file["images"]
file_names = [f"{image_id}.jpg" for image_id in train_data['image_id']]
text = train_data["description_with_commonsense_parameters"]

train_metadata = pd.DataFrame({
    "file_name": file_names,
    "text": text
})
train_metadata.to_csv(os.path.join(commonsense_labelled_data_path, "train", "metadata.csv"), index=False)

#######################################################################

file_names = [f"{image_id}.jpg" for image_id in val_data['image_id']]
text = val_data["description_with_commonsense_parameters"]

val_metadata = pd.DataFrame({
    "file_name": file_names,
    "text": text
})

val_metadata.to_csv(os.path.join(commonsense_labelled_data_path, "validation", "metadata.csv"), index=False)

# print(dataset["train"][0])
# Read the CSV
# df = pd.read_csv(csv_file_path)


# feature = image_ecoder()

# # Define a function to load images
# def load_image(image_id):
#     image = Image.open(image_folder / image_id)
#     return  feature.encode_example(image)

# # Add the image loading to the DataFrame
# df['images'] = df['images'].apply(load_image)

# # Define the split sizes
# train_size = 0.8
# val_size = 0.2
# # test_size = 0.1

# # Split the data
# df_train, df_val = train_test_split(df, train_size=train_size, shuffle=True, random_state=42)
# # df_val, df_test = train_test_split(df_temp, test_size=test_size/(test_size + val_size), shuffle=True, random_state=42)

# # Convert the DataFrames to Hugging Face datasets
# train_dataset = Dataset.from_pandas(df_train)
# val_dataset = Dataset.from_pandas(df_val)
# # test_dataset = Dataset.from_pandas(df_test)

# # Create a DatasetDict with the splits
# dataset_dict = DatasetDict({
#     'train': train_dataset,
#     'validation': val_dataset,
#     # 'test': test_dataset,
# })

# # Save the dataset (optional)
# dataset_dict.save_to_disk("hg_image_dataset")

# # Print the dataset to verify
# print(dataset_dict)

# print(dataset_dict['train'])
# print(dataset_dict['validation'])