
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from PIL import Image


MAX_LENGTH = 256
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
REPO_ID = "YOUR-HUB-REPO-TO-PUSH"

# CHECKPOINT = "/home/student/heisenberg/safety_llm/finetuning/lightning_logs/version_13/checkpoints/epoch=3-step=16.ckpt"
CHECKPOINT = "/home/student/sayantan/safety_llm/finetuning/lightning_logs/version_12/checkpoints/epoch=3-step=196.ckpt"
SAVE_PATH = "saved_model_new"


processor = AutoProcessor.from_pretrained(MODEL_ID)


## For quantized model
# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
# # Load the base model with adapters on top
model = LlavaNextForConditionalGeneration.from_pretrained(
    SAVE_PATH,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)


import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np


class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    # def forward(self, x):
    #     return self.model(x)

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    # def train_dataloader(self):
    #     return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    # def val_dataloader(self):
    #     return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

# """Let's instantiate it (based on a config dictionary which defines all hyperparameters for training).

# The batch size was determined based on PyTorch Lightning's [auto batch size finder](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#batch-size-finder) feature. It tries to find the biggest batch size for your given hardware.

# Do note that one can play around with the hyperparameters, I just use good defaults here: 10 epochs, a learning rate of 1e-4 which I found in the original Idefics2 notebook (linked at the top of this notebook), use mixed precision for training (more memory friendly). One could extend this with things like gradient accumulation and gradient checkpointing.

# I recommend [this guide](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one) which goes over all tips and tricks regarding maximizing fine-tuning performance on consumer hardware.
# """

# config = {"max_epochs": 10,
#           # "val_check_interval": 0.2, # how many times we want to validate during an epoch
#           "check_val_every_n_epoch": 1,
#           "gradient_clip_val": 1.0,
#           "accumulate_grad_batches": 8,
#           "lr": 1e-4,
#           "batch_size": 1,
#           # "seed":2022,
#           "num_nodes": 1,
#           "warmup_steps": 5,
#           "result_path": "./result",
#           "verbose": True,
# }

# model_module = LlavaModelPLModule(config, processor, model)

# pl_model = LlavaModelPLModule.load_from_checkpoint(CHECKPOINT, hparams_file="/home/student/heisenberg/safety_llm/finetuning/lightning_logs/version_12/hparams.yaml",
#                                                 config=config, processor=processor, model=model, strict=False )
# pl_model.eval()

"""Now we're ready to perform inference. We'll take a receipt image of the test set here."""

# test_example = dataset["test"][0]
# test_image = test_example["image"]
# test_image = Image.open("/home/student/heisenberg/safety_llm/finetuning/test_images/3.jpg")
test_image = Image.open("/home/student/sayantan/safety_llm/commonsense_labelled_data/validation/Dick-Pic-Meme-on-Girl.jpg")
print(test_image)

"""Next we need to prepare the image for the model, along with the text prompt we used during training. We need to apply the chat template to make sure the format is respected.

Notice that this is exactly the same as what we did for the evaluation data collate function.
"""

# prepare image and prompt for the model
# TODO this can be replaced by apply_chat_template when the processor supports this
# prompt = f"[INST] <image>\nExtract JSON [\INST]"
prompt = "[INST] <image>\nDescribe the meme and assess whether the meme has the potential to be perceived as vulgar, harmful, or unethical using specific commonsense parameters. [\INST]"

inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to("cuda")
for k,v in inputs.items():
    print(k,v.shape)

"""Next we let the model autoregressively generate tokens using the [generate()](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) method, which is recommended for use at inference time. This method feeds each predicted token back into the model as conditioning for each next time step.

Do note that there are various ways of decoding text, here we use greedy decoding which is the default. There are various fancier methods such as beam search and top-k sampling. Refer to [this amazing blog post](https://huggingface.co/blog/how-to-generate) for all details.
"""

# Generate token IDs
generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

# Decode back into text
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)


##### Task specific
import pandas as pd
import os
from tqdm import tqdm


image_data = pd.read_csv("/home/student/sayantan/safety_llm/tasks/without_text_memes/validation_memes_with_commonsense.csv")
# ICMM_data = pd.read_csv("/home/student/sayantan/safety_llm/tasks/generate_intervention/ICMM.csv")
# fhm_data = pd.read_csv("/home/student/sayantan/safety_llm/tasks/fhm/fhm_dev_seen.csv")

images = list(image_data["image_id"])
# images = list(ICMM_data["Img_Name"])
# images = list(fhm_data["img"])

# image_path = "/home/student/sayantan/safety_llm/tasks/generate_intervention/bully_data"
image_path = "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/ours/mscoco2014/val2014/"
# image_path = "/home/student/sayantan/safety_llm/LIVE-Learnable-In-Context-Vector/data/facebook-hateful-meme-dataset/data/"

valid_images = []
commonsense_parameters = []
for image in tqdm(images):
    try:
        test_image = Image.open(os.path.join(image_path, f"{image}.jpg"))
        inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to("cuda")
    
        # Generate token IDs
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

        # Decode back into text
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        commonsense_parameters.append(generated_texts)
        valid_images.append(image)

    except Exception as e:
        print(e)


df = pd.DataFrame({
    "images": valid_images,
    "commonsense_parameters": commonsense_parameters,
})

# df = pd.DataFrame({
#     "images": valid_images,
#     "commonsense_parameters": commonsense_parameters,
#     "gold_pc": list(fhm_data['gold_pc'])
# })


df.to_csv("/home/student/sayantan/safety_llm/tasks/without_text_memes/generated_commonsense_new.csv", index=False)
# df.to_csv("/home/student/sayantan/safety_llm/tasks/fhm/fhm_dev_seen_generated_commonsense.csv", index=False)


# """Based on the Donut model, we could write a `token2json` method which converts the generated token sequence into parsible JSON."""

# import re

# # let's turn that into JSON
# def token2json(tokens, is_inner_value=False, added_vocab=None):
#         """
#         Convert a (generated) token sequence into an ordered JSON format.
#         """
#         if added_vocab is None:
#             added_vocab = processor.tokenizer.get_added_vocab()

#         output = {}

#         while tokens:
#             start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
#             if start_token is None:
#                 break
#             key = start_token.group(1)
#             key_escaped = re.escape(key)

#             end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
#             start_token = start_token.group()
#             if end_token is None:
#                 tokens = tokens.replace(start_token, "")
#             else:
#                 end_token = end_token.group()
#                 start_token_escaped = re.escape(start_token)
#                 end_token_escaped = re.escape(end_token)
#                 content = re.search(
#                     f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
#                 )
#                 if content is not None:
#                     content = content.group(1).strip()
#                     if r"<s_" in content and r"</s_" in content:  # non-leaf node
#                         value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
#                         if value:
#                             if len(value) == 1:
#                                 value = value[0]
#                             output[key] = value
#                     else:  # leaf nodes
#                         output[key] = []
#                         for leaf in content.split(r"<sep/>"):
#                             leaf = leaf.strip()
#                             if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
#                                 leaf = leaf[1:-2]  # for categorical special tokens
#                             output[key].append(leaf)
#                         if len(output[key]) == 1:
#                             output[key] = output[key][0]

#                 tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
#                 if tokens[:6] == r"<sep/>":  # non-leaf nodes
#                     return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

#         if len(output):
#             return [output] if is_inner_value else output
#         else:
#             return [] if is_inner_value else {"text_sequence": tokens}

# """Let's print the final JSON!"""

# generated_json = token2json(generated_texts[0])
# print(generated_json)

# for key, value in generated_json.items():
#     print(key, value)

