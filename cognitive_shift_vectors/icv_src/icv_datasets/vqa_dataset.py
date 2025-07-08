import json
import time
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from lmm_icl_interface import LMMPromptManager

from .load_ds_utils import load_okvqa_ds, load_vqav2_ds
# from . import incontext_example_selection

class VQADataset(Dataset):
    def __init__(
        self,
        name,
        root_dir,
        train_coco_dataset_root,
        val_coco_dataset_root,
        prompt_manager: LMMPromptManager,
        instruction="",
        few_shot_num=8,
        max_train_size=10000,
        split="train",
        val_ann_file=None,
        filter_ques_type=None,
        select_from_query=True,
        training_images_relevant_mapping_idx = None
    ):
        super().__init__()
        self.prompt_manager = prompt_manager
        if name == "vqav2":
            ds = load_vqav2_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
                val_ann_file=val_ann_file,
            )

            print(root_dir)
            print(train_coco_dataset_root)
            print(val_coco_dataset_root)
            print(split)
            print(val_ann_file)

            with open("/home/student/sayantan/safety_llm/cognitive_shift_vectors/icv_src/icv_datasets/training_intervention_images_relevant_mapping_idx_commonsense_anchored.json", "r") as content:
                self.training_images_relevant_mapping_idx = json.load(content)
            # mp.set_start_method('spawn')
            # p = mp.Process(target=incontext_example_selection.build_annoy_index, args=(ds,), daemon=True)
            # p.start()
            # p.join()
            # self.all_examples_embeddings = incontext_example_selection.get_embeddings(ds)
            # self.annoy_index = p
            # time.sleep(10)
        elif name == "okvqa":
            ds = load_okvqa_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
            )
        else:
            raise ValueError(f"Dataset {name} not supported")
        self.query_ds = ds
        if filter_ques_type:
            self.query_ds = ds.filter(
                lambda x: [i == filter_ques_type for i in x["gen_question_type"]],
                batched=True,
            )
            logger.info(
                f"After Filter Question Type Query dataset size: {len(self.query_ds)}"
            )

        if max_train_size > 0 and len(self.query_ds) > max_train_size:
            random_select_idx = np.random.choice(
                len(self.query_ds), size=max_train_size, replace=False
            )
            self.query_ds = self.query_ds.select(random_select_idx)
        if select_from_query:
            self.select_ds = self.query_ds
            logger.info("only select from query dataset")
        else:
            self.select_ds = ds
            logger.info("select from all dataset")
        self.few_shot_num = few_shot_num
        self.instruction = instruction
        logger.info(
            f"Query dataset size: {len(self.query_ds)}, Select dataset size: {len(self.select_ds)}"
        )

    def __len__(self):
        return len(self.query_ds)

    def __getitem__(self, index):
        """
        Retrieves an item from the VQA dataset at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "ice_prompt": A list of prompts for the in-context examples.
                - "query_prompt": A list of prompts for the query item.
                - "query_x": A list containing the query item's image and text.
        """
        query_item = self.query_ds[index]

        few_shot_index = np.random.choice(
            len(self.select_ds), size=self.few_shot_num
        ).tolist()

        # mp.set_start_method('spawn')
        few_shot_index = self.training_images_relevant_mapping_idx[str(index)]
        print("*"*100)
        print(index)
        print(few_shot_index)
        # print(relevant_indexes)
        # print(self.select_ds[0])
        select_global_idx = self.select_ds[few_shot_index]["idx"]
        # print(select_global_idx)



        # while query_item["idx"] in select_global_idx:
        #     # few_shot_index = np.random.choice(
        #     #     len(self.select_ds), size=self.few_shot_num
        #     # ).tolist()

        #     select_global_idx = self.select_ds[few_shot_index]["idx"]



        # print(select_global_idx)
        in_context_example = [self.select_ds[idx] for idx in few_shot_index]

        print(few_shot_index)
        in_context_text = [
            [
                ice["image"],
                self.prompt_manager.gen_ice_text_with_label(ice, add_sep_token=True),
            ]
            for ice in in_context_example
        ]
        prompt = []
        if self.instruction:
            prompt = [self.instruction]
        for ic_prompt in in_context_text:
            prompt.extend(ic_prompt)

        query_prompt = [
            query_item["image"],
            self.prompt_manager.gen_ice_text_with_label(
                query_item, add_sep_token=False
            ),
        ]

        query_x = [
            query_item["image"],
            self.prompt_manager.gen_query_text_without_label(query_item),
        ]
        # print("#"*50)
        # print({
        #     "ice_prompt": prompt,
        #     "query_prompt": query_prompt,
        #     "query_x": query_x,
        # })
        # print("#"*50)
        return {
            "ice_prompt": prompt,
            "query_prompt": query_prompt,
            "query_x": query_x,
        }
