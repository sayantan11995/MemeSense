
Training & Inference from  Cognitive Shift Vector for [MemeSense: An Adaptive In-Context Framework for Social Commonsense Driven Meme Moderation](https://arxiv.org/abs/2502.11246)


## Model Prepare
- donwload the `idefics-9b` and `idefics2-8b-base` model and put it into `model_weight` folder.
```bash
mkdir model_weight && cd model_weight
huggingface-cli download --resume-download HuggingFaceM4/idefics-9b --local-dir idefics-9b

huggingface-cli download --resume-download HuggingFaceM4/idefics2-8b-base --local-dir idefics2-8b-base
```


## Data Processing
# TBA


### Set The `.env` file
```bash
vi .env

##### Then set the following variables
# Please use Absolute Path
RESULT_DIR="/path/to/ICV-VQA/result"
VQAV2_PATH="/path/to/ICV-VQA/data/vqav2_formatted_data"
OKVQA_PATH="/path/to/ICV-VQA/data/vqav2_formatted_data" [Optional]
COCO_PATH="/path/to/ICV-VQA/data/memes"
MODEL_CPK_DIR="/path/to/ICV-VQA/model_weight"
```


## Run
```bash
# Run Idefics-v1-9B on vqav2 with 32 shot
python train.py run_name="vqav2_idefics_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.task.datasets.max_train_size=8000\
                data_cfg.task.datasets.few_shot_num=32\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=1e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=vqav2 \
                lmm=idefics-9B\
                trainer.precision="16-mixed" 


# Run Idefics-v1-9B on okvqa with 32 shot
python train.py run_name="okvqa_idefics_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.task.datasets.max_train_size=8000\
                data_cfg.task.datasets.few_shot_num=32\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=5e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=ok_vqa \
                lmm=idefics-9B\
                trainer.precision="16-mixed" 


# Run Idefics-v2-9B on vqav2 with 1 shot
python train.py run_name="vqav2_idefics2_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.task.datasets.max_train_size=8000\
                data_cfg.task.datasets.few_shot_num=1\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=1e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=vqav2 \
                lmm=idefics2-8B-base\
                trainer.precision="bf16-mixed" 


# Run Idefics-v2-9B on okvqa with 1 shot
python train.py run_name="okvqa_idefics2_icv"\
                icv_module.icv_encoder.use_sigmoid=False\
                icv_module.icv_encoder.alpha_init_value=0.1\
                data_cfg.task.datasets.max_train_size=8000\
                data_cfg.task.datasets.few_shot_num=1\
                data_cfg.bs=8\
                data_cfg.num_workers=10\
                trainer.accumulate_grad_batches=2\
                trainer.devices=4 \
                icv_module.icv_lr=5e-3 \
                icv_module.hard_loss_weight=0.5 \
                data_cfg/task/datasets=ok_vqa \
                lmm=idefics2-8B-base\
                trainer.precision="bf16-mixed" 
```

## Run Inference

```shell
python inference.py run_name="vqav2_idefics_icv" \
                data_cfg/task/datasets=vqav2\
                lmm=idefics-9B

python inference.py run_name="okvqa_idefics_icv" \
                data_cfg/task/datasets=ok_vqa\
                lmm=idefics-9B

python inference.py run_name="okvqa_idefics2_icv" \
                data_cfg/task/datasets=ok_vqa\
                lmm=idefics2-8B-base

python inference.py run_name="vqav2_idefics_icv" \
                data_cfg/task/datasets=vqav2\
                lmm=idefics2-8B-base
```

## Citing this work
```bibtex
@misc{adak2025memesenseadaptiveincontextframework,
      title={MemeSense: An Adaptive In-Context Framework for Social Commonsense Driven Meme Moderation}, 
      author={Sayantan Adak and Somnath Banerjee and Rajarshi Mandal and Avik Halder and Sayan Layek and Rima Hazra and Animesh Mukherjee},
      year={2025},
      eprint={2502.11246},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.11246}, 
}
```
