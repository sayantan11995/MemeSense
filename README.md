# MemeSense: An Adaptive In-Context Framework for Social Commonsense Driven Meme Moderation
[![Arxiv](https://img.shields.io/badge/arXiv-2502.11246-B21A1B)](https://arxiv.org/abs/2502.11246)

The implementation of [MemeSense: An Adaptive In-Context Framework for Social Commonsense Driven Meme Moderation](https://arxiv.org/abs/2502.11246)

<p align="center"><img src="./cognitive_shift_vectors/assets/memesense.png" alt="teaser" /></p>

## Install 
```bash
conda create -n memesense python=3.10

conda activate memesense
pip install -r requirments.txt

# For Openflamingo, please use transformers==4.28.1 [beta]

pip install transformers==4.48.1 [tested]

# Install the lmm_icl_interface
pip install git+https://github.com/ForJadeForest/lmm_icl_interface.git
# Install the baukit
pip install git+https://github.com/davidbau/baukit.git
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
