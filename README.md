# Mapping-free Automatic Verbalizer (MAV)

## Overview

This is the source code of Boosting Prompt-Based Self-Training With Mapping-Free Automatic Verbalizer for Multi-Class Classification (_EMNLP 2023 Submission_).


---

## Overall Structure

```
MAV
├── docker # A directory for building Docker environments
│   ├── create_container.sh
│   ├── create_image.sh
│   ├── Dockerfile
│   └── requirements.txt
├── tools # A directory for generating train data
│   ├── augmentation_trec.yaml
│   ├── check_dataset.ipynb
│   ├── generate_augmented_data.py
│   └── generate_data.py
├── data # Data directory (e.g. TREC dataset)
│   └── few-shot
│       └── trec
│           ├── 12-4-100
│           ├── 12-4-13
│           ├── 12-4-21
│           ├── 12-4-42
│           └── 12-4-87
│   └── original
│       └── trec
│           └── preprocess.py
├── src # Code directory
│   ├── augmentation
│   │   ├── aug_utils.py
│   │   ├── functional.py
│   │   ├── operations.py
│   │   └── policy.py
│   ├── dataset.py
│   ├── models.py
│   ├── model_utils.py
│   ├── trainer.py
│   └── utils.py
├── script # Script files to run training and analytics code
│   ├── analysis_trec.sh
│   └── run_trec.sh
├── run.py # Main code
├── calculate_result.py # Code for aggregating results of 5 seeds
├── analysis.py # Code for further analysis (SHAP, t-SNE)
└── exp_result # A directory for saving experimental results
    ├── mav-full_sup-trec # full supervised
    ├── mav-small_sup-trec # small supervised
    └── mav-ssl-singleaug_mask-trec # semi-supervised
```

### Detailed structure of `data` directory

A data directory is created for each seed and the directory name follows the format `k-mu-seed`. Where k is the number of labeled data per class and mu is the ratio between labeled and unlabeled data.

The data directory contains train, unlabeled, dev, test data in csv format and augmentation data in npy format.

Below is an example of the data directory structure for seed 13.


```
12-4-13
├── train.csv
├── dev.csv
├── test.csv
├── unlabeled.csv
├── unlabeled_backtranslation.npy
├── unlabeled_bertaug.npy
├── unlabeled_worddelete.npy
├── unlabeled_worddelete*wordswap.npy
└── unlabeled_wordswap.npy
```


### Detailed structure of `exp_result` directory

All output files from training, inference, and further analysis are stored in the exp_result directory.

```
mav-ssl-singleaug_mask-trec
├── seed13
├── seed21
│   ├── shap_trec_s21
│   │   ├── label00_shap_bar_131.png
│   │   ├── label01_shap_bar_53.png
│   │   ├── label02_shap_bar_8.png
│   │   ├── label03_shap_bar_58.png
│   │   ├── label04_shap_bar_75.png
│   │   └── label05_shap_bar_111.png
│   ├── tsne_trec_s21
│   │   └── tsne_mask_rep_test.png
│   ├── eval_results_trec.txt
│   ├── test_results_trec.txt
│   ├── data_args.bin
│   ├── model_args.bin
│   ├── training_args.bin
│   ├── pytorch_model.bin
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── config.json
│   └── vocab.json
├── seed42
├── seed87
├── seed100
└── total_results.txt
```

---
## Requirements

```
cd docker

bash create_image.sh
bash create_container.sh
```

Our experimental environment is built on Docker (`pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel` image). Detailed dependencies are described in `docker/requirements.txt`.

---

## How to Get Few-shot Data
### 0. Download & Preprocessing

The five datasets used in the experiment were downloaded from the sources below and preprocessed in the same way.
 
- Go Emotions, TREC50, Yahoo Answers : [HuggingFace datasets](https://huggingface.co/docs/datasets/index)

- TREC : [Official repository of LM-BFF](https://github.com/princeton-nlp/LM-BFF)

- AG News : [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download&select=train.csv) 

The source file for each data is stored in the path `data/original/{data_name}`.  
They are also preprocessed into the same form by running the file `data/original/{data_name}/preprocess.py`.

### 1. Sampling Few-shot Data

With the preprocessed data, sampling is performed to match `k/mu/seed`. 
This sampling is done via `tools/generate_gewshot_data.py`, setting the arguments as shown below.  
The result is stored in the path `data/few-shot/{data_name}/{k}-{mu}-{seed}`.


```bash 
python tools/generate_fewshot_data.py --k 16 --mu 4 --task trec --data_dir data/original --output_dir data/few-shot
```

### 2. Preprocessing for Augmentation
Store augmented data for augmentation experiments.
Augmentation is defined via `tools/augmentation_{data_name}.yaml` and the results are stored as npy files in the path `data/few-shot/{data_name}/{k}_{mu}_{seed}`. 
To perform the augmentation, refer to the bash code below.
The augmentation pool that can be saved in advance and the actual application key are as follows:

- Word Swap ([Wei et al., 2019](https://github.com/jasonwei20/eda_nlp)) : "wordswap"
- Word Delete ([Wei et al., 2019](https://github.com/jasonwei20/eda_nlp)) : "worddelete"
- CBERT ([Yi et al., 2021](https://arxiv.org/abs/2103.08933)): "bertaug"
- Back Translation ([Wie et al., 2020](https://github.com/google-research/uda)): "backtranslation"

```bash
python tools/generate_augmented_data.py --config_dir tools/augmentation_trec.yaml
```


---
## How to train

```
# Train, Inference
bash script/run_trec.sh

# Further analysis (SHAP, t-SNE)
bash script/analysis_trec.sh
```
