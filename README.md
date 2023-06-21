# Mapping-free Automatic Verbalizer (MAV)
(영어로 번역 예정)

## Overview

Boosting Prompt-Based Self-Training With Mapping-Free Automatic Verbalizer for Multi-Class Classification (_TACL under review_) 논문의 소스코드입니다.


---

## 전체 구조도

```
MAV
├── docker # 도커 환경 구축을 위한 디렉토리
│   ├── create_container.sh
│   ├── create_image.sh
│   ├── Dockerfile
│   └── requirements.txt
├── tools # 학습용 데이터 생성 디렉토리
│   ├── augmentation_trec.yaml
│   ├── check_dataset.ipynb
│   ├── generate_augmented_data.py
│   └── generate_data.py
├── data # 데이터 디렉토리 (ex. trec 데이터셋)
│   └── few-shot
│       └── trec
│           ├── 12-4-100
│           ├── 12-4-13
│           ├── 12-4-21
│           ├── 12-4-42
│           └── 12-4-87
├── src # 코드 디렉토리
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
├── script # 학습 및 분석 코드 실행용 스크립트 파일
│   ├── analysis_trec.sh
│   └── run_trec.sh
├── run.py # 메인 학습용 코드
├── calculate_result.py # 5개 시드 결과 종합용 코드
├── analysis.py # 분석용(shap, tsne) 코드
└── exp_result # 실험 결과 디렉토리
    ├── mav-full_sup-trec # full supervised
    ├── mav-small_sup-trec # small supervised
    └── mav-ssl-singleaug_mask-trec # semi-supervised
```

### `data` 디렉토리 세부 구조도

seed 별로 데이터 디렉토리가 생성되며 디렉토리명 형식은 `k-mu-seed`를 따릅니다. 이때, k는 class 별 labeled data 개수를, mu는 labeled와 unlabeled 데이터 간 비율을 의미합니다.

데이터 디렉토리는 csv 형식의 train, unlabeled, dev, test 데이터와 npy 형식의 augmentation 데이터를 포함합니다.

아래는 13번 seed의 데이터 디렉토리 구조도 예시입니다.

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


### `exp_result` 디렉토리 세부 구조도

학습과 추론, 이후 분석까지 모든 결과물은 exp_result 디렉토리에 저장됩니다.

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

## How to train

```
# 학습 및 추론
bash script/run_trec.sh

# 결과 분석 (SHAP, Tsne)
bash script/analysis_trec.sh
```
