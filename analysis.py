# Analysis
#   1. SHAP value for each class
#   2. TSNE visualization of [MASK] representation

import shap
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import default_data_collator
from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments

from src.dataset import FewShotDataset
from src.models import RobertaForPromptFinetuning
from src.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping
from src.trainer import Trainer
from src.utils import set_seed, draw_tsne_plot

from datetime import datetime


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    few_shot_type: str = field(
        default='prompt',
        metadata={"help": "prompt-based fine-tuning. 'prompt'"}
    )

    mav_hidden_dim: int = field(
        default=256,
        metadata={"help": "Dimension of vocab extractor"}
    )

    return_mask_rep: bool = field(
        default=False,
        metadata={"help": "Whether return mask representation or not"}
    )

    soft_verb: bool = field(
        default=False,
        metadata={"help": "Whether train soft verbalizer(baseline) or not"}
    )

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=1,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    # --- For prompting ---
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )
    # ---

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # --- For max length ---
    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )
    # ---

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )
    # for multi label words
    dedup: bool = field(
        default=False,
        metadata={"help": "deduplication for not overlapping among label words"}
    )
    top_k: int = field(
        default=16,
        metadata={"help": "number of label words for each label"}
    )

    inf_data_type: str = field(
        default=None,
        metadata={"help": "'small_lb', 'ulb', 'full_lb', 'eval', 'test'"}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # Unify total train epoch, eval step
    eval_nums: int = field(
        default=20,
        metadata={"help": "total nums of evaluation during training"}
    )

    use_mlm_loss: bool = field(
        default=False
    )

    # --- single aug ---
    single_aug: bool = field(
        default=False,
        metadata={"help": "Whether to use single aug"}
    )

    single_aug_type: str = field(
        default=None,
        metadata={"help": "name of single augmentation"}
    )

    aug_mask_ratio: float = field(
        default=0.15,
        metadata={"help": "random masking ratio for augmentation"}
    )
    # ---

    # random aug
    randaug: bool = field(
        default=False,
        metadata={"help": "Whether to use randaug"}
    )

    # --- autoaug (DND) ---
    autoaug: bool = field(
        default=False,
        metadata={"help": "Whether to use autoaug"}
    )

    # --- For baseline(supervised) ---
    base_mode: str = field(
        default=None,
        metadata={"help": "'sup'(supervised) or 'ssl'(semi-supervised)"}
    )

    train_type: str = field(
        default=None,
        metadata={"help": "'full_train'(full) or 'train'(small)"}
    )
    # ---

    # --- For analysis ---
    inf_batch_size: int = field(
        default=32
    )

    analysis_type: str = field(
        default=None,
        metadata={"help": "'shap_value' of 'tsne'"}
    )

    zero_shot: bool = field(
        default=False,
        metadata={"help": "Whether to use pretrained model or finetuned model"}
    )
    # ---


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True
    training_args.k = data_args.num_k

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    training_args.output_dir = f"{training_args.output_dir}/seed{training_args.seed}"

    # Create Directory
    if training_args.analysis_type == "shap_value":
        shap_dir = f"{training_args.output_dir}/shap_{data_args.task_name}_s{training_args.seed}"
        if not os.path.exists(shap_dir):
            os.makedirs(shap_dir)
    elif training_args.analysis_type == "tsne":
        tsne_dir = f"{training_args.output_dir}/tsne_{data_args.task_name}_s{training_args.seed}"
        if not os.path.exists(tsne_dir):
            os.makedirs(tsne_dir)

    if data_args.task_name == "trec":
        label_id2str = {0:'Expression',1:'Entity',2:'Description',3:'Human',4:'Location',5:'Number'}
    elif data_args.task_name == "trec50":
        label_id2str = {0:'shortened',1:'animal',2:'creation',3:'medical',4:'food',5:'other',6:'sport',7:'equal',8:'definition',9:'description',10:'manner',11:'reason',12:'group',13:'individual',14:'city',15:'country',16:'location',17:'state',18:'count',19:'date',20:'money',21:'period'}
    elif data_args.task_name == "goemotions":
        label_id2str = {0:'optimism',1:'neutral',2:'amusement',3:'curiosity',4:'surprise',5:'confusion',6:'nervous',7:'disgust',8:'joy',9:'anger',10:'gratitude',11:'sadness',12:'disappointment',13:'desire',14:'embarrassment',15:'remorse',16:'realization',17:'excitement',18:'admiration',19:'disapproval',20:'caring',21:'fear',22:'approval',23:'love',24:'annoyance',25:'relief'}
    elif data_args.task_name == "yahoo_answers":
        label_id2str = {0:'Society',1:'Science',2:'Health',3:'Education',4:'Computer',5:'Sports',6:'Business',7:'Entertainment',8:'Relationship',9:'Politics'}
    elif data_args.task_name == "ag_news":
        label_id2str = {0:'World',1:'Sports',2:'Business',3:'Tech'}

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    if training_args.zero_shot:
        model_pth = 'roberta-base'
    else:
        model_pth = model_args.model_name_or_path

    # Create config
    config = AutoConfig.from_pretrained(
        model_pth,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
    )

    config.mav_hidden_dim = model_args.mav_hidden_dim
    config.return_mask_rep = model_args.return_mask_rep
    config.soft_verb = model_args.soft_verb

    model_fn = RobertaForPromptFinetuning

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_pth,
    )

    # Get our special datasets.
    print(f">>>>> [Dataset Info]")
    
    if data_args.inf_data_type == "small_lb":
        ### small labeled
        inf_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="train")
        )
    elif data_args.inf_data_type == "ulb":
        ### unlabeled
        inf_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="unlabeled")
        )
    elif data_args.inf_data_type == "full_lb":
        ### full
        inf_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="full_train", base_mode="sup")
        ) # (base_mode == "sup") and (mode=="full_train")
    elif data_args.inf_data_type == "valid":
        inf_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="dev")
        )
    elif data_args.inf_data_type == "test":
        inf_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="test")
        )

    print(f">>>>> {data_args.inf_data_type}: {len(inf_dataset)}")


    if training_args.analysis_type == "shap_value":
        train_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="train", base_mode="sup")
        )
        unlabeled_dataset = None
        eval_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="dev")
        )
        test_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="test")
        )
        print(f"eval dataset size: {len(eval_dataset)}, test dataset size: {len(test_dataset)}")

        # Build metric
        def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                # Note: the eval dataloader is sequential, so the examples are in order.
                predictions = p.predictions[0] #(n_data, n_class) # ex(trec test): (500,6)
                num_logits = predictions.shape[-1] #n_class
                logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits]) # (num_sample(1), n_data, n_class)
                logits = logits.mean(axis=0) # (n_data, n_class)
                
                if num_logits == 1:
                    preds = np.squeeze(logits)
                else:
                    preds = np.argmax(logits, axis=1) # (n_data,)

                # Just for sanity, assert label ids are the same.
                label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
                label_ids_avg = label_ids.mean(axis=0)
                label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
                assert (label_ids_avg - label_ids[0]).mean() < 1e-2
                label_ids = label_ids[0] # (n_data,)

                return compute_metrics_mapping[task_name](task_name, preds, label_ids)

            return compute_metrics_fn
    
    model = model_fn.from_pretrained(model_pth, config=config)

    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    training_args.data_dir = data_args.data_dir

    if training_args.analysis_type == "shap_value":
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()

        model = model.to(training_args.device)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            unlabeled_dataset=unlabeled_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )

        trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
        output = trainer.evaluate(eval_dataset=test_dataset)
        test_result = output.metrics

        test_predictions = output.predictions[0] #(n_data, n_class) # ex(trec test): (500,6)
        test_preds = np.argmax(test_predictions, axis=1) # (n_data,)
        test_label_ids = output.label_ids # (n_data,)

        ans_or_not = test_preds==test_label_ids
        lab_ans_or_not = list(zip(test_preds, ans_or_not)) #[(5, True), (4, False), (3, True)]
        
        lab_ans_idx = {}
        for lab in test_dataset.label_list:
            lab_ans_idx[lab] = []
        for i, x in enumerate(tqdm(lab_ans_or_not)):
            lab_id = x[0]
            ans_tf = x[1]
            if ans_tf:
                lab_ans_idx[lab_id].append(i)
        # lab_ans_idx: {0: [3, 8, 9, 18, 21, ..], 1: [66, 133, ...], ..., 5:[0, 4, 5, 12, ...]}

        print(f"\nlab_ans_idx: {lab_ans_idx}\n")

        data_collator = default_data_collator
        inf_dataloader = DataLoader(
            test_dataset,
            batch_size = training_args.inf_batch_size, 
            collate_fn = data_collator,
            num_workers = training_args.dataloader_num_workers
        )

        model.label_word_list = None

        model.eval()
        all_logits_list = []
        with torch.no_grad():
            for step, batch in enumerate(inf_dataloader):
                batch = {
                    "input_ids": batch["input_ids"].to(training_args.device),
                    "attention_mask": batch["attention_mask"].to(training_args.device),
                    "mask_pos": batch["mask_pos"].to(training_args.device),
                    "labels": batch["labels"].to(training_args.device)
                }
                outputs = model(input_ids = batch["input_ids"], attention_mask=batch["attention_mask"], mask_pos=batch["mask_pos"]) ### batch["labels"] 제외하고 입력값 투입
                all_logits_list.append(outputs[0])
        
        vocabs = list(tokenizer.get_vocab().keys())
        all_logits = torch.cat(all_logits_list)
        print(f"\nall test logits: {all_logits.shape}\n")
        
        shap_model = model.mav
        
        for lab_id, idx_list in lab_ans_idx.items():
            if len(idx_list) > 0:
                torch_data = all_logits[idx_list,:]
                print(f"\n>>> logits of label {lab_id}: {torch_data.shape}\n\n")
                
                explainer_shap = shap.DeepExplainer(shap_model, torch_data) 
                shap_values = explainer_shap.shap_values(torch_data)
                
                if lab_id < 10:
                    shap_img_dir = f"{shap_dir}/label0{lab_id}_shap_bar_{len(idx_list)}.png"
                else:
                    shap_img_dir = f"{shap_dir}/label{lab_id}_shap_bar_{len(idx_list)}.png"

                plt.figure()
                try:
                    fig1 = shap.summary_plot(shap_values[lab_id],torch_data.detach().cpu().numpy(), feature_names=vocabs, plot_type='bar')
                    plt.savefig(shap_img_dir)
                    plt.close()
                except:
                    plt.close()
                    print(f">>>>> label {lab_id} shap plotting error! -> pass to other label\n")


    if training_args.analysis_type == "tsne":
        
        assert model.label_word_list == None

        model = model.to(training_args.device)

        if training_args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        data_collator = default_data_collator
        inf_dataloader = DataLoader(
            inf_dataset,
            batch_size = training_args.inf_batch_size, 
            collate_fn = data_collator,
            num_workers = training_args.dataloader_num_workers
        )

        model.eval()
        all_mask_rep_list = []
        all_label_list = []

        with torch.no_grad():
            for step, batch in enumerate(inf_dataloader):
                batch = {
                    "input_ids": batch["input_ids"].to(training_args.device),
                    "attention_mask": batch["attention_mask"].to(training_args.device),
                    "mask_pos": batch["mask_pos"].to(training_args.device),
                    "labels": batch["labels"].to(training_args.device)
                }
                outputs = model(input_ids = batch["input_ids"], attention_mask=batch["attention_mask"], mask_pos=batch["mask_pos"])
                # all_probs: (bsz, vocab_size), sequence_mask_output: (bsz, hidden_size)
                all_mask_rep_list.append(outputs[1])
                all_label_list.append(batch["labels"])

        all_mask_rep = torch.cat(all_mask_rep_list).detach().cpu().numpy()
        all_label = torch.cat(all_label_list).detach().cpu().numpy()

        print(f"> all mask_rep: {all_mask_rep.shape}\n> all label: {all_label.shape}")

        tsne_mask_rep = TSNE(random_state=602).fit_transform(all_mask_rep)
        
        if training_args.zero_shot:
            tsne_vocab_logit_title = f"tsne_vocab_logit_{data_args.inf_data_type}_zero_shot"
            tsne_mask_rep_title = f"tsne_mask_rep_{data_args.inf_data_type}_zero_shot"
        else:
            tsne_vocab_logit_title = f"tsne_vocab_logit_{data_args.inf_data_type}_after_train"
            tsne_mask_rep_title = f"tsne_mask_rep_{data_args.inf_data_type}_after_train"

        if data_args.task_name == "trec":
            label2word = {0: 'ABBR', 1: 'ENTY', 2: 'DESC', 3: 'HUM', 4: 'LOC', 5: 'NUM'} # https://huggingface.co/datasets/trec

        draw_tsne_plot(tsne_mask_rep, all_label, label2word, tsne_dir, tsne_mask_rep_title, "png")
        draw_tsne_plot(tsne_mask_rep, all_label, label2word, tsne_dir, tsne_mask_rep_title, "pdf")
        
        # Calculate silhouette score
        kmeans = KMeans(n_clusters=num_labels)
        kmeans.fit(tsne_mask_rep)
        cluster_labels = kmeans.labels_

        sil_score = silhouette_score(tsne_mask_rep, cluster_labels, metric='euclidean')
        
        sil_txt = os.path.join(tsne_dir, f"{tsne_mask_rep_title}_silhouette.txt")
        with open(sil_txt, "w") as writer:
            print(f"\n>>> silhouette score of {tsne_mask_rep_title}: {sil_score}")
            writer.write(f"{sil_score}")



if __name__ == "__main__":
    main()
