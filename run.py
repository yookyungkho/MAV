# Main Reference
#   - Lm-BFF: https://github.com/princeton-nlp/LM-BFF
#   - SFLM: https://github.com/MatthewCYM/SFLM

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np

from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments

from src.dataset import FewShotDataset
from src.models import RobertaForPromptFinetuning
from src.trainer import Trainer
from src.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping
from src.utils import set_seed

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


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # Unify total train epoch, eval step
    eval_nums: int = field(
        default=20,
        metadata={"help": "total nums of evaluation during training"}
    )
    # --- For flexmatch(CPL; Curriculum Pseudo Labeling) ---
    is_cpl: bool = field(
        default=False,
        metadata={"help": "whether to use CPL(Curriculum Pseudo Labeling) or not"}
    )
    thresh_warmup: bool = field(
        default=False,
        metadata={"help": "whether to use threshold warmup or not"}
    )
    # ---

    # --- For baseline(sup) ---
    base_mode: str = field(
        default=None,
        metadata={"help": "'sup'(supervised) or 'ssl'(semi-supervised)"}
    )

    train_type: str = field(
        default=None,
        metadata={"help": "'full_train'(full) or 'train'(small)"}
    )
    # ---

    # --- For wandb logging ---
    wandb_project: str = field(
        default='SFLM',
        metadata={"help": "wandb project name"}
    )

    wandb_entity: str = field(
        default='text-ssl',
        metadata={"help": "wandb entity name"}
    )

    wandb_group: str = field(
        default=None,
        metadata={"help": "wandb group name"}
    )
    # ---

    save_at_last: bool = field(
        default=False,
        metadata={"help": "save the last checkpoint"}
    )

    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    # --- St Loss ---
    lam1: float = field(
        default=1,
        metadata={"help": "weight of self-training loss"}
    )

    use_st_loss: bool = field(
        default=False
    )

    st_loss_type: str = field(
        default="fix_sflm",
        metadata={"help": "vanilla or fix_sflm or flex_cpl"}
    )

    threshold: float = field(
        default=0.95,
        metadata={"help": "threshold of including self-training loss"}
    )
    # ---

    # --- MLM Loss ---
    lam2: float = field(
        default=1,
        metadata={"help": "weight of self-supervised loss"}
    )

    use_mlm_loss: bool = field(
        default=False
    )
    # ---

    # auxiliary loss - re-weight st_loss
    reweight: bool = field(
        default=False
    )

    # auxiliary loss - similarity loss
    sim_loss: str = field(
        default="none",
        metadata={"help": "type of similarity loss btw weak/strong representations"}
    ) # "cos"

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

    # --- random aug ---
    randaug: bool = field(
        default=False,
        metadata={"help": "Whether to use randaug"}
    )
    randaug_record_path: str = field(
        default=None,
        metadata={"help": "directory to save randaug_record_path"}
    )
    # ---

    # --- autoaug (DND) ---
    autoaug: bool = field(
        default=False,
        metadata={"help": "Whether to use autoaug"}
    )

    policy_temp: float = field(
        default=0.05,
        metadata={"help": "temperature for policy update"}
    )

    policy_lr: float = field(
        default=1e-3,
        metadata={"help": "learning rate for policy update"}
    )
    
    policy_update_step: int = field(
        default=1,
        metadata={"help": "update frequency of policy network"}
    )

    lambda_policy_task: float = field(
        default=1.0,
        metadata={"help": "learning rate for policy update"}
    )

    lambda_policy_sim: float = field(
        default=1.0,
        metadata={"help": "learning rate for policy update"}
    )
    # ---

    # --- parameter freeze ---
    lm_freeze: bool = field(
        default=False,
        metadata={"help": "whether to freeze parameters of lm head / model"}
    )

    freeze_type: str = field(
        default="lmhead",
        metadata={"help": "type of param freeze"}
    )
    # --- 



def main():
    # Load arguments
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    training_args.k = data_args.num_k

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Set additional arguments
    training_args.wandb_name = f"{training_args.output_dir.split('/')[1]}-{training_args.seed}"
    
    training_args.output_dir = f"{training_args.output_dir}/seed{training_args.seed}"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))


    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    config.mav_hidden_dim = model_args.mav_hidden_dim
    config.return_mask_rep = model_args.return_mask_rep
    config.soft_verb = model_args.soft_verb


    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print(f">>>>> [Dataset Info]")
    if training_args.base_mode == "sup":
        train_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode=training_args.train_type, base_mode=training_args.base_mode)
        )
        unlabeled_dataset = None
        print(f">>>>> train: {len(train_dataset)}, unlabeled: None")
    else:
        train_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="train")
        )
        unlabeled_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="unlabeled")
        )
        print(f">>>>> train: {len(train_dataset)}, unlabeled: {len(unlabeled_dataset)}")
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev")
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test")
        if training_args.do_predict
        else None
    )
    print(f">>>>> valid: {len(eval_dataset)}, test: {len(test_dataset)}")


    # Get Model
    model_fn = RobertaForPromptFinetuning

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()

    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    if training_args.lm_freeze:
        if training_args.freeze_type == "lmhead":
            for name, child in model.named_children():
                for param in child.parameters():
                    if name=='lm_head':
                        param.requires_grad = False
        elif training_args.freeze_type == "model":
            for name, child in model.named_children():
                for param in child.parameters():
                    if name == 'roberta':
                        param.requires_grad = False


    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions[0]
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            
            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    

    training_args.data_dir = data_args.data_dir
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        unlabeled_dataset=unlabeled_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )


    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)

        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
        # Reload the best checkpoint (for eval)
        model = model_fn.from_pretrained(training_args.output_dir)
        model = model.to(training_args.device)
        trainer.model = model
        
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()

        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer


    # Evaluation
    final_result = {
        'time': str(datetime.today()),
    }
    #   eval
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)
    #   test
    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test")
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

            test_results.update(test_result)

    
    return eval_results


if __name__ == "__main__":
    main()
