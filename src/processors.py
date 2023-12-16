"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def __init__(self, task_name):
        self.task_name = task_name 

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
  
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_total_train_examples(self, data_dir):
        lab_df = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None)
        unlab_df = pd.read_csv(os.path.join(data_dir, "unlabeled.csv"), header=None)
        total_df = pd.concat([lab_df,unlab_df], ignore_index=True)
        print(f">>>>> labeled({lab_df.shape[0]}) and unlabeled({unlab_df.shape[0]}) data merged! => Full labeled data size: {total_df.shape[0]} (examples below)")
        
        return self._create_examples(total_df.values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # df.values.tolist()
        # [[5, 'How far is it from Denver to Aspen ?'],
        # [4, 'What county is Modesto , California in ?'],
        # [3, 'Who was Galileo ?'], ...
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")

    def get_shap_examples(self, data_dir, idx_list):
        df = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).loc[idx_list,:]
        # print(f"\nidx_list: {idx_list}")
        return self._create_examples(df.values.tolist(), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "unlabeled.csv"), header=None).values.tolist(), "unlabeled")

    def get_labels(self):
        """See base class."""
        """ag_news, dbpedia, yahoo_answers, yelp5"""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        elif self.task_name == "yahoo_answers":
            return list(range(10)) ##### added
        elif self.task_name == "ag_news":
            return list(range(4))
        elif self.task_name == "dbpedia":
            return list(range(14))
        elif self.task_name == "trec50":
            return list(range(22)) ##### added
        elif self.task_name == "emotion":
            return list(range(6)) ##### added
        elif self.task_name == "goemotions":
            return list(range(26)) ##### added
        elif self.task_name == "yelp5":
            return list(range(5))
        else:
            raise Exception("task_name not supported.")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name in ["ag_news", "dbpedia"]:
                examples.append(InputExample(guid=guid, text_a=line[1], text_b=line[2], label=line[0]))
            elif self.task_name in ['mr', 'sst-5', 'subj', 'trec', 'trec50', 'emotion', 'goemotions', 'cr', 'mpqa', "yahoo_answers", "yelp5"]:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples

def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}


# Add your task to the following mappings
processors_mapping = {
    "mnli": MnliProcessor(),
    "mnli-mm": MnliMismatchedProcessor(),
    "mrpc": MrpcProcessor(),
    "sst-2": Sst2Processor(),
    "qnli": QnliProcessor(),
    "rte": RteProcessor(),
    "snli": SnliProcessor(),
    "mr": TextClassificationProcessor("mr"),
    "sst-5": TextClassificationProcessor("sst-5"),
    "subj": TextClassificationProcessor("subj"),
    "trec": TextClassificationProcessor("trec"),
    "trec50": TextClassificationProcessor("trec50"),
    "emotion": TextClassificationProcessor("emotion"),
    "goemotions": TextClassificationProcessor("goemotions"),
    "cr": TextClassificationProcessor("cr"),
    "mpqa": TextClassificationProcessor("mpqa"),
    "ag_news": TextClassificationProcessor("ag_news"),
    "dbpedia": TextClassificationProcessor("dbpedia"),
    "yahoo_answers": TextClassificationProcessor("yahoo_answers"),
    "yelp5": TextClassificationProcessor("yelp5"),
}

num_labels_mapping = {
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "qnli": 2,
    "rte": 2,
    "snli": 3,
    "mr": 2,
    "sst-5": 5,
    "subj": 2,
    "trec": 6,
    "emotion": 6,
    "goemotions": 26,
    "trec50": 22,
    "cr": 2,
    "mpqa": 2,
    "ag_news": 4,
    "dbpedia": 14,
    "yahoo_answers": 10,
    "yelp5": 5
}

output_modes_mapping = {
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "qnli": "classification",
    "rte": "classification",
    "snli": "classification",
    "mr": "classification",
    "sst-5": "classification",
    "subj": "classification",
    "trec": "classification",
    "emotion": "classification",
    "goemotions": "classification",
    "trec50": "classification",
    "cr": "classification",
    "mpqa": "classification",
    "ag_news": "classification",
    "dbpedia": "classification",
    "yahoo_answers": "classification",
    "yelp5": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "mnli": glue_compute_metrics,
    "mnli-mm": glue_compute_metrics,
    "mrpc": glue_compute_metrics,
    "sst-2": glue_compute_metrics,
    "qnli": glue_compute_metrics,
    "rte": glue_compute_metrics,
    "snli": text_classification_metrics,
    "mr": text_classification_metrics,
    "sst-5": text_classification_metrics,
    "subj": text_classification_metrics,
    "trec": text_classification_metrics,
    "emotion": text_classification_metrics,
    "goemotions": text_classification_metrics,
    "trec50": text_classification_metrics,
    "cr": text_classification_metrics,
    "mpqa": text_classification_metrics,
    "ag_news": text_classification_metrics,
    "dbpedia": text_classification_metrics,
    "yahoo_answers": text_classification_metrics,
    "yelp5": text_classification_metrics,
}


# For regression task only: median
median_mapping = {
    "sts-b": 2.5
}

bound_mapping = {
    "sts-b": (0, 5)
}
