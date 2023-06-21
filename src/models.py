"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
# Check original codes for each module in https://github.com/huggingface/transformers/blob/v3.4.0/src/transformers/modeling_roberta.py

import logging

from .model_utils import MAV, MaskClassificationHead


logger = logging.getLogger(__name__)


class RobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.return_mask_rep = config.return_mask_rep
        self.soft_verb = config.soft_verb

        self.roberta = RobertaModel(config)
        if self.soft_verb:
            self.classifier = MaskClassificationHead(config) # soft_verb(baseline)
        else:
            self.lm_head = RobertaLMHead(config)
            self.mav = MAV(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # to check whether Word Emb == LM Head decoder
        # print(f"\n >>>>> config.tie_word_embeddings: {config.tie_word_embeddings}")

    def bff_forward(self,
                    input_ids=None,
                    attention_mask=None,
                    inputs_embeds=None,
                    get_embeds=False,
                    mask_pos=None,
                    labels=None,
                    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze() # (bsz,)

        # Encode everything
        if get_embeds:
            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask
            )

        # Get [MASK] token representation
        sequence_output, pooled_output = outputs[:2] # (bsz, seq_len, hidden_size), (bsz, hidden_size)
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]  # (bsz, hidden_size)

        if self.soft_verb:
            logits = self.classifier(sequence_mask_output) # direct classification w/ mask representation
            all_probs = None
        else:
            # Logits over vocabulary tokens
            prediction_mask_scores = self.lm_head(sequence_mask_output)
            all_probs = F.softmax(prediction_mask_scores, dim=-1) # (bsz, vocab_size)
            
            logits = self.mav(prediction_mask_scores)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if self.label_word_list is None:
            if self.return_mask_rep:
                output = (all_probs, sequence_mask_output, logits)
                # all_probs: (bsz, vocab_size) (None - soft_verb), sequence_mask_output: (bsz, hidden_size)
            else:
                output = (all_probs,logits)
        else:
            output = (logits, pooled_output, sequence_output)

        return ((loss,) + output) if loss is not None else output

    def mlm_forward(self,
                    input_ids=None,
                    attention_mask=None,
                    labels=None,
                    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    def forward(self,
                input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                get_embeds=False,
                mask_pos=None,
                labels=None,
                mlm=False,
                data_idx=None,
                ):
        if attention_mask is None:
            attention_mask = (input_ids != 1).float() # roberta [Pad] token idx: 1

        if mlm:
            return self.mlm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            return self.bff_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                get_embeds=get_embeds,
                mask_pos=mask_pos,
                labels=labels
            )
