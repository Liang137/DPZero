"""Custom models for few-shot learning specific operations."""

from socket import ntohl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from typing import Callable, Dict, Optional, Union, List, Tuple
import random
from functools import partial


import logging
logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


def model_for_prompting_forward_dpzero(
    model,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    mask_pos=None,
    labels=None,
    sfc_input_ids=None,
    sfc_attention_mask=None,
    sfc_mask_pos=None,
    dpzero=False
):
    if sfc_input_ids is not None:
        with torch.no_grad():
            logits = model_for_prompting_forward_dpzero(model, input_ids=sfc_input_ids, attention_mask=sfc_attention_mask, mask_pos=sfc_mask_pos)[0]
        icl_sfc_bias = F.log_softmax(logits.detach().squeeze(0))

    if mask_pos is not None:
        mask_pos = mask_pos.squeeze()

    model_fn = model.get_model_fn() # this is just roberta model
    # Encode everything
    if token_type_ids is not None:
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    else:
        outputs = model_fn(
            input_ids,
            attention_mask=attention_mask,
        )

    # Get <mask> token representation
    sequence_output = outputs[0]
    if mask_pos is not None:
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
    else:
        sequence_mask_output = sequence_output[:,0] # <cls> representation
        # sequence_mask_output = sequence_output.mean(dim=1) # average representation

    if model.label_word_list is not None:
        # Logits over vocabulary tokens
        head_fn = model.get_lm_head_fn()
        prediction_mask_scores = head_fn(sequence_mask_output)

        # Exit early and only return mask logits.
        if model.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        # use MLM logit
        if model.model_args.use_task_word:
            vocab_logits = model.lm_head(sequence_mask_output)
            for _id in model.label_word_list:
                logits.append(vocab_logits[:, _id].unsqueeze(-1))
        # use learned linear head logit on top of task word representation (standard LM-BFF)
        else:
            for label_id in range(len(model.label_word_list)):
                logits.append(prediction_mask_scores[:, model.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if model.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity
    else:
        logits = model.classifier(sequence_mask_output)

    loss = None
    if labels is not None:
        if model.config.num_labels == 1:
            # Regression task
            if model.label_word_list is not None:
                labels = torch.stack([1 - (labels.view(-1) - model.lb) / (model.ub - model.lb), (labels.view(-1) - model.lb) / (model.ub - model.lb)], -1)
                loss = nn.KLDivLoss(log_target=True)(logits.view(-1, 2), labels)
            else:
                labels = (labels.float().view(-1) - model.lb) / (model.ub - model.lb)
                loss =  nn.MSELoss()(logits.view(-1), labels)
        else:
            if model.model_args.l2_loss:
                coords = torch.nn.functional.one_hot(labels.squeeze(), model.config.num_labels).float()
                loss =  nn.MSELoss()(logits.view(-1, logits.size(-1)), coords)
            else:
                if dpzero:
                    loss =  nn.CrossEntropyLoss(reduction='none')(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

    if hasattr(model, "lr_weight"):
        # Linear head
        logits = torch.matmul(F.softmax(logits, -1), model.lr_weight) 
    if hasattr(model, "lr_bias"):
        logits += model.lr_bias.unsqueeze(0)

    if model.model_args.sfc and hasattr(model, "sfc_bias"):
        logits = F.log_softmax(logits, -1) - model.sfc_bias
    if sfc_input_ids is not None:
        logits = F.log_softmax(logits, -1) - icl_sfc_bias

    output = (logits,)

    if model.model_args.use_task_word and model.num_labels == 1:
        # Regression output
        output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (model.ub - model.lb) + model.lb,)

    return ((loss,) + output) if loss is not None else output


class RobertaModelForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        logger.warn("By default for RoBERTa models the input embeddings and the output embeddings are NOT tied!!!!")
        self.num_labels = config.num_labels
        self.num_k = config.num_k
        self.model_type = config.model_type
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args, self.data_args, self.label_word_list = None, None, None

        # For regression
        self.lb, self.ub = 0.0, 1.0

        # For auto label search.
        self.return_full_softmax = None

        # forward function for dpzero and mezo
        self.fwd = partial(model_for_prompting_forward_dpzero, dpzero=config.dpzero)

    def tie_emb(self):
        output_embeddings = self.lm_head.decoder
        self._tie_or_clone_weights(output_embeddings, self.roberta.get_input_embeddings())

    def get_model_fn(self):
        return self.roberta

    def get_lm_head_fn(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        return self.fwd(self, input_ids, attention_mask, token_type_ids, mask_pos, labels)


MODEL_TYPES = {
    "roberta": RobertaModelForPromptFinetuning,
}
