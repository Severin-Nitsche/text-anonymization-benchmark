from typing_extensions import TypedDict
import torch.nn.functional as F
from typing import List,Any
from transformers import LongformerModel
from tokenizers import Encoding
from torch import nn, argmax
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class Model(nn.Module):
    """
    Full fine=tuning of all Longofrmer's parameters, with a linear classification layer on top.
    """
    def __init__(self, model, num_labels):
        super().__init__()
        self._bert = LongformerModel.from_pretrained(model)

        for param in self._bert.parameters():
           param.requires_grad = True

        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, batch):
        b = self._bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_masks"]
        )
        pooler = b.last_hidden_state
        return self.classifier(pooler)

class InferenceModel(Model):
    """
    Model for inference (Slaps an argmax layer ontop of `Model`)
    """
    def __init__(self, model, num_labels):
        super().__init__(model, num_labels)
    def forward(self, batch):
        tensor = super().forward(batch).permute(0,2,1)
        return argmax(tensor,1)
