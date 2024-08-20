import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTForImageClassification

from .net import *


class VisionTransformer(Net):
    def __init__(self, name: str = 'ViTForImageClassification', num_labels: int = 2, cluster: bool = False):
        super(VisionTransformer, self).__init__(name, cluster=cluster)
        model_id = 'google/vit-base-patch16-224'
        id2label = {0: 'unordered', 1: 'ordered'}
        label2id = {k: v for v, k in id2label.items()}
        self.vit = ViTForImageClassification.from_pretrained(model_id, num_channels=3,
                                                             ignore_mismatched_sizes=True,
                                                             id2label=id2label, label2id=label2id)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return F.sigmoid(outputs.logits)
