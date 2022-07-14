from torch import nn as nn
from transformers import AutoModel


class PretrainAiCodeModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.next_header = nn.Linear(hidden_size, 2)
        self.same_group_header = nn.Linear(hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        next_labels=None,
        same_group_labels=None,
    ):
        cls_hidden_state = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0][:, 0, :]
        next_logits = self.next_header(cls_hidden_state)
        same_group_logits = self.same_group_header(cls_hidden_state)
        output_dict = {
            'next_logits': next_logits,
            'same_group_logits': same_group_logits,
        }
        if next_labels is not None:
            loss_next = self.criterion(next_logits, next_labels)
            loss_same_group = self.criterion(
                same_group_logits, same_group_labels
            )
            loss = loss_next + loss_same_group
            output_dict.update(
                {
                    'loss': loss,
                    'loss_next': loss_next,
                    'loss_same_group': loss_same_group,
                }
            )
        return output_dict
