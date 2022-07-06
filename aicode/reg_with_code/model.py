from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class RegWithCodeModel(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.reg_head = nn.Linear(769, 1)
        self.criterion = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        md_ratios: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.encoder(input_ids, attention_mask)[0][:, 0, :]
        features = torch.cat((hidden_states, md_ratios.unsqueeze(1)), 1)
        pred_scores = self.reg_head(features)
        output = {
            'pred_scores': pred_scores,
        }
        if scores is not None:
            loss = self.criterion(pred_scores, scores)
            output['loss'] = loss
        return output
