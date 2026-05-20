import torch
import torch.nn as nn
from .transformer_model import SimpleSingleStreamTransformer

class ConditionedDiffusionModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_seq_len=256,
                 cond_dim=128):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.transformer = SimpleSingleStreamTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers, max_seq_len=max_seq_len+1
        )

    def forward(self, x, cond):
        # x: (B, L, d_model) 噪声序列
        # cond: (B, cond_dim) 条件向量
        cond_emb = self.cond_proj(cond).unsqueeze(1)   # (B, 1, d_model)
        # 拼接条件到序列开头
        x_with_cond = torch.cat([cond_emb, x], dim=1)  # (B, L+1, d_model)
        pred_video, pred_audio = self.transformer(x_with_cond)
        # 裁剪掉条件位置的预测
        pred_video = pred_video[:, 1:, :]
        pred_audio = pred_audio[:, 1:, :]
        return pred_video, pred_audio