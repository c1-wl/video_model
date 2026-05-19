import torch
import torch.nn as nn

class SimpleSingleStreamTransformer(nn.Module):
    """极简单流Transformer，用于音视频联合建模"""
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.video_head = nn.Linear(d_model, d_model)
        self.audio_head = nn.Linear(d_model, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        out = self.transformer(x)
        video_out = self.video_head(out)
        audio_out = self.audio_head(out)
        return video_out, audio_out