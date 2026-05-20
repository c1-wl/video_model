import torch
import torch.nn as nn

class SimpleTextEncoder(nn.Module):
    """极简文本编码器，用于演示。实际使用建议换成 CLIP。"""
    def __init__(self, vocab_size=1000, embed_dim=128, output_dim=128, max_len=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * max_len, output_dim)
        self.max_len = max_len
        self.output_dim = output_dim

    def encode(self, text, max_len=None):
        """将字符串转换为固定长度的向量。
        注意：实际使用中应使用 tokenizer，这里为了演示，将字符ASCII码作为 token。
        """
        if max_len is None:
            max_len = self.max_len
        # 将每个字符转为整数 (简单演示，实际需要真实tokenizer)
        tokens = [ord(c) % self.embedding.num_embeddings for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, L)
        emb = self.embedding(tokens)  # (1, L, embed_dim)
        emb = emb.view(1, -1)         # (1, L*embed_dim)
        vec = self.fc(emb)            # (1, output_dim)
        return vec

    def forward(self, tokens):
        # tokens: (B, L)
        emb = self.embedding(tokens)          # (B, L, D_emb)
        emb = emb.view(emb.size(0), -1)       # (B, L*D_emb)
        return self.fc(emb)                   # (B, output_dim)


class CLIPTextEncoder:
    """使用CLIP模型，需要安装 transformers 和 torch"""
    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cpu'):
        from transformers import CLIPProcessor, CLIPModel
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, text, max_len=None):
        inputs = self.processor(text=text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)   # (1, 512)
        return text_features.cpu()