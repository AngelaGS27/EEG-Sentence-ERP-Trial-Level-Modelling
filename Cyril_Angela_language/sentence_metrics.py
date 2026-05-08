import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class SentenceMetrics:
    def __init__(self, model_name="bert-base-uncased", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, text):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state[0]

        vec = hidden.mean(dim=0).cpu().numpy()
        return vec

    def cosine(self, a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na == 0 or nb == 0:
            return 0.0

        return float(np.dot(a, b) / (na * nb))

    def context_target_similarity(self, context, target):
        emb1 = self.embed(context)
        emb2 = self.embed(target)

        return self.cosine(emb1, emb2)