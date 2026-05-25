import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SurprisalModel:
    def __init__(self, model_name="gpt2", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def word_surprisal(self, context, target_word):
        full = context.rstrip() + " " + target_word

        context_ids = self.tokenizer.encode(
            context, return_tensors="pt"
        ).to(self.device)

        full_ids = self.tokenizer.encode(
            full, return_tensors="pt"
        ).to(self.device)

        n_context = context_ids.shape[1]

        with torch.no_grad():
            logits = self.model(full_ids).logits[0]

        log_probs = torch.nn.functional.log_softmax(
            logits, dim=-1
        )

        total = 0.0

        for i in range(n_context, full_ids.shape[1]):
            token_id = full_ids[0, i].item()
            log_p = log_probs[i - 1, token_id].item()
            total += -log_p / math.log(2)

        return total

    def sentence_surprisal(self, sentence):
        ids = self.tokenizer.encode(
            sentence, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(ids).logits[0]

        log_probs = torch.nn.functional.log_softmax(
            logits, dim=-1
        )

        vals = []

        for i in range(1, ids.shape[1]):
            token_id = ids[0, i].item()
            log_p = log_probs[i - 1, token_id].item()
            vals.append(-log_p / math.log(2))

        if len(vals) == 0:
            return 0.0

        return sum(vals) / len(vals)

    def sentence_perplexity(self, sentence):
        return 2 ** self.sentence_surprisal(sentence)