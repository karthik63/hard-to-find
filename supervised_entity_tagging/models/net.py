import transformers
import torch.nn as nn
import torch

class EntityTagger(nn.Module):
    def __init__(self, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        # d_model = getattr(self.pretrained_lm.config, 'd_model', 768)
        self.linear_map = nn.Linear(d_model, nclass)
        self.crit = nn.CrossEntropyLoss()

    def compute_cross_entropy(self, logits, labels):
        mask = labels >= 0
        return self.crit(logits[mask], labels[mask])

    def forward(self, encodings:transformers.BatchEncoding, labels:torch.LongTensor):
        if 'distilbert' in self.model_name:
            encodings.pop('token_type_ids')
        encoded = self.pretrained_lm(**encodings)
        # print(encoded.last_hidden_state.shape, 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        # print(d_model)
        # print(n_class)
        outputs = self.linear_map(encoded.last_hidden_state)
        loss = self.compute_cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        preds[labels < 0] = labels[labels < 0]
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }
