import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class MultilingualE5Large:
    def load_model(self):
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    
    def text_embedding(self, text, config={}):
        max_length = 512
        batch_dict = self.tokenizer(text, max_length=max_length, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings.tolist()
    
    def query_embedding(self, text, config={}):
        text = f'query: {text}'
        return self.text_embedding(text, config)
    
    def document_embedding(self, text, config={}):
        text = f'passage: {text}'
        return self.text_embedding(text, config)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

