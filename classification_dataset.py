from torch.utils.data import Dataset
import numpy as np
import transformers
import torch
 
class ClassificationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                    tokenizer: transformers.tokenization_bert.BertTokenizer, 
                    max_len:int=50,
                    device: torch.device=torch.device('cpu')):
        super().__init__()
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i: int):
        item = self.tokenizer.encode_plus(self.X[i], 
                                        max_length=self.max_len, 
                                        padding='max_length',
                                        truncation=True)
        
        return {
            'input_ids': torch.LongTensor(item['input_ids']).to(self.device),
            'attention_mask': torch.LongTensor(item['attention_mask']).to(self.device),
            'token_type_ids': torch.LongTensor(item['token_type_ids']).to(self.device),
            'targets': torch.LongTensor([self.y[i]]).to(self.device)
        }
