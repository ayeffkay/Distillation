import numpy as np
import youtokentome as yttm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StudentDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 bpe_model: yttm.youtokentome.BPE, lower: bool=True, 
                 max_seq_len=50, device=torch.device('cpu'), 
                 teacher_logits: np.ndarray=None):
        self.X = X
        self.y = y
        self.teacher_logits = teacher_logits if teacher_logits is not None else np.zeros((len(X), ))
        self.bpe_model = bpe_model
        self.pad_idx = bpe_model.subword_to_id('PAD')
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.device = device
        
    def __len__(self):
        return len(self.X)
    
    def change_logits(self, teacher_logits: np.ndarray):
        self.teacher_logits = teacher_logits

    def collate_fn(self, batch: list):
        X, y, y_ = [], [], []
        
        for elem in batch:
            X.append(elem['input_ids'])
            y.append(elem['targets'])
            y_.append(elem['teacher_logits'])

        X_pad = pad_sequence(X, batch_first=True, 
                            padding_value=self.pad_idx)
        y = torch.LongTensor(torch.cat(y, dim=0))
        y_ = torch.FloatTensor(torch.cat(y_, dim=0))
        return {'input_ids': torch.LongTensor(X_pad), 
                'targets': y, 
                'teacher_logits': y_}

    def __getitem__(self, i):
        text = self.X[i]

        if self.lower:
            text = text.lower()

        input_ids = self.bpe_model.encode(text.lower(), 
                                          output_type=yttm.OutputType.ID)[:self.max_seq_len]

        return {'input_ids': torch.LongTensor(input_ids).to(self.device), 
                'targets': torch.LongTensor([self.y[i]]).to(self.device), 
                'teacher_logits': torch.FloatTensor([self.teacher_logits[i]]).to(self.device)
                }
