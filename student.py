import torch
import torch.nn as nn
import torch.nn.functional as F

class Student(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int=512, 
                 padding_idx: int=None, 
                 hidden_dim: int=512, n_layers: int=1, dropout_p: float=0., 
                 n_classes: int=1, bidirect: bool=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim // int(bidirect + 1)
        self.n_layers = n_layers
        self.bidirect = bidirect
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, embedding_dim, 
                                      padding_idx=padding_idx)
        self.rnn = nn.GRU(embedding_dim, self.hidden_dim, n_layers, 
                          batch_first=True, bidirectional=bidirect, 
                          dropout=dropout_p)
        self.fc = nn.Linear(self.hidden_dim, n_classes)

    # args only for using same CustomRunner as teacher
    def forward(self, input_ids: torch.LongTensor, *args):
        embeddings = self.embedding(input_ids)
        output, hid = self.rnn(embeddings)
        hid = hid.view(self.n_layers, int(self.bidirect + 1), -1, self.hidden_dim)
        # mean of hidden states by layers and directions
        hid_mean = F.dropout(torch.mean(hid, dim=(0, 1)), p=self.dropout_p)
        logits = self.fc(hid_mean)
        return logits
