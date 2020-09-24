import torch
import torch.nn as nn
import transformers

class BertForSeqClf(nn.Module):

    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        config = transformers.BertConfig.from_pretrained(pretrained_model_name, 
                                            num_labels=num_labels)
        self.num_labels = num_labels
        self.bert = transformers.BertModel.from_pretrained(pretrained_model_name)

        self.classifier = nn.Linear(config.hidden_size, 
                                    num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids: torch.LongTensor, 
                attention_mask: torch.LongTensor, 
                token_type_ids: torch.LongTensor):
        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
