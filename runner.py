import torch
from catalyst import dl
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

class CustomRunner(dl.Runner):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def to_class_num(self, logits: torch.Tensor):
        return torch.max(logits, 1, keepdims=True)[1]

    def predict_batch(self, batch: dict, evaluate: bool=True):
        loss = score = 0

        with torch.set_grad_enabled(self.state.is_train_loader):

            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            logits = self.model(batch['input_ids'].to(self.device), 
                               attention_mask, token_type_ids)
            if evaluate:
                teacher_logits = batch.get('teacher_logits', None)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.to(self.device)

                loss = self.criterion(logits, 
                                      batch['targets'].to(self.device), 
                                      teacher_logits)
                outputs = self.to_class_num(logits)
                # because of catalyst's f1 calculates smth very strange for multiclass :(
                score = f1_score(outputs.cpu().numpy(), 
                                 batch['targets'].cpu().numpy(), 
                                 average='macro')
        return logits, loss, score

    def predict_loader(self, loader: DataLoader, evaluate: bool=True):
            sum_loss = sum_score = 0
            n = len(loader)
            total_logits = []
            for batch in loader:
                logits, loss, score = self.predict_batch(batch, evaluate)
                sum_loss += loss.item()
                sum_score += score
                total_logits.append(logits)
            return torch.cat(total_logits, dim=0), sum_loss / n, sum_score / n

    def _handle_batch(self, batch: dict):
        logits, loss, score = self.predict_batch(batch)
        self.state.batch_metrics.update({'loss': loss, 
                                         'f1_score': score})
        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()
