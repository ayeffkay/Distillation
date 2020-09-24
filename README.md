Distillation **BERT** (from transformers library) with **RNN**. Scored with <img src="https://latex.codecogs.com/gif.latex?f_1" title="f_1" /></a>.

All experiments were conducted on [RuSentiment dataset](https://github.com/strawberrypie/rusentiment/).

**How to run the code?** Just run *distillation.ipynb* (the repository *.py* modules must be in the same directory). GPU is highly recommended ;)

**Files description**
* classification_dataset.py -- instance of torch.utils.data.Dataset with pretrained BERT tokenizer (used for teacher training)
* bert.py -- BERT model for sequence classification (teacher)
* bpe_tokenizer.py -- BPE tokens for student model
* student_dataset.py -- instance of torch.utils.data.Dataset with pretrained BPE tokenizer (used for student training)
* student.py -- student model for teacher distillation
* losses.py -- modified CrossEntropy losses (for training with temperature and soft targets)
* runner.py -- instance of catalyst.dl.Runner for convenient training and inference
* distillation.ipynb -- training, inference and scores
