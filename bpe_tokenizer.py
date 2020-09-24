import youtokentome as yttm
import numpy as np

class BPETokenizer(object):

    def __init__(self, vocab_size: int=10000, 
                 train_fname: str='train_texts.txt', 
                 bpe_path: str=''):

        self.bpe_path = bpe_path if len(bpe_path) else 'yttm_bpe.bin'
        self.bpe_model = yttm.BPE(bpe_path) if len(bpe_path) else None
        self.train_fname = train_fname
        self.vocab_size = vocab_size

    def gen_txt(self, X: np.ndarray, lower: bool=True, 
                train_fname: str='train_texts.txt'):

            if len(train_fname):
                self.train_fname = train_fname

            with open(self.train_fname, 'w') as f:
                try:
                    for text in X:
                        if not text:
                            continue
                        if lower:
                            text = text.lower()
                        f.write(text + '\n')

                except:
                    print('Could not read file {}'.format(self.train_fname))

    def train(self, train_fname: str='', vocab_size: int=10000):

        if len(train_fname):
            self.train_fname = train_fname

        if vocab_size > 0:
            self.vocab_size = vocab_size

        yttm.BPE.train(data=self.train_fname, vocab_size=self.vocab_size, model=self.bpe_path)
        self.bpe_model = yttm.BPE(self.bpe_path)
