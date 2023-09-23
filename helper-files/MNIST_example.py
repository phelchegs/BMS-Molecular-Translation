from Transformer_decoder import subsequent_mask
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
import numpy as np
import torch.utils.data as data
import time

class MNISTData(data.Dataset):
    def __init__(self, images, labels, device, max_len = 8):
        self.images = images
        self.labels = labels 
        self.max_len = max_len
        self.device = device
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, ix):
        x = torch.tensor(self.images[ix]).float().view(1, 28, 28)
        # use 1 for start of sentence
        # use 2 for end of sentence
        y = torch.tensor([1] + self.labels[ix] + [2]).long()
        # use 0 for pad
        y = F.pad(y, (0, self.max_len - len(y)), 'constant', 0)
        if torch.cuda.is_available():
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
    
class MNISTDataModule():#(pl.LightningDataModule):

    def __init__(self, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.vocab = 'abcdefghijklmnopqrstuvwxyz'
        self.len_vocab = len(self.vocab) + 3
        self.device = device
        
    def label2spanish(self, ix):
        if ix == 0: return 'cero'
        if ix == 1: return 'uno'
        if ix == 2: return 'dos'
        if ix == 3: return 'tres'
        if ix == 4: return 'cuatro'
        if ix == 5: return 'cinco'
        if ix == 6: return 'seis'
        if ix == 7: return 'siete'
        if ix == 8: return 'ocho'
        if ix == 9: return 'nueve'
        
    def caption2ixs(self, caption):
        return [self.vocab.index(c) + 3 for c in caption]

    def ixs2caption(self, ixs):
        label = ''
        for ix in ixs:
            if ix == 2:
                break
            elif ix == 0:
                label += ' '
            elif ix == 1:
                continue
            else:
                label += self.vocab[ix - 3]
        return label
    
    def setup(self):#, stage = None):
        mnist = fetch_openml('mnist_784', version = 1)
        images, labels = mnist["data"].values, mnist["target"].values.astype(np.int)
        # generate captions
        captions = [self.label2spanish(ix) for ix in labels]
        vocab_encoded = [self.caption2ixs(caption) for caption in captions]
        # train / val splits
        X_train, X_test, y_train, y_test = images[:60000] / 255., images[60000:] / 255., vocab_encoded[:60000], vocab_encoded[60000:]
        self.train_ds = MNISTData(X_train, y_train, self.device)
        self.val_ds = MNISTData(X_test, y_test, self.device)
        
    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size = self.batch_size)
    

class Train_Counter:
    step: int = 0 #NO. of batches in the current epoch
    samples: int = 0 #NO. of obs processed
    tokens: int = 0 #NO. of total tokens processed
    
    
def run_one_epoch(data_iter, model, loss_function, optimizer, scheduler, mode = 'train', train_state = Train_Counter()):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens_counter = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        output = model.forward(batch.src, batch.tgt, None, batch.tgt_mask)
        prob = model.generator(output)
        loss_node = loss_function(prob, batch.tgt_y)
        if mode == 'train':
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.tgt.shape[0]
            train_state.tokens += batch.tokens
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            scheduler.step()
        total_loss += loss_node.data
        total_tokens += batch.tokens
        tokens += batch.tokens
        if i%40 == 1 and mode == 'train':
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start
            print(
                (
                    "Batch %5d in the current epoch | Loss / Token: %5.4f "
                    + "| Tokens / Sec: %7.2f | Learning Rate: %6.2e"
                )
                % (i, loss_node.data/batch.tokens, tokens/elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss_node
    return total_loss/total_tokens, train_state
    
def greedy_decode(model, image, max_len, start_symbol, device):
    memory = model.encode(image)
    ys = torch.zeros(image.size(0), 1).fill_(start_symbol).type(torch.LongTensor)
    if torch.cuda.is_available():
        ys = ys.to(device).clone().detach().requires_grad_(False)
    for i in range(max_len - 1):
        decode_mask = subsequent_mask(ys.size(1)).type_as(image.data).clone().detach().requires_grad_(False)
        if torch.cuda.is_available():
            decode_mask = decode_mask.to(device)
        output = model.decode(
            memory, ys, None, decode_mask
        )
        prob = model.generator(output[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat(
            [ys, next_word.unsqueeze(-1)], dim = 1
        )
    return ys
