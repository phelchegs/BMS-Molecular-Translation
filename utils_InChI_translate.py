from tqdm.auto import tqdm
tqdm.pandas()
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.utils.data as data
import time
from nltk.translate.bleu_score import corpus_bleu
from InChI_preprocessing import crop_image
from Transformer_decoder import subsequent_mask
import random

# def subsequent_mask(dim):
#     attn_shape = (1, dim, dim)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).type(torch.uint8)
#     return subsequent_mask == 0
    
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None

def shape_after_crop(df):
    assert 'file_path' in set(df.columns), 'make sure file_path has been added to the df.'
    shape = []
    for i in tqdm(range(len(df))):
        image = cv2.imread(df.loc[i, 'file_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = crop_image(image)
        shape.append(max(image.shape))
    temp = max(shape)
    print('the lagest size of cropped image is {} in the full df.'.format(temp))
    return temp

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

class Train_Counter:
    step: int = 0 #NO. of batches in the current epoch
    samples: int = 0 #NO. of obs processed
    tokens: int = 0 #NO. of total tokens processed
    
    
def run_one_epoch(data_iter, model, loss_function, optimizer, scheduler, CFG, tokenizer, mode = 'train', train_state = Train_Counter(), start_idx = None):
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
        if i%100 == 1 and mode == 'validation':
            list_of_reference = []
            list_of_translation = []
            seqs = greedy_decode(model, batch.src, CFG.max_text_len, start_idx, CFG.device)
            for m in range(batch.tgt.shape[0]):
                list_of_translation.append(tokenizer.predict_cap_tokens(seqs[m, :].cpu().detach().numpy()))
                list_of_reference.append([tokenizer.predict_cap_tokens(batch.tgt[m, :].cpu().detach().numpy())])
            n = random.randint(0, batch.tgt.shape[0]-1)
            bleu_score = corpus_bleu(list_of_reference, list_of_translation)
            print('Batch {0:6d} in the current validation dataloader | Bleu score {1:6.2f}'.format(i, bleu_score*100))
            print('Randomly sampled reference InChI is {}\nthe corresponding InChI predicted is {}'.format(list_of_reference[n][0], list_of_translation[n]))
        if i%100 == 1 and mode == 'train':
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start
            print(
                (
                    "Batch %6d in the current epoch | Loss / Token: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, loss_node.data/batch.tokens, tokens/elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss_node
    return total_loss/total_tokens, train_state