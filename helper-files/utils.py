import time
import torch
import torch.nn as nn
from Transformer_decoder import subsequent_mask
import torch.utils.data as data

def lr_scheduler(step, model_size, factor, warmup_step):
    #Zero raising to negative power is an error. Need to set step 0 to 1. Step 1 and 0 have the same lr_scheduler.
    if step == 0:
        step = 1
    lr_scheduler = factor*(model_size**(-0.5)*min(step**(-0.5), step*warmup_step**(-1.5)))
    return lr_scheduler

class Train_Counter:
    step: int = 0 #NO. of batches in the current epoch
    samples: int = 0 #NO. of obs processed
    tokens: int = 0 #NO. of total tokens processed

def run_one_epoch(images, data_iter, model, loss_function, optimizer, scheduler, mode = 'train', train_state = Train_Counter()):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens_counter = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        output = model.forward(images, batch.tgt, None, batch.tgt_mask)
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
                    "Batch %6d in the current epoch | Loss / Token: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, loss_node.data/batch.tokens, tokens/elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss_node
    return total_loss/total_tokens, train_state

# def loss_function(x, y, tokens):
#     loss = nn.NLLLoss()(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))/tokens
#     return loss.data*tokens, loss

class loss_function(nn.Module):
    #without label smoothing, just kldiv input and target, which present as log_softmax and one hot.
    def __init__(self, size, blank_padding):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction = "sum")
        self.padding = blank_padding
        self.size = size
        
    def forward(self, x, target):
        x, target = x.reshape(-1, x.size(-1)), target.reshape(-1)
        assert x.size(1) == self.size
        onehot_target = torch.zeros_like(x)
        onehot_target.scatter_(1, target.unsqueeze(1), 1.0)
        onehot_target[:, self.padding] = 0
        mask = torch.nonzero(target.data == self.padding)
        if mask.dim() > 0:
            onehot_target.index_fill_(0, mask.squeeze(), 0.0)
        return self.loss(x, onehot_target.clone().detach())
        
    
def greedy_decode(model, image, max_len, start_symbol, device):
    memory = model.encode(image)
    ys = torch.zeros(1, 1).fill_(start_symbol).type(torch.LongTensor)
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
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(ys.data).fill_(next_word)], dim=1
        )
    return ys

#generate data for a simple replication task.
def data_generator(V, batchsize, nbatches, device):
    for i in range(nbatches):
        temp = torch.randint(1, V, size = (batchsize, 10))
        temp[:, 0] = 1
        src = temp.requires_grad_(False).clone().detach()
        tgt = temp.requires_grad_(False).clone().detach()
        yield Batchify(src, tgt, 0, device)
        
class Batchify:
    def __init__(self, src, tgt, padding, device):
        self.src = src
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = self.make_mask(self.tgt, padding, device)
        self.tokens = (self.tgt_y != padding).sum()
        
    @staticmethod
    def make_mask(tgt, padding, dev):
        original_mask = subsequent_mask(tgt.size(-1))
        padding_mask = (tgt != padding).unsqueeze(-2)
        if torch.cuda.is_available():
            original_mask = original_mask.to(dev).clone().detach().requires_grad_(False)
            padding_mask = padding_mask.to(dev).clone().detach().requires_grad_(False)
        tgt_mask = padding_mask & original_mask
        return tgt_mask
        
def load_trained_model(model, path, device = 'cpu'):
    model.load_state_dict(torch.load(path, map_location = torch.device(device)))
    return model
