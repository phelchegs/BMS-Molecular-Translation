import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import InChI_extra_image_gen
from InChI_preprocessing import get_aug

class Extra_inchi_save_ds(Dataset):
    #The goal of this dataset class is to save image files (.png) to the folder EXTRA_INCHI_IMAGE_DIR.
    
    def __init__(self, df, tokenizer, CFG, transform = None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.texts = df['InChI_text'].values
        self.inchis = df['InChI'].values
        self.transform = transform
        self.extra_path = CFG.extra_InChI_image_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        inchi = self.inchis[idx]
        #Unless we perform training on the extra InChI dataset, the code below will have fulfilled the goal of this dataset: create images based on extra InChIs, save
        #image dataset to certain route.
        InChI_extra_image_gen.extra_inchi_image(inchi, extra_inchi_image_path = f'{self.extra_path}/extra_{idx}.png')
        #whether having transforming or not depends on the scale of the training dataset. The preliminary feeling is that the training plus transforming plus the extra
        #InChI dataset are already enough for training. Transform will be assigned as None first.
        text = self.texts[idx]
        sequence = self.tokenizer.text_to_sequence(text)
        seq_length = int(len(sequence))
        return seq_length
    
class Dataset(Dataset):
    def __init__(self, df, tokenizer, CFG, get_aug):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.texts = df['InChI_text'].values
        self.max_len = CFG.max_text_len
        self.images = df['image'].values
        self.aug = get_aug
        
    def __len__(self):
        return len(self.df)
    
    #normalize images, pad sequences (captions), return images and sequences, index of texts.
    def __getitem__(self, idx):
        image = self.images[idx]
        image_aug = self.aug(image = image)['image']
        seq = torch.LongTensor(self.tokenizer.text_to_sequence(self.texts[idx]))
        seq = F.pad(seq, (0, self.max_len - len(seq)), 'constant', self.tokenizer.text2index['<pad>'])
        if torch.cuda.is_available():
            image_aug = image_aug.to('cuda')
            seq = seq.to('cuda')
        return image_aug, seq
    
class MolecularTranslationDataModule():
    def __init__(self, train_df, valid_df, tokenizer, CFG, get_aug):
        self.batch_size = CFG.batch_size 
        self.train = Dataset(train_df, tokenizer, CFG, get_aug)
        self.valid = Dataset(valid_df, tokenizer, CFG, get_aug)
        
    #Create dataloaders for train and valid datasets.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = self.batch_size)