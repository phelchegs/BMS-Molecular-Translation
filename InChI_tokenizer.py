class Tokenizer():
    
    def __init__(self):
        self.text2index = {}
        self.index2text = {}

    def __len__(self):
        return len(self.text2index)
    
    def gen_vocab_dict(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' ')) #make sure that 'InChI_text' is created from 'InChI' by re.
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, j in enumerate(vocab):
            self.text2index[j] = i
        self.index2text = {k[1]: k[0] for k in self.text2index.items()}
        
    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.text2index['<sos>'])
        for i in text.split(' '):
            sequence.append(self.text2index[i])
        sequence.append(self.text2index['<eos>'])
        return sequence
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        text = ''
        for i in sequence:
            text += self.index2text[i]
        return text
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts
        
    def predict_cap_tokens(self, sequence):
        caption = []
        for i in sequence:
            if i == self.text2index['<eos>']:
                break
            elif i == self.text2index['<pad>']:
                caption.append(' ')
            elif i == self.text2index['<sos>']:
                continue
            else:
                caption.append(self.index2text[i])
        return caption
    
    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.text2index['<eos>']:
                break
            elif i == self.text2index['<pad>']:
                caption += ' '
            elif i == self.text2index['<sos>']:
                continue
            else:
                caption += self.index2text[i]
        return caption
    
    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions