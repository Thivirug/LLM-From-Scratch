import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text_data, tokenizer, context_size, stride):
        # initialise lists for input and target ids
        self.input_ids = []
        self.target_ids = []

        # encode the text data
        token_ids = tokenizer.tokenize(text_data)

        # create input and target ids
        for i in range(0, len(token_ids) - context_size, stride):
            self.input_ids.append(
                torch.tensor(token_ids[i:i + context_size])
            )
            self.target_ids.append(
                torch.tensor(token_ids[i+1: i+context_size+1])
            )

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
class TextDataLoader():
    def __init__(self, text_data, tokenizer, context_size, stride, batch_size, shuffle, drop_last, num_workers):
        self.dataset = TextDataset(text_data, tokenizer, context_size, stride)
        self.batch_size = batch_size   
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers 

    def get_dataLoader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )