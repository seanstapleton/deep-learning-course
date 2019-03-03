import numpy as np
import torch
from torch.utils.data import Dataset

def read_data(path):
    f = open(path,'r')
    text = f.read()
    examples = [example.split(' ') for example in text.split('\n')[:-1]]
    labels = [int(line[0]) for line in examples]
    data = [line[1:] for line in examples]
    return data,np.array(labels)

def create_vocab(data):
    flatten = [w for line in data for w in line]
    unique = list(set(flatten))
    word2idx = {word: idx for idx,word in enumerate(unique)}
    return unique,word2idx

def create_bag_of_words(data, word2idx=None):
    if word2idx is None:
        raise Error('create_bag_of_words need a word2idx mapping!')
    bag_of_words = np.zeros((len(data), len(word2idx)))
    for line in range(len(data)):
        for word in data[line]:
            if word in word2idx:
                bag_of_words[line][word2idx[word]] += 1
    return bag_of_words

class ReviewsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, train=True, word2idx=None):
        """
        Args:
            path (string): Path to the text file with reviews and labels.
        """
        data, labels = read_data(path)
        self.labels = torch.tensor(labels)
        
        if train:
            vocab,word2idx = create_vocab(data)
        elif word2idx is None:
            raise Error('Vocab must be provided for non-training data in ReviewsDataset')
            
        self.word2idx = word2idx
        
        data = create_bag_of_words(data, word2idx=word2idx)
        self.data = torch.tensor(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.labels[idx]
    
class PlainReviewsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, train=True, word2idx=None):
        """
        Args:
            path (string): Path to the text file with reviews and labels.
        """
        data, labels = read_data(path)
        self.labels = torch.tensor(labels)
        
        if train:
            vocab,word2idx = create_vocab(data)
        elif word2idx is None:
            raise Error('Vocab must be provided for non-training data in ReviewsDataset')
            
        self.word2idx = word2idx
        
        data = create_bag_of_words(data, word2idx=word2idx)
        self.data = torch.tensor(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.labels[idx]