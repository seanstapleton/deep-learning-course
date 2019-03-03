import pickle
import numpy as np
import torch
from bow import BagOfWordsClassifier
from dataset import ReviewsDataset
from torch.utils.data import DataLoader

with open('model.pkl', 'rb') as f:
    model_params = pickle.load(f)
    
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

train_data, train_labels = read_data('../data/train.txt')
test_data, test_labels = read_data('../data/test.txt')

test_labels = torch.tensor(test_labels)

_,word2idx = create_vocab(train_data)
vocab_size = len(word2idx)
test_bow = create_bag_of_words(test_data, word2idx=word2idx)
test_bow = torch.tensor(test_bow).float()

model = BagOfWordsClassifier(2, vocab_size)
print(model.state_dict())
model.load_state_dict(model_params)
print(model.state_dict())
model.eval()

outputs = model(test_bow)
_, preds = torch.max(outputs, 1)

acc = np.sum(preds.numpy() == test_labels.numpy())/len(test_labels)
print('Test Accuracy:', acc)
