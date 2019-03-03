import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class BagOfWordsClassifier(nn.Module):

    # Initialize the classifier
    def __init__(self, num_labels, vocab_size):
        super(BagOfWordsClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        
        z1 = self.linear(bow_vec)
        return torch.sigmoid(z1)