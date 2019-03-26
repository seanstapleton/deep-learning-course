import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, H, conv_size, in_channels, out_channels, pretrained_vocab=None):
        super(CNN, self).__init__()
                
        self.embedding = nn.Embedding(vocab_size,H)
        self.conv1d = nn.Conv1d(in_channels, out_channels, conv_size)
        self.avgPool = nn.AvgPool2d((out_channels,H))
        self.linear = nn.Linear(out_channels, 2)
        
        if pretrained_vocab:
            self.embedding.weight.data.copy_(pretrained_vocab.vectors)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_conv1d = self.conv1d(h_embedding)
        h_pool = self.avgPool(h_conv1d).clamp(min=0)
        h_linear = self.linear(h_pool)
        logits = F.softmax(h_linear)
        return logits
