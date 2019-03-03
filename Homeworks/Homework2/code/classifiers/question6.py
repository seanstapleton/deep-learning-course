import torch
import torchtext
from torchtext import data
import spacy
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

## Load in text data
spacy_en = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer)
# LABEL = data.Field(sequential=False, use_vocab=False)
LABEL = data.LabelField(dtype=torch.float)
train_val_fields = [('Label', LABEL),('Text', TEXT)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Generate datasets
train_set, val_set, test_set = data.TabularDataset.splits(path='../data', 
    format='tsv', 
    train='train.tsv', 
    validation='dev.tsv',
    test='test.tsv',
    fields=train_val_fields, 
    skip_header=True)

unlabelled_fields = [('id', None),('Text', TEXT)]
unlabelled_set = data.TabularDataset(path='../data/unlabelled.tsv', 
    format='tsv', 
    fields=unlabelled_fields, 
    skip_header=True)

## Build vocabulary
TEXT.build_vocab(train_set, max_size=100000, vectors='glove.6B.100d')
LABEL.build_vocab(train_set)

## Create iterators
train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_set, val_set, test_set), sort_key=lambda x: len(x.Text),
        batch_size=64, device=device)

unlabelled_it = data.BucketIterator(
    dataset=unlabelled_set,
    batch_size=1,
    device=device,
    sort_key=lambda x: len(x.Text))

### Define Classifiers ###
class EmbeddingClassifier(nn.Module):

    def __init__(self, emb_dim, num_labels, vocab_size, pretrained_vocab=None):
        super(EmbeddingClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, num_labels)
        
        if pretrained_vocab:
            self.embedding.weight.data.copy_(pretrained_vocab.vectors)
        
    def forward(self, inputs):        
        z1 = self.embedding(inputs).permute(1,0,2)
        z2 = F.avg_pool2d(z1, (z1.shape[1], 1)).squeeze(1) 
        out = self.linear(z2)
        return torch.sigmoid(out)

class RNNClassifier(nn.Module):
    def __init__(self, emb_dim, num_labels, hidden_dim, vocab_size, pretrained_vocab=None):
        super(RNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_labels)
        
        if pretrained_vocab:
            self.embedding.weight.data.copy_(pretrained_vocab.vectors)
        
    def forward(self, x):
        
        z1 = self.embedding(x)        
        z2, h2 = self.rnn(z1)
        z3 = self.linear(h2.squeeze(0))
        
        return z3
    
class LSTMClassifier(nn.Module):
    def __init__(self, emb_dim, num_labels, hidden_dim, vocab_size, pretrained_vocab=None):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_labels)
        
        if pretrained_vocab:
            self.embedding.weight.data.copy_(pretrained_vocab.vectors)

    def forward(self, x, batch_size=None):
        z1 = self.embedding(x)
        z2, (h2, c2) = self.lstm(z1)
        z3 = self.linear(h2[-1])

        return z3
    
## Linear Helpers
def binary_accuracy(preds, y):
    correct = (preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator):
        
        optimizer.zero_grad()
                
        outputs = model(batch.Text).squeeze(1)
        _,preds = torch.max(outputs,1)
        loss = criterion(outputs, batch.Label)
        acc = binary_accuracy(preds, batch.Label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            outputs = model(batch.Text).squeeze(1)
            _,preds = torch.max(outputs, 1)
            loss = criterion(outputs, batch.Label)
            
            acc = binary_accuracy(preds, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

## RNN Helpers
def binary_accuracy_rnn(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train_rnn(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        outputs = model(batch.Text).squeeze(1)
        loss = criterion(outputs, batch.Label)
        acc = binary_accuracy_rnn(outputs, batch.Label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_rnn(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            outputs = model(batch.Text).squeeze(1)
            loss = criterion(outputs, batch.Label)
            
            acc = binary_accuracy_rnn(outputs, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

### BAG OF WORDS ###
def BagOfWords():
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
        return torch.tensor(bag_of_words).float()

    train_data, train_labels = read_data('../data/train.txt')
    dev_data, dev_labels = read_data('../data/dev.txt')
    test_data, test_labels = read_data('../data/test.txt')

    train_labels = torch.tensor(train_labels)
    dev_labels = torch.tensor(dev_labels)
    test_labels = torch.tensor(test_labels)
    
    vocab,word2idx = create_vocab(train_data)
    vocab_size = len(vocab)

    train_dataset = create_bag_of_words(train_data, word2idx=word2idx)
    dev_dataset = create_bag_of_words(dev_data, word2idx=word2idx)
    test_dataset = create_bag_of_words(test_data, word2idx=word2idx)
    
    class PlainDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx],self.labels[idx]
        
    class BagOfWordsClassifier(nn.Module):
        def __init__(self, num_labels, vocab_size):
            super(BagOfWordsClassifier, self).__init__()
            self.linear = nn.Linear(vocab_size, num_labels)

        def forward(self, bow_vec):
            z1 = self.linear(bow_vec)
            return torch.sigmoid(z1)
        
    def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # If there is no training happening
        if num_epochs == 0:
            model.eval()
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # statistics
                running_corrects += torch.sum(preds == labels.data)

            best_acc = running_corrects.double() / dataset_sizes['val']

        # Training for num_epochs steps
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        ####################################################################################
                        #                             END OF YOUR CODE                                     #
                        ####################################################################################

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    with open('model.pkl', 'wb') as f:
                        pickle.dump(best_model_wts, f)



        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        if num_epochs > 0:
            model.load_state_dict(best_model_wts)
        return model
    
    def train(device, dataloaders, dataset_sizes, vocab_size):
        model = BagOfWordsClassifier(2, vocab_size)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.05)

        # Train the model for 25 epochs
        print('Train the model')
        model = train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

        train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=0)
        return model
    
    def main():
        datasets = {}
        datasets['train'] = PlainDataset(train_dataset, train_labels)
        datasets['val'] = PlainDataset(dev_dataset, dev_labels)

        dataset_sizes = { x: len(datasets[x]) for x in ['train', 'val'] }

        dataloaders = {
            x: DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = train(device, dataloaders, dataset_sizes, vocab_size)
        return model
    
    model = main()
    return model

# run the model
BagOfWords()

### HELPER ###
def run_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_it, val_it, test_it = data.BucketIterator.splits(
            datasets=(train_set, val_set, test_set),
            batch_size=4, device=device,
            sort_key=lambda x: len(x.Text),
            repeat=False,
            shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    for epoch in range(25):
        train_loss, train_acc = train(model, train_it, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_it, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    test_loss, test_acc = evaluate(model, test_it, criterion)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    
    return model

# Embedding Model
model = EmbeddingClassifier(300, 2, len(TEXT.vocab))
run_model(model)

# Pretrained Embedding Model
model = EmbeddingClassifier(100, 2, len(TEXT.vocab), pretrained_vocab=TEXT.vocab)
run_model(model)

### HELPER ###
def run_rnn_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_it, val_it, test_it = data.BucketIterator.splits(
            datasets=(train_set, val_set, test_set),
            batch_size=64, device=device,
            sort_key=lambda x: len(x.Text),
            repeat=False,
            shuffle=True)
    
    ## define model
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
        
    for epoch in range(5):
        train_loss, train_acc = train_rnn(model, train_it, optimizer, criterion)
        valid_loss, valid_acc = evaluate_rnn(model, val_it, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    test_loss, test_acc = evaluate_rnn(model, test_it, criterion)
    
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    
    return model

# RNN Model
model = RNNClassifier(100, 1, 300, len(TEXT.vocab), pretrained_vocab=TEXT.vocab)
run_rnn_model(model)

# LSTM Model
model = LSTMClassifier(100, 1, 300, len(TEXT.vocab), pretrained_vocab=TEXT.vocab)
run_rnn_model(model)