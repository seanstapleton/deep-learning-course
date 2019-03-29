from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 400

        self.fc1  = nn.Linear(input_size + class_size, self.units)
        self.fc2 = nn.Linear(self.units, self.units)
        self.fc3 = nn.Linear(self.units, self.units)
        self.layer_mu = nn.Linear(self.units, latent_size)
        self.layer_logvar = nn.Linear(self.units, latent_size)
        
        self.fc4 = nn.Linear(latent_size + class_size, self.units)
        self.fc5 = nn.Linear(self.units, self.units)
        self.fc6 = nn.Linear(self.units, self.units)
        self.layer_output = nn.Linear(self.units, input_size)

    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor
    
        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar PyTorch Variable of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        
        x_cat = torch.cat([x, c], dim=1)
        z1 = self.fc1(x_cat)
        a1 = F.relu(z1)
        
        z2 = self.fc2(a1)
        a2 = F.relu(z2)
        
        z3 = self.fc3(a2)
        a3 = F.relu(z3)
        
        z_mu = self.layer_mu(a3)
        z_var = self.layer_logvar(a3)
        return z_mu, z_var


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c): # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor
    
        Inputs:
        - z: PyTorch Variable of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        """
        x_cat = torch.cat([z, c], dim=1)
        z4 = self.fc4(x_cat)
        a4 = F.relu(z4)
        
        z5 = self.fc5(a4)
        a5 = F.relu(z5)
        
        z6 = self.fc6(a5)
        a6 = F.relu(z6)
        
        z_out = self.layer_output(a6)
        x_hat = F.sigmoid(z_out)

        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model
    
        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch Variable of shape (batch_size, latent_size)
                  for the posterior logvar
        """
    
        mu, logvar = self.recognition_model(x.view(-1, 28*28), c)
        z = self.reparametrize(mu, logvar)
        x_hat = self.generation_model(z, c)
        return x_hat, mu, logvar


def to_var(x, use_cuda):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x


def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)


def train(epoch, model, train_loader, optimizer, num_classes, use_cuda):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data, use_cuda).view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lowerbound for conditional vae
    Note: We compute -lowerbound because we optimize the network by minimizing a loss

    Inputs:
    - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
    - x: PyTorch Variable of shape (batch_size, input_size) for the real data
    - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
    - logvar: PyTorch Variable of shape (batch_size, latent_size) for the posterior logvar
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the negative lowerbound.
    """
    first_loss = F.binary_cross_entropy(x_hat, x.view(-1, 28*28), size_average=False)
    D_kl = -(1/2)*torch.sum(1 + logvar - (mu**2) - logvar.exp())
    return (first_loss+D_kl)/x.shape[0]


def main():
    # Load MNIST dataset
    use_cuda = False
    input_size = 28 * 28
    units = 400
    batch_size = 32
    latent_size = 20 # z dim
    num_classes = 10
    num_epochs = 11
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

    model = CVAE(input_size, latent_size, num_classes)

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, num_epochs):
        train(epoch, model, train_loader, optimizer, num_classes, use_cuda)

    # Generate images with condition labels
    c = torch.eye(num_classes, num_classes) # [one hot labels for 0-9]
    c = to_var(c, use_cuda)
    z = to_var(torch.randn(num_classes, latent_size), use_cuda)
    samples = model.generation_model(z, c).data.cpu().numpy()

    fig = plt.figure(figsize=(10, 1))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show()


if __name__ == "__main__":
  main()

