# pylint: disable = not-callable, no-member

import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math 
import torch

from torch.optim.lr_scheduler import LambdaLR
from torch.linalg import norm 
from torch.nn import Linear, Sequential

from objectives import Objective


torch.set_default_tensor_type("torch.DoubleTensor")


class DGMCell(torch.nn.Module):

    def __init__(self, M, dim, A1, A2):
        """ 
        An itermediate GRU-like cell used in the DGM neural network

        Arguments:
            M (int): the number of nodes in the layer (width)
            dim (int): dimensions of the input
            A1 (str): the base activation function for all sublayers except H
            A2 (str): the activation function for the H-sublayer
        """

        super(DGMCell, self).__init__()

        self.A1 = getattr(torch.nn, A1)()
        self.A2 = getattr(torch.nn, A2)()
        
        self.L1 = Linear(M, M)
        self.L2 = Linear(dim, M, False)
        self.L3 = Linear(M, M)
        self.L4 = Linear(dim, M, False)
        self.L5 = Linear(M, M)
        self.L6 = Linear(dim, M, False)
        self.L7 = Linear(M, M)
        self.L8 = Linear(dim, M, False)

    def forward(self, inputs, state):
        """ 
        Runs a series of computations through sublayers 
        
        Arguments: 
            inputs (torch.Tensor): the original input tensor
            state (torch.Tensor): the previous layer's result tensor
        
        Returns:
            state (torch.Tensor): the resulting tensor 
        """
    
        Z = self.A1(self.L1(state) + self.L2(inputs))
        G = self.A1(self.L3(state) + self.L4(inputs))
        R = self.A1(self.L5(state) + self.L6(inputs))
        H = self.A2(self.L7(torch.mul(state, R)) + self.L8(inputs))
        
        state = torch.mul(torch.ones_like(G) - G, H) + torch.mul(state, Z)
        
        return state    
        
    def extra_repr(self):
        
        return "units={}, dim={}, A1={}, A2={}".format(
            self.M, self.dim, self.A1, self.A2
        )


def initialize(model):
    """ Initializes weights and biases for a model """

    for m in model.modules():
        if isinstance(m, Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    return model.cuda()


def minibatch(N, dim):
    """ Samples N points for a given dimension """

    batch = torch.Tensor([])
    while batch.size()[0] < N:
        sample = 2 * torch.rand(N, 2) - 1
        idx = torch.where(norm(sample, dim = 1) < 1)
        sample = sample[idx]
        batch = torch.cat([batch, sample])
    
    batch = batch[0:N,:]

    return batch.cuda()


def schedule(epoch):
    """ An ADAM optimizer learning schedule """

    if epoch <= 5000:     # 0.0001
        return 1e-4 
    elif epoch <= 10000:  # 0.0005
        return 5e-4
    elif epoch <= 20000:  # 0.00001
        return 1e-5 
    elif epoch <= 30000:  # 0.000005
        return 5e-6
    elif epoch <= 40000:  # 0.000001
        return 1e-6
    elif epoch <= 45000:  # 0.0000005
        return 5e-7
    elif epoch > 45000:   # 0.0000001
        return 1e-7


def save(directory, epoch, model, optimizer, objective, identifier):
    """ Saves model, optimizer, epoch, and loss states """

    torch.save({
        "epoch": epoch, 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": objective.records[-1]["loss"],
    }, directory + "/dgm_{}.pt".format(identifier))

    objective.save(directory + "/loss_{}.csv".format(identifier))


def load(directory, model, optimizer, identifier):
    """ Reinitializes saved objects from a checkpoint """

    checkpoint = torch.load(directory + "/dgm_{}.pt".format(identifier))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, model, optimizer, loss 
