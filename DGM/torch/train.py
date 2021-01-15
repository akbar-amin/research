import torch
from model import *
from objectives import Objective

directory = "data"

start = 1     # start epoch
stop = 1250   # stop epoch 
epoch_steps = 5 # gradient steps per batch before reset

N = 5000      # batch size
M = 50        # layer width (nodes)
dim = 8       # input dimensions

A1 = "Sigmoid" # activations for all sublayers except H
A2 = "Sigmoid" # activation for H
Nb = 500       # boundary condition batch size
step = .0001   # diff. operator adjustment 
penalty = 400  # penalty multiplier

learning = .0001 # optimizer learning rate
decay = .01      # optimizer weight decay


def train():

    net = torch.nn.ModuleList([
        Linear(dim, M),
        DGMCell(M, dim, A1, A2),
        DGMCell(M, dim, A1, A2),
        DGMCell(M, dim, A1, A2),
        Linear(M, 1)
    ])

    DGM = initialize(net)

    objective = Objective(DGM, penalty, step, Nb)
    optimizer = torch.optim.Adam(DGM.parameters(), lr = learning, weight_decay = decay)

    for epoch in range(start, stop + 1):
        optimizer.zero_grad()
        inputs = minibatch(N, dim)
        inputs.requires_grad = True

        for _ in range(epoch_steps):
            loss = objective.loss(inputs, epoch)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            objective.verbose()

    save(directory, stop, DGM, optimizer, objective, identifier = "train")

if __name__ == "__main__":
    train()


