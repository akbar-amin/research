# pylint: disable = not-callable, no-member

import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time, math, csv, torch
from torch.linalg import norm 

torch.set_default_tensor_type("torch.DoubleTensor")

class Objective:

    def __init__(self, model, penalty = 100.0, step = 0.0001, Nb = 100):
        """
        Holds objective parameters and calculates metrics for the DGM model 

        Arguments:
            model (nn.Sequential): the model to evaluate
            penalty (float): multiplier for the boundary condition
            step (float): step size for determining differential operator loss
            Nb (int): the number of boundary points to sample 
        """
        
        self.records = []

        self.model = model
        self.penalty = penalty
        self.step = step 
        self.Nb = Nb 

        self.N = 0
        self.dim = 0

    def loss(self, inputs, epoch):
        """ Manages loss and error metrics """

        self.N = inputs.size()[0]  
        self.dim = inputs.size()[-1]

        loss1 = self.diff_operator_loss(inputs)
        loss2 = self.boundary_condition_loss()
        total = loss1 + self.penalty * loss2 
        error = self.relative_error(inputs)

        results = {
            "ts": time.time(),
            "epoch": int(epoch),
            "error": float(error),
            "loss": float(total),
            "L1": float(loss1),
            "L2": float(loss2)
        }

        self.records.append(results)

        return total

    def forward(self, inputs):
        """ Sequentially calls the forward for each layer in a model """
        
        state = None 
        for idx, cell in enumerate(self.model):
            if idx == 0:
                state = cell(inputs)
            elif idx == len(self.model) - 1:
                state = state + cell(state)
            else:
                state = state + cell(inputs, state)

        return state 

    def exact(self, inputs):
        return torch.sin(math.pi/2 * (1 - norm(inputs, dim = 1))).reshape((inputs.size()[0], 1))

    def approximate(self, inputs):
        
        iex = self.exact(inputs)
        inr = norm(inputs, dim = 1)
        ipx =  1/4 * math.pi * torch.sin(math.pi/2 * (1 - inr)) + \
                        1/2 * math.pi * (inputs.size()[-1] - 1)/ \
                        inr * torch.cos(math.pi/2 * (1 - inr))
        
        return ipx.reshape([inputs.size()[0], 1]) + iex**3 

    def boundary_condition_loss(self):

        bc = (2 * torch.rand(self.Nb, self.dim) - 1).cuda()
        bc_norm = norm(bc, dim = 1).cuda()

        for i in range(self.dim):
            bc[:, i] = bc[:, i] / bc_norm
        
        L2 = (torch.sum((self.forward(bc) - self.exact(bc))**2)/self.Nb)

        return L2

    def diff_operator_loss(self, inputs):
        
        x = self.forward(inputs)
        x_temp = torch.zeros(self.N, self.dim).cuda()

        for i in range(self.dim):
            dx = torch.zeros(self.N, self.dim).cuda()
            dx[:, i] = torch.ones(self.N)
            x_temp[:, i] = (self.forward(inputs + self.step * dx) - 2 * self.forward(inputs) + \
                                self.forward(inputs - self.step * dx))[:, 0]/self.step**2 
        
        x_laplace = (torch.sum(x_temp, dim = 1)).reshape([self.N, 1])
        x_approx = self.approximate(inputs)

        L1 = torch.sum((-x_laplace + x**3 - x_approx)**2)/self.N 

        return L1 

    def relative_error(self, inputs):

        predict = self.forward(inputs)
        exact = self.exact(inputs)
        error = torch.sqrt(torch.sum((predict - exact)**2))/torch.sqrt(torch.sum(exact)**2)

        return error 

    def verbose(self):

        item = self.records[-1]
        print(f"Epoch: {item['epoch']:<5} | Error: {item['error']:<6.5f}  | Loss: {item['loss']:<6.5f} ")


    def save(self, path):
        
        fieldnames = ["ts", "epoch", "error", "loss", "L1", "L2"]

        with open(path, "a+", newline = "") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames)
            writer.writerows(self.records)

