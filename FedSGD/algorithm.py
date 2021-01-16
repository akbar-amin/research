#pylint: disable = unbalanced-tuple-unpacking

import time 
import copy 
import pathlib
import numpy as np
import pandas as pd

from sklearn.datasets import load_svmlight_file as loadsvm
from scipy.stats import logistic


from utils import assign, metrics

class FedAc:

    datadir = pathlib.Path("home/data/libsvm")
    outdir = pathlib.Path("home/projects/models/FedAC")

    def __init__(self, dataset, outfile):
        
        self.store = pd.HDFStore(str(self.outdir / outfile))

        self.X, self.Y = loadsvm(str(self.datadir / dataset))
        self.X = self.X.toarray()
        self.samples, self.features = self.X.shape 


    def sample(self, pool):

        indexes = np.random.choice(self.samples, self.batchsize * self.M, True)

        X_sample = self.X[indexes, :].reshape(self.batchsize, self.M, self.features)
        Y_sample = self.Y[indexes].reshape(self.batchsize, self.M)

        X_temp = np.sum(X_sample * pool, axis = -1)
        Y_temp = (-Y_sample * (1 - (logistic.cdf(Y_sample * X_temp))))
        pgrad = np.mean(Y_temp[:,:, np.newaxis] * X_sample, axis = 0) 

        return pgrad + self.decay * pool

    def loss(self, X, Y, weight):

        w_unif = -np.mean(np.log(logistic.cdf((X @ weight) * Y)))
        w_decay = 0.5 * self.decay * np.linalg.norm(weight) ** 2 
        
        return w_unif + w_decay

    def poploss(self, weight):
        return self.loss(self.X, self.Y, weight)

    def broadcast(self, pool):

        average = pool.mean(axis = 0)
        pool = np.repeat(average[np.newaxis, :], pool.shape[0], axis = 0)

        return pool 
    
    def run(self, seed = 0):
        
        sequence = pd.Series([], dtype = pd.StringDtype())
        np.random.seed(seed)

        w_com = np.random.randn(self.features)
        w_pool = np.repeat(w_com[np.newaxis, :], self.M, axis = 0)
        w_ag_pool = np.copy(w_pool)
    
        for count in range(self.T + 1):
            if count % self.K == 0:
                w_pool = self.broadcast(w_pool)
                w_ag_pool = self.broadcast(w_ag_pool)

                if count % self.record == 0:
                    poploss = self.poploss(w_ag_pool[0, :])
                    sequence.at[count] = poploss 
                    print(f"Epoch: {count:<5}  | Loss: {poploss:.4f}")

            w_md_pool = (1/self.beta) * w_pool + (1 - (1/self.beta)) * w_ag_pool
            w_md_grad_pool = self.sample(w_md_pool)
            w_ag_pool = w_md_pool - self.eta * w_md_grad_pool
            w_pool = (1 - (1/self.alpha)) * w_pool + (1/self.alpha) * w_md_pool - self.gamma * w_md_grad_pool

        return sequence


    def train(self, version, key, eta, decay, batchsize, M, K, T, record, **kwargs):

        self.batchsize = batchsize
        self.record = record 
        self.version = version
        self.decay = decay 
        self.eta = eta

        self.M = M 
        self.K = K 
        self.T = T  

        self.gamma = kwargs.pop("gamma", assign(self, "gamma"))
        self.alpha = kwargs.pop("alpha", assign(self, "alpha"))
        self.beta = kwargs.pop("beta", assign(self, "beta"))

        parameters = copy.deepcopy(metrics)
        parameters.update(kwargs)
        parameters.update({"gamma": self.gamma, "alpha": self.alpha, "beta": self.beta, 
                            "method": self.version, "batchsize": self.batchsize, "eta": self.eta, 
                            "decay": self.decay, "M": self.M, "K": self.K, "T": self.T, "record": self.record})

        print(parameters)

        start = time.time()
        sequence = self.run(**kwargs)
        stop = time.time() - start 

        parameters.update({"runtime": stop})
        self.store.put(key, sequence)
        self.store.get_storer(key).attrs.metadata = parameters
