import numpy as np 
import pandas as pd 
import pathlib

metrics = {"method": None, "batchsize": None, "seed": None, "eta": None,"decay": None, 
         "M": None, "K": None, "T": None, "record": None, "gamma": None, "alpha": None, "beta": None}


def assign(self, param):

    if self.version == 0:
        if param == "gamma": 
            return np.sqrt(self.eta/self.decay) 

        elif param == "alpha":
            return 1/(self.gamma * self.decay)

        elif param == "beta":
            return self.alpha + 1

    elif self.version == 1:
        if param == "gamma": 
            return max(self.eta, np.sqrt(self.eta/(self.decay * self.K))) 

        elif param == "alpha":
            return 1/(self.gamma * self.decay)

        elif param == "beta":
            return self.alpha + 1

    elif self.version == 2:
        if param == "gamma": 
            return max(self.eta, np.sqrt(self.eta/(self.decay * self.K))) 

        elif param == "alpha":
            return 3/(2 * self.gamma * self.decay) - (1/2)

        elif param == "beta":
            return (2 * self.alpha * self.alpha)/(self.alpha - 1)


def load(outfile, key):

    outdir = pathlib.Path("home/projects/models/FedAC")
    store = pd.HDFStore(str(outdir / outfile))

    metadata = store.get_storer(key).attrs.metadata
    data = store[key]

    return data, metadata


def load_by_metric(outfile, metric = "eta"):

    outdir = pathlib.Path("home/projects/models/FedAC/data")
    store = pd.HDFStore(str(outdir / outfile))

    temp = []

    if metric == "decay":
        fmtr = "{:.2f}"
    elif metric == "eta":
        fmtr = "{:.5f}"
        
    for item in store.keys():
        if metric in item:
            name = fmtr.format(float(store.get_storer(item).attrs.metadata[metric]))
            data = store[item]
            data.name = name
            temp.append(data)

    df = pd.concat(temp, axis = 1)         

    return df 

def load_runtimes(outfile, metric = "M2", submetric = "B"):

    outdir = pathlib.Path("home/projects/models/FedAC/data")
    store = pd.HDFStore(str(outdir / outfile))

    temp = {}

    query = "/SUSY_{}_{}".format(metric, submetric)
    for item in store.keys():
        if query in item:
            name = item.replace(query, "")
            runtime = "{:.5f}".format(float(store.get_storer(item).attrs.metadata["runtime"]))
            temp.update({name: runtime})

    df = pd.Series(temp)   
    df = df.astype(float) 
    df.index = df.index.map(int)    
    df = df.sort_index()

    return df
