import pandas as pd 
from pathlib import Path 
from sklearn import preprocessing

activations = ["hardsigmoid", "leakyrelu", "logsigmoid", "relu", "sigmoid", "tanh"]
columns = ["timestamp", "epoch", "error", "loss", "L1", "L2"]
dimensions = ["2D", "4D", "8D", "12D"]

def load_dimension_data(dim, col = "loss"):

    scalar = preprocessing.MinMaxScaler()
    
    results = []
    idx = columns.index(col)

    query = "loss_{}D_".format(dim)
    for item in Path("data").iterdir():
        if query in item.stem:
            
            activation = item.stem.replace(query, "")
            df = pd.read_csv(str(item), 
                                usecols = [idx], 
                                names = [activation], 
                                header = None, 
                                squeeze = True
                            )
            results.append(df)
    
    temp = pd.concat(results, axis = 1)
    scaled = scalar.fit_transform(temp)
    
    return pd.DataFrame(scaled, columns = temp.columns)


def load_activation_data(active, col = "loss"): 

    scalar = preprocessing.MinMaxScaler()
    
    results = []
    idx = columns.index(col)

    query = "_{}".format(active)
    for item in Path("data").iterdir():
        if query in item.stem:
            dim = item.stem.replace(query, "")
            dim = dim.replace("loss_", "")

            df = pd.read_csv(str(item), 
                                usecols = [idx], 
                                names = [dim], 
                                header = None, 
                                squeeze = True
                            )
            
            results.append(df)
    
    temp = pd.concat(results, axis = 1)
    temp = temp[dimensions]
    scaled = scalar.fit_transform(temp)
    
    return pd.DataFrame(scaled, columns = temp.columns) 