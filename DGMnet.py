import tensorflow as tf 


class Minibatches:

    def __init__(self, nBoundary, nTerminal, dimensions, T_bounds, X_bounds):

        """ Minibatch iterator for time and price points. 
        
        Args:
            nBoundary, nTerminal (float): minibatch-sizes for testing the boundary and terminal conditions, respectively 
            dimensions (int): the dimensionality of both minibatches
            T_bounds (tuple, list): lower and upper time boundaries; ex: [1e-10, 1]
            X_bounds (tuple, list): lower and upper price boundaries; ex: [1e-10, 75]
        
        Note: While points are created separately, pairs of points are stacked within the model. 

        Returns:
            A minibatch with time and price points sampled within the given boundaries, and 
                another minibatch containing time points sampled at T = 0 and price points 
                within the given boundaries 
        """
        
        self.nBoundary  = nBoundary
        self.nTerminal = nTerminal
        self.dimensions = dimensions

        self.T_min = T_bounds[0]
        self.T_max = T_bounds[1]
        self.X_min = X_bounds[0]
        self.X_max = X_bounds[1]

    def __iter__(self):
        return self 

    def __next__(self):
        
        t_boundary = tf.Variable(tf.random.uniform((self.nBoundary, self.dimensions), self.T_min, self.T_max))
        x_boundary = tf.Variable(tf.random.uniform((self.nBoundary, self.dimensions), self.X_min, self.X_max))

        t_terminal = tf.Variable(tf.ones((self.nTerminal, self.dimensions)))
        x_terminal = tf.Variable(tf.random.uniform((self.nTerminal, self.dimensions), self.X_min, self.X_max))

        return t_boundary, x_boundary, t_terminal, x_terminal

class HiddenDGM(tf.keras.layers.Layer):

    def __init__(self, units, dimensions, active1, active2, **kwargs):
        """ An itermediate LSTM-like layer used in the DGM neural network. 

        Args:
            units (int): the number of nodes in the layer (width)
            dimensions (int): expected dimensions of the input
            active1 (str): the base activation function for all sublayers except H
            active2 (str): the activation function for the H-sublayer
        
        """

        super(HiddenDGM, self).__init__(**kwargs)

        self.units = units 
        self.dimensions = dimensions 
        self.active1 =  tf.keras.activations.get(active1)
        self.active2 =  tf.keras.activations.get(active2) 

        self.Wz = self.add_weight("Wz", (self.units, self.units), initializer = "glorot_uniform")
        self.Wg = self.add_weight("Wg", (self.units, self.units), initializer = "glorot_uniform")
        self.Wr = self.add_weight("Wr", (self.units, self.units), initializer = "glorot_uniform")
        self.Wh = self.add_weight("Wh", (self.units, self.units), initializer = "glorot_uniform")

        self.Uz = self.add_weight("Uz", (self.dimensions, self.units), initializer = "glorot_uniform")
        self.Ug = self.add_weight("Ug", (self.dimensions, self.units), initializer = "glorot_uniform")
        self.Ur = self.add_weight("Ur", (self.dimensions, self.units), initializer = "glorot_uniform")
        self.Uh = self.add_weight("Uh", (self.dimensions, self.units), initializer = "glorot_uniform")

        self.bz = self.add_weight("bz", (self.units,), initializer = "zeros")
        self.bg = self.add_weight("bg", (self.units,), initializer = "zeros")
        self.br = self.add_weight("br", (self.units,), initializer = "zeros")
        self.bh = self.add_weight("bh", (self.units,), initializer = "zeros")

    def call(self, S, C):
        """ Runs previous and original tensors through a series of computations within each sublayer 

        Args: 
            S (tf.Tensor): the tensor being evaluated; contains the previous layer's results
            C (tf.Tensor): the original input tensor; contains points (t, x)
        
        Returns:
            S_new (tf.Tensor): the resulting tensor 
        """

        Z = self.active1((tf.matmul(C, self.Uz) + tf.matmul(S, self.Wz)) + self.bz) 
        G = self.active1((tf.matmul(C, self.Ug) + tf.matmul(S, self.Wg)) + self.bg) 
        R = self.active1((tf.matmul(C, self.Ur) + tf.matmul(S, self.Wr)) + self.br) 
        H = self.active2(tf.matmul(C, self.Uh) + tf.matmul(S * R, self.Wh) + self.bh)

        S_new = ((tf.ones_like(G) - G) * H) + Z * S
        
        return S_new


class DGM(tf.keras.Model):

    def __init__(self, hidden, units, dimensions, active1 = "tanh", active2 = "tanh", active3 = "linear"):
        """ Architecture of the DGM neural network. 
        
        Args:
            hidden (int): the number of 'HiddenDGM' layers 
            units (int): the number of nodes per hidden layer 
            dimensions (int): expected dimensions of the minibatch 
            active1 (str): activation for the initial layer and all HiddenDGM sublayers except the H-sublayer
            active2 (str): activation for the H-sublayer in every HiddenDGM layer
            active3 (str): activation for the final layer

        Note: The last activation is not required by the DGM architecture. Leave as "linear" for it to be pass-through.
        """

        super(DGM, self).__init__()

        self.dimensions = dimensions 
        self.hidden = hidden 
        self.units = units 
        self.active1 = active1
        self.active2 = active2
        self.active3 = active3


        self.input_layer = tf.keras.layers.Dense(self.units, self.active1, name = "Dense_Input")

        self.hidden_layers = [HiddenDGM(self.units, self.dimensions, self.active1, self.active2, 
                    name = "HiddenDGM_{}".format(i + 1)) for i in range(self.hidden)]

        self.output_layer = tf.keras.layers.Dense(1, self.active3, name = "Dense_Output")

    def call(self, inputs):
        """ Runs a minibatch of time and price points through the DGM network.
        
        Args:
            t, x (tf.Variable, tf.Tensor): time and price tensors with the same shape

        A tensor (S) is created with the original input (C) from the Dense input layer. 
        The tensor is individually cycled through every hidden layer, where its output becomes
            the next hidden layer's input.
        When the last hidden layer returns, its output is run through the Dense output layer, 
            and the resulting tensor (S_new) is returned. 
        """

        C = tf.stack(inputs, axis = 0)  
    
        S = self.input_layer(C)               # Get the initial "output", and then...

        for lstmlayer in self.hidden_layers:  # for every hidden layer, 
            S = lstmlayer(S, C)               # pipe the output back in as input,

        S_new = self.output_layer(S)          # and return the final output. 
        
        return S_new
