import csv, os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from DGMnet import DGM, Minibatches

# References:
#   [1] Sirignano, J., Spiliopoulos, K., 2018. DGM: A deep learning algorithm for solving partial differential equations. https://arxiv.org/pdf/1708.07469v5.pdf
#   [2] Al-Aradi, A., 2018. Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning. https://arxiv.org/pdf/1811.08782.pdf
#   [3] Chen, J., Du, R., Wu, K., 2020. A Comparison Study of Deep Galerkin Method and Deep Ritz Method for Elliptic Problems with Different Boundary Conditions. https://arxiv.org/pdf/2005.04554.pdf



K = 1         # strike price 
T = 2         # time-to-expiration (years)
X0 = 1        # spot price
sigma = 0.55  # underlying volatility  
r = 0         # risk-free interest rate

outfile = "/projects/models/DGM"  # for saving the model


def AdamLearningSchedule(iteration):
    """ The Adam optimizer learning rate schedule found most effective in [1]. """

    if iteration <= 5000:
        LR = 1e-4 
    elif iteration <= 10000:
        LR = 5e-4
    elif iteration <= 20000:
        LR = 1e-5 
    elif iteration <= 30000:
        LR = 5e-6
    elif iteration <= 40000:
        LR = 1e-6
    elif iteration <= 45000:
        LR = 5e-7
    else:
        LR = 1e-7

    return LR 


def train(model, generator, optimizer, iterations, steps):
    """ Trains the DGM model based on the procedure described in [2].

        Args:
            model (tf.keras.Model): an instance of 'DGM' or compatible tensorflow/keras model
            generator (Iterator[tf.Tensor]): an instance of 'Minibatches' or compatible iterator
            optimizer (tf.optimizers.Optimizer): a tensorflow/keras optimizer 
            iterations (int): the total number of training cycles 
            steps (int): the number of SGD steps to take per training cycle 

        Note: The last two arguments are similar to tensorflow's 'epochs' and 'steps_per_epoch' except 
                a minibatch is repeated across all 'steps_per_epoch'

        Metrics:
            L1 (tf.Tensor): PDE loss
            L2 (tf.Tensor): Minimized L2 error
            L3 (tf.Tensor): Boundary condition Loss
            L4 (tf.Tensor): Terminal condition loss
            loss (tf.Tensor): sum of the losses described above

        For every iteration, a minibatch pair will be sampled.
        Then for every step, the minibatches will be run through the training procedure (i.e. take a descent step). 
    """

    losses = []
    
    for i in range(1, iterations + 1):

        t_boundary, x_boundary, t_terminal, x_terminal = next(generator)

        for _ in range(steps):
            
            trainables = model.trainable_variables

            # tapeSGD will take the SGD step 
            with tf.GradientTape() as tapeSGD:
                tapeSGD.watch(trainables)

                # tapeD1 and tapeD2 take the first and second derivatives 
                with tf.GradientTape() as tapeD2:
                    tapeD2.watch([t_boundary, x_boundary])

                    with tf.GradientTape(persistent = True) as tapeD1:
                        tapeD1.watch([t_boundary, x_boundary])

                        V = model([t_boundary, x_boundary]) # DGM option value
                        P = model([t_terminal, x_terminal]) # DGM option payoff
                        
                    V_t = tapeD1.gradient(V, t_boundary)
                    V_x = tapeD1.gradient(V, x_boundary)
                    
                V_xx = tapeD2.gradient(V_x, x_boundary) 
                V_diff = V_t + 0.5 * sigma**2 * x_boundary**2 * V_xx + r * x_boundary * V_x - r * V

                P_boundary = tf.nn.relu(K - x_boundary)
                L1 = tf.reduce_mean(tf.square(V_diff * (V - P_boundary)))

                V_diff_min = tf.nn.relu(V_diff)
                L2 = tf.reduce_mean(tf.square(V_diff_min))
 
                V_min = tf.nn.relu(-(V - P_boundary))
                L3 = tf.reduce_mean(tf.square(V_min))

                P_terminal = tf.nn.relu(K - x_terminal)
                L4 = tf.reduce_mean(tf.square(P - P_terminal))

                loss = L1 + L2 + L3 + L4 
            
            gradients = tapeSGD.gradient(loss, trainables)
            optimizer.apply_gradients(zip(gradients, trainables))

        print(f"Iteration: {i} | Loss: {float(loss):.4f} | L1: {float(L1):.4f} | L2: {float(L2):.4f} | L3: {float(L3):.4f} | L4: {float(L4):.4f}")
        
        losses.append(tuple(float(m) for m in [i, loss, L1, L2, L3, L4]))
        optimizer.learning_rate.assign(AdamLearningSchedule(i))

        if i % 25 == 0:           # save checkpoint every 25 iterations
            model.save(outfile)

            with open(outfile + "/losses.csv", "a+") as fp:
                writer = csv.writer(fp, delimiter=",", lineterminator="\n")
                writer.writerows(losses)
            
            losses.clear()
            print(f"\nSaved information for iterations {i - 25}-{i}\n")

    model.save(outfile)




generator = Minibatches(
    nBoundary = 5000, 
    nTerminal = 750, 
    dimensions = 20, 
    T_bounds = [1e-10, T], 
    X_bounds = [1e-10, X0 * 2.5]
)

optimizer = tf.optimizers.Adam(AdamLearningSchedule(0))

model = DGM(
    hidden = 3, 
    units = 50, 
    dimensions = 20, 
    active1 = "tanh", 
    active2 = "tanh", 
    active3 = "linear"
)

train(model = model, generator = generator, optimizer = optimizer, iterations = 50000, steps = 12)


# model = tf.keras.models.load_model(outfile, compile = False)  
