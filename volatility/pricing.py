import numpy as np 

NaN = None

class CrankNicolson:

    """ Crank-Nicolson Method for American Call options"""

    # Put Parameters (not implemented)
    #   
    # Boundaries:
    #   V_temp[:, 0] = K 
    #   V_temp[:, M] = 0
    # 
    # Payoff:
    #   np.maximum(K - np.arange(0, Smax + dS/2.0, dS), 0)
    #                   alternatively...
    #   np.maximum(np.multiply(np.arange(0, Smax + dS/2.0, dS), -1) - (K * -1), 0)

    def __init__(self, **kwargs):
        
        self.M = kwargs.pop("M", 100)
        self.N = kwargs.pop("N", 50)
        self.r = kwargs.pop("r", 0.)
        self.div = kwargs.pop("div", 0.)
        self.S_max = kwargs.pop("S_max", lambda S: 3 * S)

    def __call__(self, S, T, K, sigma, M = NaN, N = NaN, r = NaN, div = NaN, S_max = NaN):

        M = M or self.M 
        N = N or self.N 
        r = r or self.r 
        div = div or self.div 
        S_max = S_max or self.S_max


        # spot multiplier (default 3x)
        S = S_max(S)

        # time and price steps
        dt, dS = T/N, S/M

        # risk neutral probabilities alpha, beta, and omega
        P_mat = np.zeros((M - 1,))
        Q_mat = np.zeros((M - 1,))
        R_mat = np.zeros((M - 1,))
        
        for i in range(M - 1):
            P_mat[i] = .25 * sigma**2 * i**2 * dt - .25 * (r - div) * i * dt
            Q_mat[i] = -.5 * sigma**2 * i**2 * dt - .5 * (r - div) * dt
            R_mat[i] = .25 * sigma**2 * i**2 * dt + .25 * (r - div) * i * dt
        
        # computation matrices A and B 
        A_mat = np.diag(1 - Q_mat) + np.diag(-P_mat[1: M - 1], -1) + np.diag(-R_mat[0: M - 2], 1)
        B_mat = np.diag(1 + Q_mat) + np.diag(P_mat[1: M - 1], -1) + np.diag(R_mat[0: M - 2], 1)

        # boundary matrix
        V_temp = np.zeros((N + 1, M + 1))
        V_temp[:, 0] = 0
        V_temp[:, M] = [S * np.exp(-r * (N - i) * dt) for i in range(N + 1)]
        
        # call option payoff -> option value matrix
        V_temp[N, :] = np.maximum(np.arange(0, S + dS/2.0, dS) - K, 0)
        V_mat = np.matrix(np.array(V_temp))

        # backward loop (discounting)
        for i in range(N - 1, -1, -1):

            # used for computing matrix B  
            B_temp = np.zeros((M - 1, 1))
            B_temp[0] = (0.25 * sigma**2 * dt - 0.25 * (r - div) * dt)*(V_mat[i, 0]+ V_mat[i + 1, 0])
            B_temp[M - 2] = (0.25 * dt * (sigma**2 * (M - 1)**2 + (r - div) * (M - 1))) * (V_mat[i, M] + V_mat[i + 1, M])

            # solve at point V_mat(x, i)
            V = (np.linalg.inv(A_mat)) * (B_mat * V_mat[i + 1, 1:M].T + B_temp)
            V_mat[i, 1:M] = V.T 
            V_mat[i, :] = np.maximum(np.matrix(np.arange(0, S + dS/2.0, dS)) - K, V_mat[i, :])
        
        V = V_mat[0, int((M + 1)/2)]

        return V 

