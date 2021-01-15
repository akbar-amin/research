Pseudo-implementations of the DGM Neural Network in PyTorch and Tensorflow. 

## Deep Galerkin Method (DGM)

The idea behind DGM is to solve high-dimensional partial differential equations (PDEs) without getting tangled in mesh. 

DGM takes advantage of minibatch sampling, where time-space are randomly sampled within a function's domain. By processing small minibatches sequentially, the network can learn some differential function and avoid the computational bottleneck present with grid-based methods when trying to solve high-dimensional PDEs

The neural network architecture used is similar to a highway or long-term short-term (LSTM) network, where an input and some recurrent connection are run through a stack of layers containing multiple gates, which are triggered by a pair of nonlinear activations.

### Motivation

Placeholder

### References

1. Sirignano, J., Spiliopoulos, K., 2018. [DGM: A deep learning algorithm for solving partial differential equations](https://arxiv.org/pdf/1708.07469v5.pdf)
2. Al-Aradi, A., et al., 2018. [Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning](https://arxiv.org/pdf/1811.08782.pdf)
3. Chen, J., Du, R., Wu, K., 2020. [A Comparison Study of Deep Galerkin Method and Deep Ritz Method for Elliptic Problems with Different Boundary Conditions](https://arxiv.org/pdf/2005.04554.pdf)

### Other

Testing and training was done on [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) cloud GPU instances since my personal computer does not run very well. 
