## Deep Galerkin Method (DGM)
The purpose of this project was to gain a better understanding of deep neural networks (DNNs). Specifically, my goal is to use DNNs to solve high-dimensional partial differential equations (PDEs) as accurately as possible. While there are many networks that can do this, I decide to focus on the Deep Galerkin Method (DGM) as its architecture makes sense practically and the opportunities for experimentation seemed endless. 

Implementations of DGM are done in PyTorch and Tensorflow, although only the PyTorch version is used for training and experiments. For me, PyTorch felt more comfortable to use than TensorFlow, but I may rewrite the TensorFlow version when I get more familiar with TensorFlow 2. Only the loss function differs between the two implementation (discussed further in the writeup). 

## Directory

* PyTorch 
  * [Source](https://github.com/akbar-amin/DNN-Research/blob/main/DGM/torch) - code for the model, objective functions (losses), and training loop 
  * [Notebooks](https://github.com/akbar-amin/DNN-Research/tree/main/DGM/results) - Jupyter notebooks for visualizing experiment results and some examples of the model in use
  * [Visuals](https://github.com/akbar-amin/DNN-Research/tree/main/DGM/results/visuals) - individual metric plots from experiments
  * [Data](https://github.com/akbar-amin/DNN-Research/tree/main/DGM/data) - raw data by epoch from experiments

* Tensorflow
  * [Source](https://github.com/akbar-amin/DNN-Research/tree/main/DGM/tensorflow) - code for all of the model's assets and an attempt at a training loop 

## References 

1. [[1708.07469] DGM: A deep learning algorithm for solving partial differential equations](https://arxiv.org/abs/1708.07469)
2. [[1811.08782] Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning](https://arxiv.org/abs/1811.08782)
3. [[2005.04554] A comparison study of deep Galerkin method and deep Ritz method for elliptic problems with different boundary conditions](https://arxiv.org/abs/2005.04554)
